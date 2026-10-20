"""Trainable fly-brain chess agent (PyTorch backend)."""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    torch.sparse.check_sparse_tensor_invariants(False)
except Exception:
    pass

from run_pytorch import (
    MODEL_PARAMS,
    AlphaLIF,
    PoissonSpikeGenerator,
    get_hash_tables,
    get_weights,
)
from chess_moves import NUM_MOVES, move_to_index

DT = 0.1  # ms


class ChessBrain(nn.Module):
    def __init__(
        self,
        comp_path: str,
        conn_path: str,
        wt_dir: str,
        num_neurons: int,
        readout_size: int = 1024,
        stim_rate: float = 200.0,
        sim_ms: float = 50.0,
        device: str = "cpu",
        train_backbone: bool = False,
    ):
        super().__init__()
        self.device = device
        self.num_neurons = num_neurons
        self.readout_size = readout_size
        self.stim_rate = stim_rate
        self.sim_steps = max(1, int(sim_ms / DT))
        self.train_backbone = train_backbone

        flyid2i, i2flyid = get_hash_tables(comp_path)
        self.flyid2i = flyid2i
        self.i2flyid = i2flyid

        weights = get_weights(conn_path, comp_path, wt_dir, csr=True)
        weights = weights.to(device=device)
        self.num_neurons = weights.shape[0]

        # fixed sparse topology
        if weights.is_sparse and weights.layout == torch.sparse_csr:
            self.register_buffer("crow_indices", weights.crow_indices().clone())
            self.register_buffer("col_indices", weights.col_indices().clone())
        else:
            coo = weights.to_sparse_coo()
            self.register_buffer("crow_indices", None)
            self.register_buffer("col_indices", coo.indices()[1].clone())
            # store row indices separately for coo fallback
            self.register_buffer("row_indices", coo.indices()[0].clone())
        self.register_buffer("weight_size", torch.tensor(self.num_neurons, dtype=torch.long))

        vals = weights.values().detach().clone().float()
        self.weight_values = nn.Parameter(vals, requires_grad=train_backbone)

        # neuron model (batch=1)
        self.neurons = AlphaLIF(
            1, self.num_neurons, DT, MODEL_PARAMS,
            exc_indices=None, device=device,
        )
        self.poisson = PoissonSpikeGenerator(DT, MODEL_PARAMS["scalePoisson"], device=device)
        self.scale = MODEL_PARAMS["wScale"]

        # readout neurons: last readout_size neurons
        self.readout_indices = list(range(self.num_neurons - readout_size, self.num_neurons))
        self.readout = nn.Linear(readout_size, NUM_MOVES)
        self.value_head = nn.Linear(readout_size, 1)

        if device == "cuda":
            try:
                self._simulate = torch.compile(self._simulate, mode="reduce-overhead", dynamic=True)
            except Exception:
                pass

    def _build_weights(self) -> torch.Tensor:
        if self.crow_indices is not None:
            return torch.sparse_csr_tensor(
                self.crow_indices, self.col_indices, self.weight_values,
                size=(self.num_neurons, self.num_neurons), device=self.device,
            )
        # coo fallback
        indices = torch.stack([self.row_indices, self.col_indices], dim=0)
        return torch.sparse_coo_tensor(
            indices, self.weight_values,
            size=(self.num_neurons, self.num_neurons), device=self.device,
        )

    def simulate(self, rates: torch.Tensor) -> torch.Tensor:
        """Run brain for sim_steps; return readout spike counts (1, readout_size)."""
        rates = rates.to(self.device).float()
        if rates.dim() == 1:
            rates = rates.unsqueeze(0)
        weights = self._build_weights()

        if not self.train_backbone:
            with torch.no_grad():
                return self._simulate(rates, weights).detach()

        return self._simulate(rates, weights)

    def _simulate(self, rates: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        batch = rates.shape[0]
        conductance = torch.zeros(batch, self.num_neurons, device=self.device)
        delay_buffer = torch.zeros(
            batch, self.neurons.synapse.steps_delay + 1, self.num_neurons, device=self.device
        )
        spikes = torch.zeros(batch, self.num_neurons, device=self.device)
        v = torch.zeros(batch, self.num_neurons, device=self.device) + MODEL_PARAMS["v0"]
        refrac = self.neurons.refrac_steps.unsqueeze(0).repeat(batch, 1).float()
        readout_counts = torch.zeros(1, self.readout_size, device=self.device)
        for step in range(self.sim_steps):
            poisson_spikes = self.poisson(rates)
            voltage_stim = self.scale * poisson_spikes
            if weights.is_sparse:
                weighted = torch.sparse.mm(weights, spikes.t()).t()
            else:
                weighted = torch.matmul(spikes, weights.t())
            recurrent_input = self.scale * weighted
            conductance, delay_buffer, spikes, v, refrac = self.neurons(
                recurrent_input, voltage_stim, conductance, delay_buffer, spikes, v, refrac,
            )
            readout_counts = readout_counts + spikes[:, self.readout_indices]
            if step % 100 == 0 and torch.cuda.is_available():
                print(f"  sim step {step}/{self.sim_steps} "
                      f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
        return readout_counts

    def forward(self, rates: torch.Tensor):
        counts = self.simulate(rates)
        logits = self.readout(counts)
        value = self.value_head(counts)
        return logits, value

    def set_train_backbone(self, flag: bool):
        self.train_backbone = flag
        self.weight_values.requires_grad_(flag)


def load_checkpoint(path: str, device: str = "cpu") -> ChessBrain:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    agent = ChessBrain(
        comp_path=cfg["comp_path"],
        conn_path=cfg["conn_path"],
        wt_dir=cfg["wt_dir"],
        num_neurons=cfg.get("num_neurons", 0),
        readout_size=cfg.get("readout_size", 1024),
        stim_rate=cfg.get("stim_rate", 200.0),
        sim_ms=cfg.get("sim_ms", 50.0),
        device=device,
        train_backbone=cfg.get("train_backbone", False),
    )
    agent.load_state_dict(ckpt["model"])
    agent.to(device)
    return agent
