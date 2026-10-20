"""
PyTorch benchmark runner for the Drosophila brain model.
"""

import pandas as pd
import pyarrow  # noqa: F401
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from time import perf_counter as time
import traceback

DT = 0.1
MODEL_PARAMS = {
    'tauSyn': 5.0, 'tDelay': 1.8, 'v0': -52.0, 'vReset': -52.0,
    'vRest': -52.0, 'vThreshold': -45.0, 'tauMem': 20.0,
    'tRefrac': 2.2, 'scalePoisson': 250, 'wScale': 0.275,
}


class PoissonSpikeGenerator(nn.Module):
    def __init__(self, dt, scale, device='cpu'):
        super().__init__()
        self.prob_scale = dt / 1000.0
        self.scale = scale
        self.device = device

    def forward(self, rates, generator=None):
        return torch.bernoulli(rates * self.prob_scale, generator=generator) * self.scale


class AlphaSynapse(nn.Module):
    def __init__(self, batch, size, dt, params, device='cpu'):
        super().__init__()
        self.time_factor = dt / params['tauSyn']
        self.steps_delay = int(params['tDelay'] / dt)
        self.size = size
        self.device = device
        self.batch = batch

    def state_init(self):
        conductance = torch.zeros(self.batch, self.size, device=self.device)
        delay_buffer = torch.zeros(self.batch, self.steps_delay + 1, self.size, device=self.device)
        return conductance, delay_buffer

    def forward(self, input_, conductance, delay_buffer, refrac):
        conductance_new = conductance * (1 - self.time_factor) + delay_buffer[:, 0, :] * refrac
        delay_buffer = torch.roll(delay_buffer, shifts=-1, dims=1)
        delay_buffer[:, -1, :] = input_
        return conductance_new, delay_buffer


class LIFNeuron(nn.Module):
    def __init__(self, batch, size, dt, params, device='cpu'):
        super().__init__()
        self.size = size
        self.dt = dt
        self.tau_mem = params['tauMem']
        self.v_reset = params['vReset']
        self.v_rest = params['vRest']
        self.v_threshold = params['vThreshold']
        self.v_0 = params['v0']
        self.time_factor = dt / self.tau_mem
        self.spike_gradient = self.ATan.apply
        self.device = device
        self.batch = batch

    def state_init(self):
        v = torch.zeros(self.batch, self.size, device=self.device) + self.v_0
        spikes = torch.zeros(self.batch, self.size, device=self.device)
        return spikes, v

    def forward(self, conductance, voltage_stim, v):
        v = v + voltage_stim
        v = v + self.time_factor * (conductance - (v - self.v_rest))
        spike = self.spike_gradient(v - self.v_threshold)
        reset = ((v - self.v_reset) * spike).detach()
        v = v - reset
        return spike, v

    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v):
            spike = (v > 0).float()
            ctx.save_for_backward(v)
            return spike
        @staticmethod
        def backward(ctx, grad_output):
            (v,) = ctx.saved_tensors
            grad = 1 / (1 + (np.pi * v).pow_(2)) * grad_output
            return grad


class AlphaLIF(nn.Module):
    def __init__(self, batch, size, dt, params, exc_indices=None, device='cpu'):
        super().__init__()
        self.size = size
        self.synapse = AlphaSynapse(batch, size, dt, params, device=device)
        self.neuron = LIFNeuron(batch, size, dt, params, device=device)
        base_refrac = int(round(params['tRefrac'] / dt))
        self.refrac_steps = torch.full((size,), base_refrac, dtype=torch.long, device=device)
        if exc_indices is not None:
            self.refrac_steps[exc_indices] = 0

    def state_init(self):
        conductance, delay_buffer = self.synapse.state_init()
        spikes, v = self.neuron.state_init()
        refrac = self.refrac_steps.unsqueeze(0).repeat(self.neuron.batch, 1).float()
        return conductance, delay_buffer, spikes, v, refrac

    def forward(self, recurrent_input, voltage_stim, conductance, delay_buffer, spikes, v, refrac):
        refrac = torch.where(spikes > 0, torch.zeros_like(refrac), refrac + 1)
        conductance_new, delay_buffer = self.synapse(
            recurrent_input, conductance, delay_buffer,
            (refrac >= self.refrac_steps.unsqueeze(0)).float()
        )
        spikes, v_new = self.neuron(conductance, voltage_stim, v)
        conductance_reset = (conductance_new * spikes).detach()
        conductance_new = conductance_new - conductance_reset
        return conductance_new, delay_buffer, spikes, v_new, refrac


def get_hash_tables(comp_path):
    df_comp = pd.read_csv(comp_path, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {j: i for i, j in flyid2i.items()}
    return flyid2i, i2flyid


def get_weights(conn_path, comp_path, wt_dir, csr=True):
    wt_dir = Path(wt_dir)
    coo_path = wt_dir / 'weight_coo.pkl'
    csr_path = wt_dir / 'weight_csr.pkl'
    data_conn = pd.read_parquet(conn_path)
    data_name = pd.read_csv(comp_path)
    num_neurons = data_name.shape[0]
    try:
        with open(coo_path, 'rb') as f:
            weight_coo = pickle.load(f)
    except FileNotFoundError:
        idx = [data_conn['Postsynaptic_Index'].to_list(), data_conn['Presynaptic_Index'].to_list()]
        val = data_conn['Excitatory x Connectivity'].to_list()
        weight_coo = torch.sparse_coo_tensor(idx, val, (num_neurons, num_neurons)).to(torch.float32)
        with open(coo_path, 'wb') as f:
            pickle.dump(weight_coo, f)
    if csr:
        try:
            with open(csr_path, 'rb') as f:
                weight_csr = pickle.load(f)
        except FileNotFoundError:
            weight_csr = weight_coo.to_sparse_csr()
            with open(csr_path, 'wb') as f:
                pickle.dump(weight_csr, f)
        return weight_csr
    return weight_coo
