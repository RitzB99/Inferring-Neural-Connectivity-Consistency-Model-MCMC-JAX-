# Three-Neuron Calcium Simulation (2 presynaptic -> 1 postsynaptic)

import math
from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random
from pytensor.link.jax.dispatch import jax_funcify

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ModelParams:
    # Simulation timing
    sim_time: float = 300.0   # total time in ms
    step_dt: float = 0.1      # integration step size (ms)
    ca_bin: float = 1.0       # sampling window for calcium (ms)

    # LIF neuron dynamics (values in mV, ms, MΩ)
    tau_mem: float = 2.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    v_thresh: float = -55.0
    R_mem: float = 4.0

    # External input (used for presynaptic neurons 0 and 1)
    stim_freq: float = 10.0
    stim_amp: float = 3.0
    stim_offset: float = 1.5

    # Whether the synaptic effect is delayed by one step
    delayed: bool = True

    # Soft spike smoothing factor
    slope_mV: float = 0.5

    # Calcium and fluorescence parameters
    Kd: float = 200.0
    alpha: float = 1.0
    beta: float = 0.0
    Ca_base: float = 24.0
    Ca_jump: float = 80.0
    tau_ca: float = 0.200  # in seconds

def gen_input(cfg: ModelParams):
    steps = int(round(cfg.sim_time / cfg.step_dt))
    t_axis = jnp.arange(steps) * cfg.step_dt
    t_sec = t_axis * 1e-3

    stim0 = (
        cfg.stim_offset
        + cfg.stim_amp * jnp.sin(2 * jnp.pi * cfg.stim_freq * t_sec)
    )

    stim1 = (
        cfg.stim_offset
        + cfg.stim_amp * jnp.sin(2 * jnp.pi * (cfg.stim_freq * 6.0) * t_sec)
    )

    stim2 = jnp.zeros_like(stim0)

    return jnp.stack([stim0, stim1, stim2]), t_axis
    
"""
def gen_input(cfg: ModelParams):
    steps = int(round(cfg.sim_time / cfg.step_dt))
    t_axis = jnp.arange(steps) * cfg.step_dt
    t_sec = t_axis * 1e-3

    stim_base = cfg.stim_offset + cfg.stim_amp * jnp.sin(
        2 * jnp.pi * cfg.stim_freq * t_sec
    )

    half = steps // 2

    env0 = jnp.concatenate([jnp.ones(half), jnp.zeros(steps - half)])
    env1 = jnp.concatenate([jnp.zeros(half), jnp.ones(steps - half)])

    stim0 = stim_base * env0
    stim1 = stim_base * env1
    stim2 = jnp.zeros_like(stim0)

    return jnp.stack([stim0, stim1, stim2]), t_axis
"""

def run_lif_binary(weights: jnp.ndarray, cfg: ModelParams) -> jnp.ndarray:
    """
    Classic LIF network with 3 neurons, hard threshold and reset, outputs 0/1 spikes.

    Structure:
        Neuron 0  
                    -> Neuron 2 (postsynaptic)
        Neuron 1  

    weights: shape (2,) -> [w0_to_2, w1_to_2]
    """
    I_in, _ = gen_input(cfg)
    n_steps = I_in.shape[1]

    w0, w1 = weights[0], weights[1]

    def loop(carry, I_t):
        v, pre_prev = carry  # v: (3,), pre_prev: (2,) previous spikes from neurons 0 and 1
        dv = ((cfg.v_rest - v) + cfg.R_mem * I_t) * (cfg.step_dt / cfg.tau_mem)
        v_tmp = v + dv

        # synaptic input to neuron 2 from neurons 0 and 1
        syn_2 = w0 * pre_prev[0] + w1 * pre_prev[1]
        v_tmp = v_tmp + jnp.array([0.0, 0.0, syn_2])

        fired = v_tmp > cfg.v_thresh
        v_next = jnp.where(fired, cfg.v_reset, v_tmp)

        # store previous spikes for neurons 0 and 1
        pre_next = jnp.array([
            fired[0].astype(v_next.dtype),
            fired[1].astype(v_next.dtype),
        ])

        return (v_next, pre_next), fired.astype(jnp.int32)

    init_v = jnp.array([cfg.v_rest, cfg.v_rest, cfg.v_rest])
    init_pre = jnp.array([0.0, 0.0])
    (_, _), spike_log = lax.scan(loop, (init_v, init_pre), I_in.T)
    return spike_log.T  # (3, n_steps)


def run_lif_soft(weights: jnp.ndarray, cfg: ModelParams) -> jnp.ndarray:
    """
    Differentiable 3-neuron LIF with smooth spike outputs between 0 and 1.
    """
    I_in, _ = gen_input(cfg)
    kappa = cfg.slope_mV

    w0, w1 = weights[0], weights[1]
    sigmoid = lambda x: 0.5 * (1 + jnp.tanh(0.5 * x))

    def loop(carry, I_t):
        v, pre_prev = carry
        base = v + ((cfg.v_rest - v) + cfg.R_mem * I_t) * (cfg.step_dt / cfg.tau_mem)
        v0_tmp, v1_tmp, v2_tmp = base[0], base[1], base[2]

        # soft spikes from presynaptic neurons 0 and 1
        s0 = sigmoid((v0_tmp - cfg.v_thresh) / kappa)
        s1 = sigmoid((v1_tmp - cfg.v_thresh) / kappa)

        # Use current soft spikes if instant coupling, else use delayed spikes
        if cfg.delayed:
            drive0 = pre_prev[0]
            drive1 = pre_prev[1]
        else:
            drive0 = s0
            drive1 = s1

        # synaptic drive into neuron 2
        v2_tmp = v2_tmp + w0 * drive0 + w1 * drive1

        # soft spike from neuron 2
        s2 = sigmoid((v2_tmp - cfg.v_thresh) / kappa)

        out = jnp.array([s0, s1, s2])

        # reset towards resting potential based on spike strength
        v_next = jnp.array([
            v0_tmp - s0 * (v0_tmp - cfg.v_reset),
            v1_tmp - s1 * (v1_tmp - cfg.v_reset),
            v2_tmp - s2 * (v2_tmp - cfg.v_reset),
        ])

        pre_next = jnp.array([s0, s1])
        return (v_next, pre_next), out

    init_v = jnp.array([cfg.v_rest, cfg.v_rest, cfg.v_rest])
    init_pre = jnp.array([0.0, 0.0])
    (_, _), spikes = lax.scan(loop, (init_v, init_pre), I_in.T)
    return spikes.T  # (3, n_steps)

def convert_CF(spike_seq: jnp.ndarray, cfg: ModelParams):
    """
    Takes spike trains for 3 neurons, bins them, and generates calcium and fluorescence traces.

    spike_seq: (3, n_steps)
    Returns:
        C_all: (3, ca_steps)
        F_all: (3, ca_steps)
    """
    n_neurons, n_steps = spike_seq.shape
    assert n_neurons == 3

    ca_steps = int(cfg.sim_time / cfg.ca_bin)
    factor = int(round(cfg.ca_bin / cfg.step_dt))
    assert n_steps == ca_steps * factor

    spk_binned = spike_seq.reshape(3, ca_steps, factor).sum(-1)
    decay = jnp.exp(-(cfg.ca_bin * 1e-3) / cfg.tau_ca)

    def ca_track(spk):
        def loop(C_prev, s):
            C_t = decay * C_prev + (1 - decay) * cfg.Ca_base + cfg.Ca_jump * s
            F_t = cfg.alpha * (C_t / (C_t + cfg.Kd)) + cfg.beta
            return C_t, (C_t, F_t)

        _, (C_seq, F_seq) = lax.scan(loop, 0.0, spk)
        return C_seq, F_seq

    # vectorize over neurons
    C_all, F_all = jax.vmap(ca_track, in_axes=0, out_axes=(0, 0))(spk_binned)
    return C_all, F_all  # (3, ca_steps)


def simulate_pair(weights: jnp.ndarray, cfg: ModelParams):
    """
    Runs the soft LIF 3-neuron model and returns (C, F).

    weights: shape (2,) -> [w0_to_2, w1_to_2]
    """
    spk = run_lif_soft(weights, cfg)
    return convert_CF(spk, cfg)


simulate_pair_jit = jax.jit(simulate_pair, static_argnames=("cfg",))


def get_F(weights: jnp.ndarray, cfg: ModelParams, mode: Literal["hard", "soft"] = "soft"):
    """
    Convenience function: directly returns fluorescence for soft or hard spikes.
    Output shape: (3, ca_steps)

    weights: (2,) array [w0_to_2, w1_to_2]
    """
    if mode == "soft":
        spk = run_lif_soft(weights, cfg)
    else:
        spk = run_lif_binary(weights, cfg)

    _, F = convert_CF(spk, cfg)
    return F


get_F_jit = jax.jit(get_F, static_argnames=("cfg", "mode"))


@dataclass(frozen=True)
class NoiseParams:
    # measurement noise
    sigma_f: float = 4e-4
    gamma: float = 1e-4
    # calcium process noise
    sigma_c: float = 28.0
    # lower bound to avoid too small variances
    floor: float = 1e-3


def pred_sigma(F_est, C_est, cfg: ModelParams, noise: NoiseParams):
    """
    Computes expected noise level (sigma) for each data point.
    Works element-wise over (3, T) arrays.
    """
    meas = jnp.sqrt(noise.sigma_f**2 + noise.gamma * jnp.maximum(F_est, 0.0))

    if (noise.sigma_c > 0.0) and (C_est is not None):
        dF_dC = cfg.alpha * cfg.Kd / (C_est + cfg.Kd) ** 2
        sigma_proc = dF_dC * noise.sigma_c * jnp.sqrt(cfg.ca_bin * 1e-3)
    else:
        sigma_proc = 0.0

    return jnp.sqrt(jnp.maximum(noise.floor**2, meas**2 + jnp.square(sigma_proc)))


def noisy_F(F_clean, C_clean, cfg: ModelParams, noise: NoiseParams, key):
    """
    Adds Gaussian noise to fluorescence based on predicted sigma.
    Shapes preserved: input (3, T) -> output (3, T)
    """
    sigma = pred_sigma(F_clean, C_clean, cfg, noise)
    eps = random.normal(key, F_clean.shape, dtype=F_clean.dtype)
    return F_clean + sigma * eps, sigma


def make_dataset(
    w_vec: jnp.ndarray,   # shape (2,) -> [w0_to_2, w1_to_2]
    cfg: ModelParams,
    noise: NoiseParams | None = None,
    seed: int = 0,
    with_noise: bool = True,
    flatten: bool = False,
    with_sigma: bool = False,
):
    """
    Generates a single (x, y) pair:
        x: synaptic weights vector (2,)
        y: fluorescence trace of 3 neurons, shape:
           - (3, T_ca) if flatten=False
           - (3 * T_ca,) if flatten=True
    """
    C_clean, F_clean = simulate_pair_jit(w_vec, cfg)

    if with_noise and noise is not None:
        key = random.PRNGKey(seed)
        F_obs, sigma = noisy_F(F_clean, C_clean, cfg, noise, key)
    else:
        F_obs, sigma = F_clean, None

    out_y = F_obs.reshape(-1) if flatten else F_obs
    result = {"x": w_vec.astype(jnp.float64), "y": out_y}

    if with_sigma and (sigma is not None):
        result["sigma"] = sigma.reshape(-1) if flatten else sigma

    return result
