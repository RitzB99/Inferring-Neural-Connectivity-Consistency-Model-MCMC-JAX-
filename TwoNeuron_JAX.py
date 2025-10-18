# Two-Neuron Calcium Simulation 

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

    # External input to neuron 0
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
    """Generates the sinusoidal stimulus for neuron 0 and keeps neuron 1 silent."""
    steps = int(round(cfg.sim_time / cfg.step_dt))
    t_axis = jnp.arange(steps) * cfg.step_dt
    t_sec = t_axis * 1e-3
    stim0 = cfg.stim_offset + cfg.stim_amp * jnp.sin(2 * jnp.pi * cfg.stim_freq * t_sec)
    stim1 = jnp.zeros_like(stim0)
    return jnp.stack([stim0, stim1]), t_axis


def run_lif_binary(weight: float, cfg: ModelParams) -> jnp.ndarray:
    """Classic LIF neuron with a hard threshold and reset, outputs 0/1 spikes."""
    I_in, _ = gen_input(cfg)
    n_steps = I_in.shape[1]

    def loop(carry, I_t):
        v, pre_prev = carry
        dv = ((cfg.v_rest - v) + cfg.R_mem * I_t) * (cfg.step_dt / cfg.tau_mem)
        v_tmp = v + dv
        v_tmp = v_tmp + jnp.array([0.0, weight]) * pre_prev

        fired = v_tmp > cfg.v_thresh
        v_next = jnp.where(fired, cfg.v_reset, v_tmp)
        pre_next = fired[0].astype(v_next.dtype)
        return (v_next, pre_next), fired.astype(jnp.int32)

    init_v = jnp.array([cfg.v_rest, cfg.v_rest])
    (_, _), spike_log = lax.scan(loop, (init_v, jnp.array(0.0)), I_in.T)
    return spike_log.T


def run_lif_soft(weight: float, cfg: ModelParams) -> jnp.ndarray:
    """Differentiable version of LIF with smooth spike outputs between 0 and 1."""
    I_in, _ = gen_input(cfg)
    kappa = cfg.slope_mV
    use_delay = 1.0 if cfg.delayed else 0.0
    use_inst = 1.0 - use_delay

    sigmoid = lambda x: 0.5 * (1 + jnp.tanh(0.5 * x))

    def loop(carry, I_t):
        v, pre_prev = carry
        base = v + ((cfg.v_rest - v) + cfg.R_mem * I_t) * (cfg.step_dt / cfg.tau_mem)
        v0_tmp, v1_tmp = base[0], base[1]

        # soft spike from neuron 0
        s0 = sigmoid((v0_tmp - cfg.v_thresh) / kappa)

        # apply synaptic input to neuron 1
        v1_tmp = v1_tmp + weight * (use_delay * pre_prev + use_inst * s0)

        # soft spike from neuron 1
        s1 = sigmoid((v1_tmp - cfg.v_thresh) / kappa)
        out = jnp.array([s0, s1])

        # reset voltage towards resting potential
        v_next = jnp.array([
            v0_tmp - s0 * (v0_tmp - cfg.v_reset),
            v1_tmp - s1 * (v1_tmp - cfg.v_reset),
        ])
        return (v_next, s0), out

    init_v = jnp.array([cfg.v_rest, cfg.v_rest])
    (_, _), spikes = lax.scan(loop, (init_v, jnp.array(0.0)), I_in.T)
    return spikes.T


def convert_CF(spike_seq: jnp.ndarray, cfg: ModelParams):
    """Takes spike trains, bins them, and generates calcium and fluorescence traces."""
    n_steps = spike_seq.shape[1]
    ca_steps = int(cfg.sim_time / cfg.ca_bin)
    factor = int(round(cfg.ca_bin / cfg.step_dt))
    assert n_steps == ca_steps * factor

    spk_binned = spike_seq.reshape(2, ca_steps, factor).sum(-1)
    decay = jnp.exp(-(cfg.ca_bin * 1e-3) / cfg.tau_ca)

    def ca_track(spk):
        def loop(C_prev, s):
            C_t = decay * C_prev + (1 - decay) * cfg.Ca_base + cfg.Ca_jump * s
            F_t = cfg.alpha * (C_t / (C_t + cfg.Kd)) + cfg.beta
            return C_t, (C_t, F_t)
        _, (C_seq, F_seq) = lax.scan(loop, 0.0, spk)
        return C_seq, F_seq

    C0, F0 = ca_track(spk_binned[0])
    C1, F1 = ca_track(spk_binned[1])
    return jnp.stack([C0, C1]), jnp.stack([F0, F1])


def simulate_pair(weight: float, cfg: ModelParams):
    """Runs the soft LIF model and returns calcium and fluorescence."""
    spk = run_lif_soft(weight, cfg)
    return convert_CF(spk, cfg)

simulate_pair_jit = jax.jit(simulate_pair, static_argnames=("cfg",))


def get_F(weight: float, cfg: ModelParams, mode: Literal["hard","soft"]="soft"):
    """Convenience function: directly returns fluorescence for soft or hard spikes."""
    spk = run_lif_soft(weight, cfg) if mode=="soft" else run_lif_binary(weight, cfg)
    _, F = convert_CF(spk, cfg)
    return F

get_F_jit = jax.jit(get_F, static_argnames=("cfg","mode"))


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
    """Computes expected noise level (sigma) for each data point."""
    meas = jnp.sqrt(noise.sigma_f**2 + noise.gamma * jnp.maximum(F_est, 0.0))
    if (noise.sigma_c > 0.0) and (C_est is not None):
        dF_dC = cfg.alpha * cfg.Kd / (C_est + cfg.Kd)**2
        sigma_proc = dF_dC * noise.sigma_c * jnp.sqrt(cfg.ca_bin * 1e-3)
    else:
        sigma_proc = 0.0
    return jnp.sqrt(jnp.maximum(noise.floor**2, meas**2 + jnp.square(sigma_proc)))


def noisy_F(F_clean, C_clean, cfg: ModelParams, noise: NoiseParams, key):
    """Adds Gaussian noise to fluorescence based on predicted sigma."""
    sigma = pred_sigma(F_clean, C_clean, cfg, noise)
    eps = random.normal(key, F_clean.shape, dtype=F_clean.dtype)
    return F_clean + sigma * eps, sigma


def make_dataset(
    w_val: float,
    cfg: ModelParams,
    noise: NoiseParams | None = None,
    seed: int = 0,
    with_noise: bool = True,
    flatten: bool = False,
    with_sigma: bool = False,
):
    """Generates a single (x,y) pair: synaptic weight and fluorescence trace."""
    C_clean, F_clean = simulate_pair_jit(w_val, cfg)

    if with_noise and noise is not None:
        key = random.PRNGKey(seed)
        F_obs, sigma = noisy_F(F_clean, C_clean, cfg, noise, key)
    else:
        F_obs, sigma = F_clean, None

    out_y = F_obs.reshape(-1) if flatten else F_obs
    result = {"x": jnp.asarray(w_val, dtype=jnp.float64), "y": out_y}
    if with_sigma and (sigma is not None):
        result["sigma"] = sigma.reshape(-1) if flatten else sigma
    return result
