# Inferring Neural Connectivity with Consistency Models and MCMC (JAX)

This repository contains the core code and outputs developed as part of the Master's Thesis project titled  
**“Inferring Neural Connectivity with the Consistency Model in Simulated Circuits.”**

The project focuses on simulating calcium imaging data from a minimal two-neuron network and performing inference of synaptic connectivity using two complementary approaches:
1. **Consistency Models (CM)** – an amortized inference framework capable of mapping noisy fluorescence signals to underlying physiological variables with high computational efficiency.
2. **Markov Chain Monte Carlo (MCMC)** – a Bayesian sampling method used as a benchmark to validate and compare the accuracy of CM-based inference.

All experiments were implemented using **JAX**, leveraging its automatic differentiation and vectorized computation capabilities for simulation, training, and inference.

---

## Repository Structure

| File | Description |
|------|--------------|
| `JAX_Simulation.ipynb` | Generates two-neuron calcium imaging data using a simplified LIF-based model. |
| `TwoNeuron_JAX.py` | Core simulation script defining neuron and synapse dynamics. |
| `model.py` | Implementation of the 1D U-Net Consistency Model architecture (*adopted and modified from Zhao et al., 2025*). |
| `mcmc_cm_inference_comparison.ipynb` | Compares inference accuracy between MCMC and Consistency Model approaches. |
| `README.md` | Overview and documentation of the repository. |

---

## Methodology Summary

1. **Simulation:**  
   Synthetic calcium traces were generated from a two-neuron system using a leaky-integrate-and-fire (LIF) model implemented in JAX. The simulated fluorescence signals include calcium decay and Gaussian noise to replicate realistic imaging conditions.

2. **Inference Approaches:**  
   - **MCMC Sampling:** Used as the ground-truth Bayesian inference method to estimate synaptic weights and uncertainty.  
   - **Consistency Model:** A trained neural network that performs direct (amortized) inference of synaptic weights from calcium traces, achieving faster evaluation after training.

3. **Evaluation:**  
   Inference results from both methods were compared based on RMSE, posterior alignment, and qualitative trace reconstruction performance.

---
