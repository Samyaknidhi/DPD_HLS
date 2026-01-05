# DPD_HLS
Digital Predistortion Implemented via High level Synthesis language (Vivado HLS 2017.4)

# Digital Pre-Distortion using Orthogonal Polynomial Model

## Overview

This project implements a Digital Pre-Distortion (DPD) system based on an orthogonal polynomial representation to linearize the non-linear behavior of Radio Frequency (RF) Power Amplifiers (PAs). The work focuses on a structured, model-based DPD formulation that is amenable to fixed-point arithmetic and hardware implementation.

The system simulates a complete digital communication transmit and receive chain, including constellation mapping, pulse shaping, digital up-conversion, non-linear PA modeling, digital down-conversion, and a feedback-based coefficient adaptation loop using an indirect learning architecture.

This implementation was developed in parallel with an evolutionary-optimization–based DPD approach, enabling a comparative study between model-structured and optimizer-driven DPD formulations.

---

## Motivation

Power Amplifiers (PAs) exhibit strong non-linear behavior when operated near saturation to maximize power efficiency. These non-linearities result in in-band distortion and spectral regrowth, degrading signal integrity and violating spectral emission masks.

Digital Pre-Distortion (DPD) mitigates these effects by applying an inverse non-linear transformation to the baseband signal prior to amplification. While global optimization techniques can achieve high modeling accuracy, practical hardware DPD implementations overwhelmingly rely on **structured polynomial models** due to their deterministic behavior, numerical stability, and lower computational complexity.

This project investigates an orthogonal polynomial–based DPD formulation to understand its effectiveness, convergence behavior, and suitability for hardware-oriented realization.

---

## System Architecture

The DPD system follows an **Indirect Learning Architecture (ILA)**, where the inverse PA characteristic is learned using a feedback loop from the PA output.

### Operating Modes

* **`BASELINE` Mode**  
  The signal passes through the PA without pre-distortion, capturing the intrinsic non-linear behavior of the amplifier.

* **`ADAPT` Mode**  
  The orthogonal polynomial coefficients are estimated using the feedback signal from the PA output by minimizing the error between the desired and observed signals.

* **`FINAL` Mode**  
  The learned coefficients are applied to the forward path, demonstrating the linearization effect of the DPD.

---

## Key Features

* **Orthogonal Polynomial DPD Model**  
  Implements a structured polynomial basis (e.g., orthogonalized memory polynomial) to represent the inverse PA characteristic.

* **Indirect Learning Architecture (ILA)**  
  Adaptation is performed in the feedback path, allowing stable coefficient estimation without direct inversion of the PA model.

* **Complete Baseband Signal Chain**  
  Includes:
  * Constellation Mapper (QPSK)
  * Pulse Shaping Filter
  * Digital Quadrature Modulator
  * Digital Up-Converter (DUC)
  * Saleh Power Amplifier Model
  * Digital Down-Converter (DDC)
  * Analog-to-Digital Converter (ADC)
  * Feedback Pulse Shaping Filter

* **Hardware-Oriented Design**  
  Emphasizes determinism, numerical stability, and modularity suitable for fixed-point and RTL/HLS translation.

* **Intermediate Signal Visibility**  
  Dumps I/Q samples at multiple stages of the signal chain for offline analysis and verification.

---

## Orthogonal Polynomial Model Details

The DPD block models the inverse PA characteristic using a weighted sum of orthogonal basis functions derived from the input signal magnitude and memory terms. Orthogonalization reduces coefficient correlation, improving numerical conditioning and convergence behavior compared to standard polynomial models.

Key design considerations include:
- Polynomial order vs. modeling accuracy
- Memory depth vs. computational complexity
- Sensitivity to coefficient quantization
- Suitability for fixed-point arithmetic

---

## Implementation Notes

* The implementation is written in C/C++ with a modular structure to allow gradual migration toward HLS or RTL-based realization.
* Floating-point arithmetic is used for algorithmic validation, with clear identification of blocks requiring fixed-point redesign for hardware deployment.
* The orthogonal polynomial formulation was evaluated alongside an evolutionary optimization–based DPD approach to understand practical trade-offs.

---

## Execution Flow

The program executes in three phases:

1. **Baseline Run**  
   Captures PA non-linearity without DPD.

2. **Adaptation Phase**  
   Estimates orthogonal polynomial coefficients using the feedback signal.

3. **Final Run**  
   Applies the learned coefficients to demonstrate PA linearization.

All intermediate and final outputs are written to text files for post-processing and visualization.

---

## Output Files

The simulation generates `.txt` files containing floating-point I/Q samples and metrics at various stages of the signal chain, including:

* Constellation mapper outputs
* Pulse-shaped baseband signals
* Pre-distorted DPD output
* PA output magnitude and gain
* Feedback path signals before and after DPD

These files can be analyzed using MATLAB, Python, or Octave to evaluate constellation integrity, AM/AM behavior, and spectral regrowth.

---

## Comparative Context

This project was developed in parallel with a DPD implementation based on Modified Differential Evolution (MDE). While the MDE-based approach offers global optimization capability, the orthogonal polynomial model demonstrated superior determinism and hardware feasibility, highlighting why structured models are preferred in practical real-time DPD systems.

---

## Acknowledgements

* **Digital Pre-Distortion (DPD)**: Based on established DPD principles and memory polynomial models.
* **Orthogonal Polynomial Modeling**: Inspired by practical DPD formulations used in RF transmitters.
* **Saleh Power Amplifier Model**: Widely used behavioral PA model for non-linear analysis.

