# DeepSeek-Light_Matter_Interaction
RealQM Engine — A Computational Companion to the Z-Papers
# RealQM — A Computational Engine for the Realist Interpretation of Quantum Mechanics

This repository contains the simulation engine that accompanies the Z-2 and Z-3 papers on RealQM — a realist reinterpretation of quantum mechanics in which the photon is a localized electromagnetic wavepacket, the electron is a localized charge oscillator, and the glass is an amorphous ensemble of such oscillators.

The engine reproduces the optical phenomena of a real glass slab — refraction, Fresnel reflection, Rayleigh scattering, Beer-Lambert attenuation, Fabry-Perot fringes — from first principles, without probability amplitudes, path integrals, or collapse postulates. Every number in the accompanying papers traces back to a named constant or a tested module here.

## What the engine does

Given a photon energy and a set of material parameters (mean resonance frequency, disorder width, damping rate, number density, boundary width), the engine computes:

- The ensemble-averaged susceptibility `⟨χ(ω)⟩` of an amorphous dielectric.
- The complex refractive index `n(ω)`, including the Sellmeier low-frequency limit.
- The Fresnel reflection coefficient of a finite-width boundary, and its convergence to the standard result as the boundary width goes to zero.
- The Rayleigh `λ⁻⁴` extinction coefficient and the Beer-Lambert attenuation.
- The full energy ledger of a plane-parallel slab: reflected, transmitted, and side-scattered fractions, summing exactly to 1.
- The Fabry-Perot (Airy) transmission, including the multiple-reflection sum.

## What the engine does *not* do

- It does not use probability amplitudes, path integrals, wavefunction collapse, or virtual particles.
- It does not model thermal absorption. The photons are transmitted, reflected, or scattered; they are not absorbed as heat. A small residual absorption from the tail of the UV resonance is present, but it is negligible at visible wavelengths.
- It does not model multi-layer coatings. The high-reflectivity case in Figure 6.2b is illustrative of the Airy formula at `R = 0.9`, not of a slab the model can produce with a single boundary.
- It does not (yet) provide an interactive visualisation. An HTML widget was attempted and is not included; see the Z-3 paper for the honest account.

## Installation

The engine requires Python 3.10 or later, NumPy, SciPy, Matplotlib, and pytest.
