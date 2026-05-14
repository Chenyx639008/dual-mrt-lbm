"""Validation suite for Huang & Wu (2016) MRT-LBM SCMP solver.

Modules:
- cs_eos: Carnahan-Starling EOS host-side + Maxwell construction
- analytical: Shared post-processing (interface detection, sigma fitting)
- static_droplet: Droplet sweep driver
- laplace_law: Laplace law analyzer
- coexistence: Flat-interface coexistence curve
- spurious_currents: Spurious currents comparison
- decoupling_sweep: σ-decoupling headline test
"""
