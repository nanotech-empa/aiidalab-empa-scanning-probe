[![DOI](https://zenodo.org/badge/136625855.svg)](https://zenodo.org/badge/latestdoi/136625855)

## AiiDAlab app for Scanning Probe Microscopy simulations

This repository contains AiiDA workflows and Jupyter GUI for running scanning probe microscopy simulations.
The calculations are performed on top of a CP2K wave function optimization (ran with diagonalization) with the [cp2k-spm-tools](https://github.com/nanotech-empa/cp2k-spm-tools).
The external tools need to be set up on the remote computer and can be conveniently done with `setup_codes` interface, which also installs the corresponding AiiDA plugins.

Features include:

* Scanning tunnelling microscopy/spectroscopy (STM/STS) simulations.
* Orbital analysis - calculates the STM/STS signatures of orbitals of isolated molecules.
* Projected Density of States (PDOS) analysis - calculates the DOS projection on the adsorbed molecule together with gas-phase orbital content.
* Atomic force microscopy (AFM) simulations - calls the [ProbeParticle](https://github.com/ProkopHapala/ProbeParticleModel) code.
* High-resolution STM - simulates the STM/STS signatures performed with the CO tip.

## Software attribution and scientific acknowledgements

Please credit [aiidalab-empa-scanning-probe](https://github.com/nanotech-empa/aiidalab-empa-scanning-probe) and its
[contributors](https://github.com/nanotech-empa/aiidalab-empa-scanning-probe/graphs/contributors),
and identify the release or immutable commit used. Existing author,
contributor and funding acknowledgements remain authoritative.

Preserve the copyright and permission notices in [LICENSE.txt](LICENSE.txt)
when copying or distributing copies or substantial portions of MIT-licensed
code. Separately licensed components retain their own notices. A name or
repository link does not replace the applicable licence notices.

For research using this app, acknowledge the [AiiDA team](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md#aiida)
and [AiiDAlab team](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md#aiidalab), and cite the three AiiDA papers
and both AiiDAlab platform papers listed in that guide.

Retain this scanning-probe app's own DOI and contributor credits, separately from the Surfaces app. Also credit cp2k-spm-tools, CP2K, ProbeParticle and other methods or tools actually used.

The [shared credit guide](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md) keeps software attribution, licence
compliance and scientific citation requests separate. Scientific citation
requests are not additional MIT licence conditions.
