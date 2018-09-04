## Materials Cloud AiiDA Lab Tools for Scanning Probe Microscopy simulations

This repository contains AiiDA workflows and Jupyter (Materials Cloud) GUI for running scanning probe microscopy simulations.
Most of the simulations are run by external tools, which must be set up on the remote computer and in AiiDA by the `setup_codes` app.
The AiiDA plugins for these external codes are also provided and are automatically set up also in the `setup_codes` app.
All calculations are performed on top of a CP2K wave function optimization (ran with diagonalization) and the tools extract the wavefunctions from the `.wfn` file.
