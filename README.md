# Sediment Corrections for Distributed Acoustis Sensing

[![DOI](https://zenodo.org/badge/748286787.svg)](https://zenodo.org/doi/10.5281/zenodo.10678784)

This repository contains the codes necessary to reproduce the figures from Trabattoni et al. (2024).

First choose where to store the travel-time lookup table in `config.py`.

Then run `run.sh`.

Warning: the TTLUT size is about 80GB. Your hardware should be enough to store it but
also to load it on RAM. Otherwise reduce the TTLUT resolution in `0_ttlut.py`.