# Bayesian Subspace Inference for Predicting Adamantinomatous Craniopharyngioma (ACP) from Preoperative MRI

## Motivation

This repository provides an end-to-end python with TensorFlow implementation of the Bayesian Subspace Inference methodology proposed by Izmailov et al. This method is applied to a deep learning image classifier for identifying ACP from other suprasellar tumors. The input of this system is 2D representative images of preoperative MRI of patients diagnosed with suprasellar tumor and the output of this system is the predicted diagnosis with calibrated predictive uncertainty.

This implementataion was used in the manuscript

> Prince E, Ghosh D, Görg C, Hankinson TC. "*Uncertainty-Aware Deep Learning Classification of Adamantinomatous Craniopharyngioma from Preoperative MRI*"<br>
(under review at [Diagnostics](https://www.mdpi.com/journal/diagnostics/special_issues/3USI9QQ3UD) as of Mar 3, 2023)

## Installation

First, clone the repository:

```
cd <location/to/store/package>
git clone lericnet/bsi-acp-mri
```

There are two components to this repository: the scripts to reproduce the results within the paper and the python library (Bayes). The Quick Start Section covers reproducing resuts, while the Advanced Application section details the use of the python library.

## Usage - Quick Start


## Usage - Advanced Application

**NOTE:** we do not officially support advanced application at this time.

## Preloaded Data

We provide:
- the data used in the mansucript. This can be accessed via the python library.
- the starting points/trained SWA models used to build the curve which we sampled from.

## Troubleshooting
Please report any issues, comments, or quesitons to Eric Prince vis email at Eric.Prince@CUAnschutz.edu, or [file an issue](https://github.com/lericnet/issues).
