# Well-Free Acoustic Impedance Inversion with Bayesian Physics-Informed Neural Networks (B-PINN)

> **Status: work in progress** (Master's research, Institut Teknologi Sepuluh Nopember). Results below are preliminary and not peer-reviewed.

## Problem

Acoustic impedance (Zp) is normally estimated from post-stack seismic data using well logs as constraints. This project asks whether Zp can be estimated **without well control** on post-stack 2D seismic data, and whether the **uncertainty** of that estimate can be quantified.

## Approach

- Forward model: <<ISI: convolutional model? wavelet used? how the physics term enters the loss?>>
- Network: <<ISI: architecture, inputs (trace / window), outputs>>
- Uncertainty quantification (compared):
  - Variational inference (main method)
  - Hamiltonian Monte Carlo (1D reference only)
  - Deep ensembles (baseline)
- Planned extensions: lithology and total porosity via rock-physics relations (not implemented yet).

## Experiments in this repository

| Notebook | What it does | Data |
|---|---|---|
| `E1_BPINN_1D_Colab.ipynb` | <<ISI: 1D synthetic test, single run>> | <<ISI>> |
| `E1_BPINN_1D_5seeds_Colab.ipynb` | Same as E1 repeated over 5 random seeds to check stability | <<ISI>> |
| `Marmousi2_4windows_Colab.ipynb` | <<ISI: 2D test on Marmousi2, 4 windows>> | Marmousi2 (public) |
| `Window_B_E2_E3_E4_Colab.ipynb` | <<ISI: what E2, E3, E4 each vary>> | <<ISI>> |
| `Field_line_final_4windows_Colab.ipynb` | <<ISI: field line application>> | <<ISI: source and permission to publish>> |

## Results (preliminary)

<<ISI: insert 1-2 figures: predicted vs true impedance, and uncertainty map. Add one metric table (e.g. correlation, RMSE) for VI vs HMC vs ensemble. Report only numbers you can reproduce from the notebooks.>>

## Reproducing

1. Open a notebook in Google Colab (GPU runtime recommended).
2. <<ISI: where to get the data / how to download it>>
3. Run all cells. <<ISI: approximate runtime>>

Dependencies: <<ISI: PyTorch version, other libraries>>. Random seeds are set in each notebook.

## Limitations

- Validated primarily on synthetic data; field-data validation is ongoing.
- Without well control the result is non-unique; the uncertainty estimates reflect the model and prior, not necessarily the true error.
- <<ISI: any other limitation you have observed>>

## Data note

<<ISI: state the data source and license. Do not commit proprietary or third-party seismic data without written permission.>>

## Author

Mirza Amanda Azzahra, M.Sc. student, Geophysical Engineering, ITS. Contact: mirzaamanda1102@gmail.com
