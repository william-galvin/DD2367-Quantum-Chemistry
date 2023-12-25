# Very Classical Variational Quantum Eigensolver (VCVQE)

## Background

This is a classical implementation of the Variational Quantum Eigensolver for quantum chemistry described in [this PennyLane tutorial](https://pennylane.ai/qml/demos/tutorial_vqe/). See the accompanying write-up in `../paper` for more details. 

This is considered a "classical" implementation because it takes the ideas from the VQE and implements them in classical matrix algebra, an approach that quickly becomes unmanageable as molecules grow large. For example, with a H2O molecule, this requires 14 qubits and therefor a Hamiltonian $H \in \mathbb{C}^{2^{14} \times 2^{14}}$, which is too large to even write down on a laptop. Nonetheless, it provides an instructive window into the inner workings of the VQE

The energies calculated by this program account for only electron-electron and electron-nucleus interactions and electron kinetic energy, under the Born-Oppenheimer and Hartree-Fock approoximations. It does not incorporate nucleus-nucleus interactions, nuclear kinetic energies, or pairwise electron-electron energies. As such, **the energies calculated here do not match the energies calculatd by PySCF and PennyLane**.

This program takes as input a molecular geometry (the 3D coordinates of the nuclei) and attempts to solve the electron structure problem. It prints a brief report of its findings.

## Installation
1. Clone this repo and `cd` to the root directory `vcvqe`
2. Install with `pip install -r requirements.txt` and `pip install -e .`

## Usage
Run with 
```
$ vcvqe [-h] [--random] [--no-random] [--num-H NUM_H] [--file FILE] [--max-iter MAX_ITER] [--min-iter MIN_ITER] [--threshold THRESHOLD] [--step-size STEP_SIZE]
```
For example
```
vcvqe --file data/h4.xyz
```
or
```
vcvqe --random --num-H 2 --max-iter 25 --min-iter 5 --threshold 0.001 --step-size 0.25
```

This program can be used without `pip install -e .` by running `python src/main.py [args]`.

### args
- `random`: Use a hydrogen molecules with atoms randomly placed in 3D space. Almost certainly will not be a physically observable molecule. If `random` is used, `file` should not be used and `num-H` should be used.
- `num-H`: Number of hydrogen atoms to use in random hydrogen molecule. H2 and H4 are supported, but note that H3 is not (PySCF doesn't support it). Defaults to 4
- `file`: path to file with a molecule's `.xyz` coords. Theoretically may be any molecule, works best if it's a very small molecule (H2 or H4).
- `[max|min]-iter` max/min iterations of optimization loop to perform. Defaults to 25 and 5
- `threshold` If two consecutuve energies are calculated with a difference not greater than threshold, break optimization loop. Defaults to 0.001.
- `step-size` Learning rate in optimization step. Defaults to 1
