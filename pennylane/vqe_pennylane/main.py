#!/usr/bin/env python3

import argparse
import os

from pennylane import numpy as np
import numpy
import pennylane as qml


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        help="Path to .xyz file for input molecule. Use this xor --mol"
    )

    parser.add_argument(
        "--mol",
        help="Name of molecule in pennylane dataset. Use this xor --file"
    )

    parser.add_argument(
        "--max-iter",
        type=int, 
        default=50,
        help="Maximum number of optimization cycles"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Convergence threshdold"
    )

    parser.add_argument(
        "--output-dir",
        default="."
    )
    
    args = parser.parse_args()

    if args.file is not None:
        symbols, geometry = qml.qchem.read_structure(args.file)

        mol = qml.qchem.Molecule(symbols, geometry)
        electrons = mol.n_electrons
        H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)
        hf = qml.qchem.hf_state(electrons, qubits)
    else:
        dataset = qml.data.load("qchem", molname=args.mol)[0]
        H = dataset.hamiltonian 
        qubits = len(H.wires)
        hf = dataset.hf_state
        electrons = str(hf).count("1")
    
    if qubits > 8:
        print(f"{qubits} qubits: This may take awhile...")
    else:
        print("Number of qubits = ", qubits)


    dev = qml.device("lightning.qubit", wires=qubits)

    
    print("Hartree-fock basis state:", hf)
    

    def circuit(param, wires):
        qml.BasisState(hf, wires=wires)
        singles, doubles = qml.qchem.excitations(electrons, qubits)

        for i, s in enumerate(singles):
            qml.SingleExcitation(param[i], wires=s)
        for i, s in enumerate(doubles):
            qml.DoubleExcitation(param[i + len(singles)], wires=s)


    @qml.qnode(dev, interface="autograd")
    def cost_fn(param):
        circuit(param, wires=range(qubits))
        return qml.expval(H)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    theta = np.random.random((len(singles) + len(doubles)), requires_grad=True)


    energy = [cost_fn(theta)]
    angle = numpy.zeros((args.max_iter, theta.shape[0]))

    for n in range(args.max_iter):
        angle[n] = theta
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        energy.append(cost_fn(theta))

        conv = np.abs(energy[-1] - prev_energy)

        if n % 2 == 0:
            print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

        if conv <= args.threshold:
            break

    print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
    print("\n" f"Optimal value of the circuit parameter:")
    for i in theta:
        print(f"{i: 5f}")


    import matplotlib.pyplot as plt

    plt.plot(range(n + 2), energy, "tab:red", ls="-.")
    plt.xlabel("Optimization step", fontsize=13)
    plt.ylabel("Energy (Hartree)", fontsize=13)
    plt.ylabel(r"$E_\mathrm{HF}$", fontsize=15)
    plt.title("Calculated Energy vs Optimization step")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    plt.savefig(f"{args.output_dir}/plot.png")
    print(f"\nPlot saved at {args.output_dir}/plot.png")

    numpy.save(f"{args.output_dir}/thetas.npy", angle)
    print(f"Thetas saved at {args.output_dir}/thetas.npy")


if __name__ == "__main__":
    main()