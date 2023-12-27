import tempfile
import sys
import argparse
import io

import jax
jax.config.update("jax_platform_name", "cpu")

import numpy as np

from jax import numpy as jnp
from jax import value_and_grad

from src.givens import givens
from src.givens import hf_ground
from src.givens import full_wavefunction
from src.givens import excitations
from src.hamiltonian import expectation
from src.hamiltonian import hamiltonian
from src.utils import get_electrons_qubits
from src.utils import random_xyz


def energy(H: np.ndarray, electrons: int, qubits: int, thetas: np.ndarray) -> float:
    """
    Returns the expected value of H for the system specified by (electrons, qubits, thetas)

    Params: 
    - H: hamiltonian
    - electrons: n_electrons
    - qubits: n_qubits
    - thetas: vector of angles
    """
    G = givens(electrons, qubits, thetas)

    psi_valid = hf_ground(electrons, qubits) @ G
    psi = full_wavefunction(psi_valid, electrons, qubits)

    return expectation(H, psi)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="This script implements a very classical VQE to find the lowest enegery state of a hydrogen molecule",
        epilog="For more information or context, see the paper accompanying this repo, or any VQE for quantum chemistry tutorial (PennyLane, Azure, etc)."
    )

    parser.add_argument(
        "--random",
        action="store_true",
        default=False,
        help="Specify whether to use a random molecule or supply your own. If --random is set, --num-H must be too. If not, --file must be set"
    )
    parser.add_argument(
        "--no-random",
        dest="random",
        action="store_false"
    )

    parser.add_argument(
        "--num-H",
        type=int,
        default=4,
        help="The number of hydrogen atoms to randomly place. This argument is only used if --random is used. If PySCF doesn't support this molecule, program crashes."
    )

    parser.add_argument(
        "--file",
        help="Path to .xyz file of input molecule if --random is not set"
    )

    parser.add_argument(
        "--max-iter",
        type=int, 
        default=25,
        help="Maximum number of optimization cycles"
    )

    parser.add_argument(
        "--min-iter",
        type=int, 
        default=5,
        help="Minimum number of optimization cycles"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Convergence threshdold"
    )

    parser.add_argument(
        "--step-size",
        type=float,
        default=1,
        help="Optimization step size aka learning rate"
    )

    return parser.parse_args(), parser


def main():
    linebreak = "\n" + 100 * "=" + "\n"

    args, parser = get_args()
    if not args.random and args.file is None:
        parser.print_help()
        exit(-1)

    with tempfile.NamedTemporaryFile() as f:
        if args.random:
            if args.num_H is None:
                raise ValueError("Must specify --num-H if --random is used")
            
            random_xyz({"H": args.num_H}, f.name)
            print(f"Random H{args.num_H} molecule generated:\n")
            print(open(f.name).read())
            print(linebreak)

            file = f.name 
        else:
            file = args.file

        try:
            electrons, qubits = get_electrons_qubits(file)
        except RuntimeError as e:
            raise RuntimeError(f"Molecule not valid for PySCF:\n {str(e)}")
        
        print(f"\nCreating Hamiltonian for {electrons} electrons and {qubits} spin orbitals")
        H = hamiltonian(file)
        print(linebreak)

    singles, doubles = excitations(electrons, qubits)
    print(linebreak)
    print(f"{len(singles)} single excitations and {len(doubles)} double excitations")
    print(linebreak)

    theta = np.random.random(len(singles) + len(doubles)) * jnp.pi

    dE = value_and_grad(energy, argnums=3)

    print("Beginning optimization:")

    sys.stderr = io.StringIO()
    prev = float("inf")
    for i in range(args.max_iter):
        E, theta_prime = dE(H, electrons, qubits, theta)
        theta -= theta_prime * args.step_size

        print(f"\tstep: {i: 2}\t\tE: {E: .3f}\t\tDifference: {E - prev: .3f}")
        
        if abs(E - prev) < args.threshold and i >= args.min_iter - 1:
            break

        prev = E

    print(linebreak)
    print(f"Final E converged to {E: 5f} in {i + 1} steps\n")


if __name__ == "__main__":
    main()