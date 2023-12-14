import numpy as np
import pyscf

from typing import Dict
from typing import Tuple


def random_xyz(mol: Dict[str, int], path: str):
    """
    Creates a random xyz file with a given set of molecules.
    Does not respect the laws of the physical universe.

    Params:
    - mol: mapping of atom names to counts
    - path: place to write file
    """
    with open(path, "w+") as f:
        f.write(str(sum(mol.values())))
        f.write("\n\n")

        for sym, count in mol.items():
            for _ in range(count):
                f.write(sym)
                for _ in range(3):
                    f.write(f"\t\t{(np.random.random() - 0.5) * 5:.4f}")
                f.write("\n")


def get_electrons_qubits(path: str) -> Tuple[int, int]:
    """
    Returns the number of electrons and qubits (spin orbitals)
    from an atom specified in a .xyz file
    """
    mol = pyscf.gto.M(atom=path)
    return sum(mol.nelec), len(mol.spinor_labels())

