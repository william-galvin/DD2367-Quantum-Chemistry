import numpy as np

from typing import Dict


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




    



