import numpy as np
import pyscf
import itertools


X = np.array([[0,  1 ], [1,  0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1,  0 ], [0, -1]], dtype=complex)
I = np.array([[1,  0 ], [0,  1]], dtype=complex)


def hamiltonian(xyz_path: str, basis: str = "sto-3g") -> np.ndarray:
    """
    Constructs the electronic hamiltonian for a given molecule

    Params:
    - xyz_path: path to .xyz file containing coordinates of atoms. These
      can be found here: https://cccbdb.nist.gov/expgeom1x.asp among other places.
      (Enter your formula in the text box, scroll down until you see each atom with a 
      list of x,y,z coordinates).

    - basis: molecurlar orbital basis to use. Defaults to 'sto-3g'

    Returns array/gate representation of the electronic hamiltonian
    """
    mol_pyscf = pyscf.gto.M(atom=xyz_path, basis=basis)

    h1 = mol_pyscf.intor("int1e_kin_spinor") + mol_pyscf.intor("int1e_nuc_spinor")
    h2 = mol_pyscf.intor("int2e_spinor")

    N = len(mol_pyscf.spinor_labels())

    H = np.zeros((2 ** N, 2 ** N), dtype=complex)

    for p, q in itertools.product(range(N), range(N)):
        H += h1[p, q] * creation(p, N) @ annihilation(q, N)

    for p, q, r, s in itertools.product(range(N), range(N), range(N), range(N)):
        H -= 0.5 * h2[p, q, r, s] * creation(p, N) @ creation(q, N) @ annihilation(r, N) @ annihilation(s, N)

    return H


def creation(i: int, N: int) -> np.ndarray:
    """
    Returns a 2^N x 2^N matrix representing the creation_i operator

    Params:
    - i: (0-indexed) spin orbital to create 
    - N: number of spin orbitals

    Example:
    ```python
    (creation(1, 2)
    >>>
    [[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
    [ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
    [ 0.+0.j  0.+0.j -0.+0.j -0.+0.j]
    [ 0.+0.j  0.+0.j -1.+0.j -0.+0.j]]
    ```
    """
    A = np.array([[1]])
    for n in range(N):
        if n < i:
            t = Z
        elif n == i:
            t = (X - 1j * Y) / 2
        else:
            t = I
        A = np.kron(A, t)
    return A


def annihilation(i: int, N: int) -> np.ndarray:
    """
    Returns a 2^N x 2^N matrix representing the annihilation_i operator

    Params:
    - i: (0-indexed) spin orbital to create
    - N: number of spin orbitals

    Example:
    ```python
    annihilation(1, 2)
    >>>
    [[ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
    [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
    [ 0.+0.j  0.+0.j -0.+0.j -1.+0.j]
    [ 0.+0.j  0.+0.j -0.+0.j -0.+0.j]]
    ```
    """
    A = np.array([[1]])
    for n in range(N):
        if n < i:
            t = Z
        elif n == i:
            t = (X + 1j * Y) / 2
        else:
            t = I
        A = np.kron(A, t)
    return A


def target_energy(path: str, basis: str = "sto-3g") -> float:
    """
    Uses pyscf to calculate the target electronic (e1) energy.
    When we minimize theta, the expectation of H w.r.t. psi
    should be close to this
    """
    mol_pyscf = pyscf.gto.M(atom=path, basis=basis)
    hartree, coulomb = mol_pyscf.energy_elec()
    return (hartree - coulomb).real


def expectation(H: np.ndarray, psi: np.ndarray) -> float:
    """
    Calculates <psi*|H|psi>/<psi*|psi> = the expectation
    of the Hamiltonian with respect to the wave function
    """
    return ((psi.conj() @ H @ psi) / (psi.conj() @ psi)).real


if __name__ == "__main__":
    file = "data/h4.xyz"
    H = hamiltonian(file)
    print("H shape:", H.shape)
    print("target energy:", target_energy(file))
    np.save("h4_H.npy", H)

