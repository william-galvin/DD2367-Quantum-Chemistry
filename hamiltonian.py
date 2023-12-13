import numpy as np
import pyscf
import itertools
import openfermion
import openfermionpyscf


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

    # mol = pyscf_to_openfermion(mol_pyscf)
    # openfermionpyscf.run_pyscf(mol)
    # one_e, two_e = mol.get_integrals()

    hf = mol_pyscf.RHF()
    hf.kernel()
    for j in range(10):
        mo = hf.stability()[0]
        if np.allclose(mo, hf.mo_coeff):
            break
        dm = hf.make_rdm1(mo, hf.mo_occ)
        hf = hf.run(dm)

    h1e = mol_pyscf.intor("int1e_kin") + mol_pyscf.intor("int1e_nuc")
    h2e = mol_pyscf.intor("int2e")

    scf_c = hf.mo_coeff
    nuclear_repulsion = mol_pyscf.energy_nuc()
    constant = nuclear_repulsion

    one_e = scf_c.T @ h1e @ scf_c
    for i in range(4):
        h2e = np.tensordot(h2e, scf_c, axes=1).transpose(3, 0, 1, 2)
    two_e = h2e.transpose(0, 2, 3, 1)

    N = len(mol_pyscf.spinor_labels())

    H = np.zeros((2 ** N, 2 ** N), dtype=complex)

    for p, q in itertools.product(range(N), range(N)):
        if p % 2 == q % 2:
            H += one_e[p // 2, q // 2] * creation(p, N) @ annihilation(q, N)

    for p, q, r, s in itertools.product(range(N), range(N), range(N), range(N)):
        if p % 2 == r % 2 and q % 2 == s % 2:
            H += 0.5 * two_e[p // 2, q // 2, r // 2, s // 2] * creation(p, N) @ creation(q, N) @ annihilation(r, N) @ annihilation(s, N)

    return H  + constant


def pyscf_to_openfermion(mol: pyscf.gto.mole.Mole) -> openfermion.chem.MolecularData:
    geometry = []
    for line in mol.tostring().split("\n"):
        sym, x, y, z = line.split()
        x, y, z = float(x), float(y), float(z)
        geometry.append((sym, (x, y, z)))

    return openfermion.chem.MolecularData(
        geometry=geometry,
        basis=mol.basis,
        multiplicity=mol.multiplicity
    )


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
    A =np.array([[1]])
    for n in range(N):
        if n < i:
            t = Z
        elif n == i:
            t = (X + 1j * Y) / 2
        else:
            t = I
        A = np.kron(A, t)
    return A


def expectation(H: np.ndarray, psi: np.array) -> float:
    """
    Calculates <psi*|H|psi>/<psi*|psi> = the expectation
    of the Hamiltonian with respect to the wave function
    """
    return (psi.conj() @ H @ psi) / (psi.conj() @ psi)


if __name__ == "__main__":
    H = hamiltonian("data/h2.xyz")
    print("H shape:", H.shape)
    np.save("h2_H.npy", H)

