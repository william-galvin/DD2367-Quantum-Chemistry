import numpy as np

from src import hamiltonian

def test_creation_annihilation_shape():
    for n in range(8):
        for i in range(n):
            c = hamiltonian.creation(i, n)
            assert c.shape == (2 ** n, 2 ** n)

            a = hamiltonian.creation(i, n)
            assert a.shape == (2 ** n, 2 ** n)


def test_creation_annihilation_anti_symmetry():
    for n in range(8):
        for i in range(n):
            c = hamiltonian.creation(i, n)
            a = hamiltonian.creation(i, n)

            assert (c @ a == - (a @ c)).all()


def test_hamiltonian_hermitian():
    file = "data/h2.xyz"
    H = hamiltonian.hamiltonian(file)
    assert np.allclose(H, H.conj().T)
