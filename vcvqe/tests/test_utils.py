import tempfile
import os

from vcvqe import utils

root = "." if os.path.isdir("data") else ".."

def test_random_xyz():
    atoms = {"H": 12, "O": 89, "Cl": 4}
    with tempfile.NamedTemporaryFile() as f:
        utils.random_xyz(atoms, f.name)
        xyz = f.read().decode("utf-8")
        for sym, count in atoms.items():
            assert xyz.count(sym) == count


def test_get_electrons_qubits():
    file = f"{root}/data/h2.xyz"
    assert utils.get_electrons_qubits(file) == (2, 4)

    file = f"{root}/data/h4.xyz"
    assert utils.get_electrons_qubits(file) == (4, 8)

    file = f"{root}/data/h2o.xyz"
    assert utils.get_electrons_qubits(file) == (10, 14)