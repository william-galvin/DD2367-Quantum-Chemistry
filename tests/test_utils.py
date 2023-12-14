import tempfile

from src import utils

def test_random_xyz():
    atoms = {"H": 12, "O": 89, "Cl": 4}
    with tempfile.NamedTemporaryFile() as f:
        utils.random_xyz(atoms, f.name)
        xyz = f.read().decode("utf-8")
        for sym, count in atoms.items():
            assert xyz.count(sym) == count
