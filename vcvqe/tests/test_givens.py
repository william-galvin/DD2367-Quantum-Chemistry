import numpy as np
import math

from vcvqe import givens

def test_givens_shape():
    for i in range(1, 5):
        electrons, qubits = i * 2, (i + 1) * 2

        states = givens.get_states(electrons, qubits)
        single, double = givens.excitations(electrons, qubits)
        thetas = np.random.random(len(single) + len(double))

        G = givens.givens(electrons, qubits, thetas)
        assert G.shape == (len(states), len(states))


def test_givens_unitary():
    for i in range(1, 5):
        electrons, qubits = i * 2, (i + 1) * 2

        states = givens.get_states(electrons, qubits)
        single, double = givens.excitations(electrons, qubits)
        thetas = np.random.random(len(single) + len(double))

        G = givens.givens(electrons, qubits, thetas)

        assert np.allclose(np.eye(len(states)), G @ G.conj().T, atol=1e-6)


def test_hf_ground_state():
    for i in range(1, 5):
        electrons, qubits = i * 2, (i + 1) * 2

        state = givens.hf_ground(electrons, qubits)
        assert sum(state) == 1

    
def test_full_wavefunction():
    for i in range(1, 5):
        electrons, qubits = i * 2, (i + 1) * 2

        state = givens.hf_ground(electrons, qubits)
        state = givens.full_wavefunction(state, electrons, qubits)
        
        idx = 0
        for e in range(electrons):
            idx |= 1 << (qubits - e - 1) 

        assert state[idx] == 1


def test_get_states():
    for i in range(1, 5):
        electrons, qubits = i * 2, (i + 1) * 2

        states = givens.get_states(electrons, qubits)

        assert len(states) == math.comb(qubits, electrons)

        for state in states:
            assert state.count("1") == electrons