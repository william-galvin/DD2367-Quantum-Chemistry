from jax import numpy as jnp

from typing import Dict
from typing import Tuple
from typing import List


def givens(electrons: int, qubits: int, thetas: List[float]) -> jnp.ndarray:
    """
    Finds the full transition matrix for a given system and given thetas

    example:
    ```python

    thetas = [0, 0, .2]
    electrons, qubits = 2, 4

    G = givens(electrons, qubits, thetas)
    psi_valid = hf_ground(electrons, qubits) @ G

    H = hamiltonian.hamiltonian(...)
    psi = full_wavefunction(psi_valid, electrons, qubits)

    E = hamiltonian.expectation(H, psi)

    ```
    """
    states = get_states(electrons, qubits)
    singles, doubles = excitations(electrons, qubits)

    M = jnp.eye(len(states))

    for i, (src, dst) in enumerate(singles):
        M = M @ givens_single(src, dst, thetas[i], states)

    for i, idxs in enumerate(doubles):
        src = idxs[: len(idxs) // 2]
        dst = idxs[len(idxs) // 2 :]
        M = M @ givens_double(src, dst, thetas[i + len(singles)], states)

    return M


def hf_ground(electrons: int, qubits: int) -> jnp.ndarray:
    """ 
    Returns the hf-ground state of the wave function. The wave function vector
    contains only states thar are chemically valid (correct number of electrons)
    """
    states = get_states(electrons, qubits)

    hf_ground = jnp.zeros(len(states))
    hf_ground_str = "1" * electrons + "0" * (qubits - electrons)
    hf_ground = hf_ground.at[states[hf_ground_str][0]].set(1)

    return hf_ground


def full_wavefunction(valid_wavefunction: jnp.array, electrons: int, qubits: int) -> jnp.ndarray:
    """
    Converts a wave function containing only the valid states to the full wave function
    """
    states = get_states(electrons, qubits)
    full_psi = jnp.zeros(2 ** qubits)
    for state, (valid_idx, full_idx) in states.items():
        full_psi = full_psi.at[full_idx].set(valid_wavefunction[valid_idx])
    return full_psi


def excitations(electrons: int, orbitals: int) -> Tuple[List[int], List[int]]:
    """
    Enumerate the valid excitations.
    Exact copy of PennyLane implementation.
    https://docs.pennylane.ai/en/stable/_modules/pennylane/qchem/structure.html#excitations
    """
    delta_sz = 0
    sz = jnp.array([0.5 if (i % 2 == 0) else -0.5 for i in range(orbitals)])

    singles = [
        [r, p]
        for r in range(electrons)
        for p in range(electrons, orbitals)
        if sz[p] - sz[r] == delta_sz
    ]

    doubles = [
        [s, r, q, p]
        for s in range(electrons - 1)
        for r in range(s + 1, electrons)
        for q in range(electrons, orbitals - 1)
        for p in range(q + 1, orbitals)
        if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz
    ]

    return singles, doubles


def get_states(electrons: int, qubits: int) -> Dict[str, Tuple[int, int]]:
    """
    Finds the valid hf states for the given
    qubits (spin orbitals) and electrons.

    This is much less than the full 2 ** qubits states, so
    return is dictionary of binary strings of length q to 
    (valid_idx, i) where valid_idx is n when the state is the nth valid
    state seen, and i is the index into the full 2 ** qubits states
    """
    states = {}
    valid_idx = 0
    for i in range(2 ** qubits):
        if bin(i).count("1") == electrons:
            state_str = format(i, f"#0{qubits + 2}b")[2:]
            states[state_str] = (valid_idx, i)
            valid_idx += 1

    return states


def get_flipped_state(state: str, src: int, dst: int) -> str:
    """
    Given a binary string, flips the bits at src and dst
    """
    res = ""
    for i, c in enumerate(state):
        if i == src or i == dst:
            res += str(1 - int(c))
        else:
            res += c
    return res


def givens_single(src: int, dst: int, theta: float, states: Dict[str, Tuple[int, int]]) -> jnp.ndarray:
    """
    Returns the state transition matrix for a single excitation
    parameterized by theta.

    Src, dst should be ints
    theta should be a scalar.

    Returned matrix will not be  2 ** q x 2 ** q, but rather 
    len(states) x len(states)
    """
    M = jnp.eye(len(states))
    for state, (i, _) in states.items():
        if state[src] == "1" and state[dst] == "0":
            flipped = get_flipped_state(state, src, dst)
            j, _ = states[flipped]

            M = M.at[i, i].set( jnp.cos(theta / 2))  
            M = M.at[i, j].set(-jnp.sin(theta / 2))
            M = M.at[j, i].set( jnp.sin(theta / 2))
            M = M.at[j, j].set( jnp.cos(theta / 2))
    return M


def givens_double(src: List[int], dst: List[int], theta: float, states: Dict[str, Tuple[int, int]]) -> jnp.ndarray:
    """
    Returns the state transition matrix for a double excitation
    parameterized by theta.

    Src, dst should be lists of ints of size 2
    theta should be a scalar.

    Returned matrix will not be  2 ** q x 2 ** q, but rather 
    len(states) x len(states)
    """
    M = jnp.eye(len(states))
    for state, (i, _) in states.items():
        if state[src[0]] == "1" and state[dst[0]] == "0" and state[src[1]] == "1" and state[dst[1]] == "0":
            flipped = get_flipped_state(state, src[0], dst[0])
            flipped = get_flipped_state(flipped, src[1], dst[1])
            j, _ = states[flipped]

            M = M.at[i, i].set( jnp.cos(theta / 2))  
            M = M.at[i, j].set(-jnp.sin(theta / 2))
            M = M.at[j, i].set( jnp.sin(theta / 2))
            M = M.at[j, j].set( jnp.cos(theta / 2))
    return M
