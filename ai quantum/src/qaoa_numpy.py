
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class QAOAResult:
    bitstring: np.ndarray
    probability: float
    expected_energy: float
    gamma: float
    beta: float
    top_states: list[dict]


def all_bitstrings(n: int) -> np.ndarray:
    states = np.arange(2**n, dtype=np.uint64)
    shifts = np.arange(n, dtype=np.uint64)
    return ((states[:, None] >> shifts) & 1).astype(float)


def qubo_energies(Q: np.ndarray, bits: np.ndarray) -> np.ndarray:
    return np.einsum("bi,ij,bj->b", bits, Q, bits, optimize=True)


def _apply_rx_mixer(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    out = state.copy()
    c = np.cos(beta)
    s = -1j * np.sin(beta)

    for q in range(n):
        stride = 1 << q
        block = stride << 1
        for start in range(0, len(out), block):
            a_idx = np.arange(start, start + stride)
            b_idx = a_idx + stride
            a = out[a_idx].copy()
            b = out[b_idx].copy()
            out[a_idx] = c * a + s * b
            out[b_idx] = s * a + c * b
    return out


def qaoa_state(Q: np.ndarray, gamma: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    n = Q.shape[0]
    bits = all_bitstrings(n)
    energies = qubo_energies(Q, bits)

    state = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    state *= np.exp(-1j * gamma * energies)
    state = _apply_rx_mixer(state, beta, n)
    return state, energies


def expected_energy(params: np.ndarray, Q: np.ndarray) -> float:
    gamma, beta = params
    state, energies = qaoa_state(Q, gamma, beta)
    probs = np.abs(state) ** 2
    return float(np.dot(probs, energies))


def solve_qaoa(
    Q: np.ndarray,
    cardinality: int,
    grid_gamma: int = 14,
    grid_beta: int = 10,
) -> QAOAResult:
    """QAOA p=1 bằng statevector NumPy cho PoC 8–12 qubit."""
    n = Q.shape[0]
    best = (np.inf, 0.0, 0.0)

    for gamma in np.linspace(0.0, 2 * np.pi, grid_gamma, endpoint=False):
        for beta in np.linspace(0.0, np.pi / 2, grid_beta):
            value = expected_energy(np.array([gamma, beta]), Q)
            if value < best[0]:
                best = (value, gamma, beta)

    opt = minimize(
        expected_energy,
        x0=np.array([best[1], best[2]]),
        args=(Q,),
        method="L-BFGS-B",
        bounds=[(0.0, 2 * np.pi), (0.0, np.pi / 2)],
        options={"maxiter": 80},
    )

    gamma, beta = opt.x
    state, energies = qaoa_state(Q, gamma, beta)
    probs = np.abs(state) ** 2
    bits = all_bitstrings(n)
    feasible = np.isclose(bits.sum(axis=1), cardinality)

    feasible_idx = np.where(feasible)[0]
    order = sorted(
        feasible_idx.tolist(),
        key=lambda i: (-float(probs[i]), float(energies[i])),
    )
    idx = int(order[0])

    top_idx = np.argsort(probs)[::-1][:10]
    top_states = [
        {
            "state": "".join(str(int(v)) for v in bits[i][::-1]),
            "probability": float(probs[i]),
            "energy": float(energies[i]),
            "cardinality": int(bits[i].sum()),
        }
        for i in top_idx
    ]

    return QAOAResult(
        bitstring=bits[idx].astype(int),
        probability=float(probs[idx]),
        expected_energy=float(np.dot(probs, energies)),
        gamma=float(gamma),
        beta=float(beta),
        top_states=top_states,
    )
