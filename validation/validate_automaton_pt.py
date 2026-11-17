"""
Check AutomatonBath against an exact quantum simulation of the paper's QCA bath (Sec. 7.1, App. C).

Quantum model (paper Eqs. 82, 84, 85, 87, 89, 91), per round, bath dimension 2**n:
    1. bath-local noise  S(a,b): 0->1 w.p. a, 1->0 w.p. b                      (Eq. 84)
    2. red-control / black-target rotations  prod_j exp(-i phi X_j K_j)         (Eq. 85, 87)
    3. black-control / red-target rotations                                      (Eq. 85, 87)
    4. system-bath coupling  U_i = |0><0|_Ei (x) I + |1><1|_Ei (x) (n.sigma)_Si (Eq. 89, 91)
       system qubits are Pauli-twirled every round.
K_j = number of excited neighbours of j. Bath starts in |0...0>.

The twirl is NOT assumed to dephase the bath: for every system qubit other than the one we
watch, the twirled environment map is computed from U_i with comb_operator (the same A_k
formula as the storm), and its labels are summed over. The watched qubit's Pauli labels over
N rounds are then enumerated exactly and compared with AutomatonBath.

AutomatonBath uses flip probability sin^2(k*theta_code/2); the paper's text has sin^2(k*phi) for
exp(-i phi X). We run the exact model with phi = theta_code/2 (expect a match) and phi = theta_code
(expect a mismatch), which pins down the convention.
Only the lattice geometry (red/black split, neighbours) is taken from the class.
"""
import os
import sys

import numpy as np
import stim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from noise.automaton_bath import AutomatonBath
from noise.process_tensor import I2, PAULI, comb_operator

a, b = 0.3, 0.2                  # large on purpose so the bath is populated within a few rounds
theta_code = 0.4 * np.pi         # the theta passed to AutomatonBath
N, shots = 3, 60_000
rng = np.random.default_rng(7)

circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=2, rounds=2)
bath = AutomatonBath(circuit, a=a, b=b, theta=theta_code, rng=rng)
n = bath.num_qubits
dE = 2 ** n
red, black = list(bath.red), list(bath.black)
nbrs = [[j for j in bath.nbr[i] if j < n] for i in range(n)]
target = int(np.argmax([len(x) for x in nbrs]))   # watch the best-connected qubit
n_vec = np.ones(3) / np.sqrt(3)
V1 = sum(c * PAULI[k + 1] for k, c in enumerate(n_vec))
print(f"{n} bath qubits (dE={dE}), watching qubit {target} with {len(nbrs[target])} neighbours, N={N}")

P0, P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
X = PAULI[1]


def op_on(q, m):
    out = np.eye(1, dtype=complex)
    for i in range(n):
        out = np.kron(out, m if i == q else I2)
    return out


n_op = [op_on(i, P1) for i in range(n)]
x_op = [op_on(i, X) for i in range(n)]


def rotation(phi, targets):
    '''exp(-i phi sum_{j in targets} X_j K_j), K_j = sum of n_i over neighbours i of j.'''
    H = sum(x_op[j] @ sum(n_op[i] for i in nbrs[j]) for j in targets)
    w, v = np.linalg.eigh(H)
    return (v * np.exp(-1j * phi * w)) @ v.conj().T


# bath-local noise S(a,b): Kraus operators (Eq. 84) on every bath qubit
ab_kraus = [np.sqrt(a) * np.array([[0, 0], [1, 0]]), np.sqrt(1 - a) * np.diag([1, 0]),
            np.sqrt(b) * np.array([[0, 1], [0, 0]]), np.sqrt(1 - b) * np.diag([0, 1])]
ab_ops = [[op_on(q, L.astype(complex)) for L in ab_kraus] for q in range(n)]

# twirled environment maps from the system-bath coupling U_i (derived, not assumed)
A_sys = []
for i in range(n):
    M = np.kron(I2, op_on(i, P0)) + np.kron(V1, op_on(i, P1))
    A_sys.append(comb_operator([M], dE))                       # (4, 1, dE, dE)
others = [A_sys[i][:, 0] for i in range(n) if i != target]     # summed over labels below


def conj(L, rho):
    return L @ rho @ L.conj().T   # matmul broadcasts over the branch axis


def exact(phi):
    U_rb = rotation(phi, black)   # red controls, black targets (first half-step, as in AutomatonBath)
    U_br = rotation(phi, red)     # black controls, red targets (second half-step)
    W = U_br @ U_rb
    rho = np.zeros((1, dE, dE), dtype=complex)
    rho[0, 0, 0] = 1.0
    for _ in range(N):
        for q in range(n):                                    # S(a,b)
            rho = sum(conj(L, rho) for L in ab_ops[q])
        rho = conj(W, rho)                                    # QCA half-steps
        for Ai in others:                                     # other system qubits twirled: sum over labels
            rho = sum(conj(Ai[k], rho) for k in range(4))
        At = A_sys[target][:, 0]
        rho = np.stack([conj(At[k], rho) for k in range(4)], axis=1).reshape(-1, dE, dE)  # branch on the watched label
    return np.real(np.trace(rho, axis1=1, axis2=2))


def empirical():
    idx = np.zeros(shots, dtype=np.int64)
    seqs = np.zeros((N, shots), dtype=np.int64)
    for s in range(shots):
        bath.reset()
        for t in range(N):
            seqs[t, s] = bath.step()[bath.ids[target]]
    for row in seqs:
        idx = idx * 4 + row
    return np.bincount(idx, minlength=4 ** N) / shots


tvd = lambda x, y: 0.5 * np.abs(x - y).sum()

P_half = exact(theta_code / 2)    # exp(-i (theta/2) X): matches AutomatonBath's sin^2(k theta/2)
P_full = exact(theta_code)        # exp(-i theta X):     the paper's text read literally
emp = empirical()
base = np.bincount(rng.choice(4 ** N, size=shots, p=P_half), minlength=4 ** N) / shots

print(f"sum P = {P_half.sum():.12f}")
print(f"TVD(exact phi=theta/2 , AutomatonBath) = {tvd(P_half, emp):.4f}   <- want ~ baseline")
print(f"TVD(exact phi=theta   , AutomatonBath) = {tvd(P_full, emp):.4f}   <- should be clearly larger")
print(f"sampling-noise baseline ({shots:,} shots)  = {tvd(base, P_half):.4f}")
ok = tvd(P_half, emp) < 2 * tvd(base, P_half) and tvd(P_full, emp) > 3 * tvd(base, P_half)
print("PASS" if ok else "FAIL")
