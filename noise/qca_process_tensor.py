"""
Exact quantum model of the paper's QCA bath (Sec. 7.1, App. C) for small lattices, and a fast
sampler built from it.

  Lattice(circuit)           geometry (red/black split, neighbours) read from a Stim circuit
  exact_model(L, a, b, phi)  one-round bath-state kernel T[s, x] and emission table w[i, k, x],
                             built from the quantum model (Eqs. 82, 84, 85, 87, 89, 91):
                             rotation exp(-i phi sum_j X_j K_j), bath noise S(a,b), and the twirled
                             system-bath coupling via comb_operator. Needs dE = 2**n.
  QCAComb(T, w, ids, n_total) samples bath trajectories and Pauli labels from that kernel.
                             Same reset()/step() interface as AutomatonBath.
"""
import numpy as np

from noise.automaton_bath import AutomatonBath
from noise.process_tensor import I2, PAULI, comb_operator

P0, P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)


def kron_all(mats):
    out = np.eye(1)
    for m in mats:
        out = np.kron(out, m)
    return out


def bits(x, n):
    return [(x >> (n - 1 - i)) & 1 for i in range(n)]


class Lattice:
    def __init__(self, circuit):
        self.bath = AutomatonBath(circuit)
        b = self.bath
        self.n = b.num_qubits
        self.dE = 2 ** self.n
        self.ids = b.ids
        self.n_total = b.n_total
        self.red, self.black = [int(i) for i in b.red], [int(i) for i in b.black]
        self.nbrs = [[int(j) for j in b.nbr[i] if j < self.n] for i in range(self.n)]
        self.V1 = sum(c * PAULI[k + 1] for k, c in enumerate(np.sqrt(b.p_pauli)))   # n . sigma

    def op_on(self, q, m):
        return kron_all([m if i == q else I2 for i in range(self.n)])


def exact_model(L, a, b, phi):
    n, dE = L.n, L.dE
    kraus = [np.sqrt(a) * np.array([[0, 0], [1, 0]]), np.sqrt(1 - a) * np.diag([1, 0]),
             np.sqrt(b) * np.array([[0, 1], [0, 0]]), np.sqrt(1 - b) * np.diag([0, 1])]
    m2 = sum(np.abs(K) ** 2 for K in kraus).T          # m2[from, to]
    S = kron_all([m2] * n)

    n_op = [L.op_on(i, P1) for i in range(n)]
    x_op = [L.op_on(i, PAULI[1]) for i in range(n)]

    def rotation(targets):
        H = sum(x_op[j] @ sum(n_op[i] for i in L.nbrs[j]) for j in targets)
        w, v = np.linalg.eigh(H)
        return (v * np.exp(-1j * phi * w)) @ v.conj().T

    W = rotation(L.red) @ rotation(L.black)             # black targets first, then red targets
    T = S @ (np.abs(W) ** 2).T                          # T[s, x]

    w_tab = np.zeros((n, 4, dE))
    for i in range(n):
        M = np.kron(I2, L.op_on(i, P0)) + np.kron(L.V1, L.op_on(i, P1))
        A = comb_operator([M], dE)[:, 0]
        off = max(np.abs(A[k] - np.diag(np.diag(A[k]))).max() for k in range(4))
        assert off < 1e-12, f"A_k not diagonal for qubit {i} (off-diagonal {off:.1e}): bath is not re-classicalised"
        for k in range(4):
            w_tab[i, k] = np.abs(np.diag(A[k])) ** 2
    return T, w_tab


class QCAComb:
    '''
    Samples (bath state, Pauli labels) round by round from the exact kernel. After system twirling the
    bath is a classical bit-string at the start of each round, so one round is: draw the next bath state
    x from T[s, :], then draw each qubit's label from w[i, :, x]. step() returns an int64 array of length
    n_total (0=I, 1=X, 2=Y, 3=Z), non-zero only at the lattice qubits.
    '''
    def __init__(self, L, T, w_tab, rng=None):
        self.L = L
        self.cdf_T = np.cumsum(T, axis=1)
        self.cdf_w = np.cumsum(w_tab, axis=1)            # (n, 4, dE)
        self.bits = np.array([bits(x, L.n) for x in range(L.dE)])   # (dE, n)
        pi = np.linalg.matrix_power(T, 4000)[0]          # stationary distribution (T is a stochastic matrix)
        self.p_marg = float(pi @ self.bits.mean(axis=1))
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.s = 0

    def step(self):
        L = self.L
        x = min(int(np.searchsorted(self.cdf_T[self.s], self.rng.random() * self.cdf_T[self.s, -1])), L.dE - 1)
        u = self.rng.random(L.n)
        k = (u[:, None] * self.cdf_w[np.arange(L.n), -1, x][:, None] > self.cdf_w[np.arange(L.n), :, x]).sum(axis=1)
        k = np.minimum(k, 3)
        out = np.zeros(L.n_total, dtype=np.int64)
        out[L.ids] = k
        self.s = x
        return out
