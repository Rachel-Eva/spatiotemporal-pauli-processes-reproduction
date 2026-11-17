import numpy as np

I2 = np.eye(2, dtype=complex)
PAULI = np.stack([
    I2,
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex)
])

def env_only(kraus_env):
    '''
    Takes an environment only Kraus operator and returns the corresponding system+environment Kraus operators.
    Tracking the environment gives the correlations and decides the chance of the next error.
    '''
    return [np.kron(I2, B) for B in kraus_env]

def compose(kraus1, kraus2):
    '''
    Takes two sets of Kraus operators and returns the composed Kraus operators.
    '''
    return [np.dot(A, B) for A in kraus1 for B in kraus2]

def comb_operator(kraus, dE, tol=1e-14):
    '''
    A: the twirled process tensor. It encodes the noise model.
    A is a 4 dimensional array with the shape: (4, num_kraus, dE, dE)
        Axis 0: Pauli basis index (0: I, 1: X, 2: Y, 3: Z)
        Axis 1: Kraus operator index
        Axis 2: Environment dimension index (row)
        Axis 3: Environment dimension index (column)
    '''
    A = np.array([[np.einsum("st,tisj->ij", PAULI[k], M.reshape(2,dE,2,dE))/2 for M in kraus] for k in range(4)])
    A = A[:,np.abs(A).max(axis=(0,2,3)) > tol]
    completeness = sum(A[k,j].conj().T @ A[k,j] for k in range(4) for j in range(A.shape[1]))
    assert np.allclose(completeness, np.eye(dE)), "Kraus operators do not satisfy completeness relation"
    return A

def exact_distribution(rho0, n_rounds, A):
    '''
    rho0: initial environment density matrix, shape (dE, dE)
    n_rounds: number of rounds of noise to apply
    A: the twirled process tensor. It encodes the noise model.
        A is a 4 dimensional array with the shape: (4, num_kraus, dE, dE)
    The return value is the probability of each of the 4^N error sequences
    '''
    rho = rho0[None] # makes it a 3D array with shape (1, dE, dE) where 1 tracks the branching
    for _ in range(n_rounds):
        new = np.einsum("kjab,nbc,kjdc->nkad", A, rho, A.conj())
        rho = new.reshape(-1, *rho0.shape)
    return np.real(np.trace(rho, axis1=1, axis2=2))

class PauliComb:
    def __init__(self, A, rho0, num_qubits, rng=None):
        self.A = A
        self.num_qubits = num_qubits
        self.rho0 = rho0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        '''
        Resets all qubits to the initial state rho0. 
        '''
        self.rho = np.broadcast_to(self.rho0, (self.num_qubits, *self.rho0.shape)).copy()

    def step(self):
        '''
        R is the environment state after the round for each possible outcome k, one per qubit
            The shape of R is (4, num_qubits, dE, dE).
        p is the probability of each branch. It is the trace of R.
            The shape of p is (4, num_qubits).
        cdf is the cumulative distribution function of p over the 4 labels.
        n is the num_qubits
        k is the winning branch for each qubit.
        rho is the density matrix of the environment, with shape (num_qubits, dE, dE)
        '''
        R = np.einsum("kjab,nbc,kjdc->knad", self.A, self.rho, self.A.conj())
        p = np.real(np.trace(R, axis1=2, axis2=3)) # (4, n)
        cdf = np.cumsum(p, axis=0) # cumulative over the 4 labels
        u = self.rng.uniform(size=self.num_qubits)* cdf[-1]
        k = np.minimum((u[None]>cdf).sum(axis=0), 3)
        idx = np.arange(self.num_qubits)

        # Update the density matrix by keeping only the winning branch and normalizing
        self.rho = R[k, idx] / p[k, idx][:, None, None]
        return k.astype(np.int64)

    def estimate_marginal(self, n_steps=20_000, burn_in=2_000):
        self.reset()
        tot = 0.0
        for t in range(n_steps + burn_in):
            out = self.step()
            if t >= burn_in:
                tot += (out != 0).mean()
        self.reset()
        return float(tot / n_steps)

def storm_model(p, corr_len):
    '''
    Temporal storm as a process tensor: environment = one classical bit (0 calm, 1 storm).
    Round = Markov transition of the bit, then the system gets a uniformly random
    X/Y/Z error if the (new) bit is 1, nothing if 0. Same ordering as TemporalStorm.step().
    Returns (A, rho0) with rho0 the stationary environment state.
    '''
    delta = 1.0 if corr_len == 0 else 1.0 - np.exp(-1.0 / corr_len)
    a, b = delta * p, delta * (1 - p)
    T = np.array([[1 - a, a], [b, 1 - b]])
    ket = np.eye(2)
    B = [np.sqrt(T[s, s2]) * np.outer(ket[s2], ket[s]).astype(complex)
         for s in range(2) for s2 in range(2)]
    P0 = np.diag([1, 0]).astype(complex)
    P1 = np.diag([0, 1]).astype(complex)
    M = [np.kron(I2, P0)] + [np.kron(PAULI[k] / np.sqrt(3), P1) for k in (1, 2, 3)]
    A = comb_operator(compose(M, env_only(B)), dE=2)
    rho0 = np.diag([1 - p, p]).astype(complex)
    return A, rho0