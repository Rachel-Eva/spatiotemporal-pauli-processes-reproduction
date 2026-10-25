import numpy as np

class TemporalStorm:
    def __init__(self, p:float, corr_len:float, num_qubits:int, rng=None):
        '''
        a is the transition probability from calm to stormy.
        b is the transition probability from stormy to calm.
        delta is the spectral gap denoting how fast the system forgets its past.
            a small spectral gap indicates longer memory retention and persistant storm.
            a large spectral gap indicates shorter memory retention and less persistant storm.
        p is the probability of being in a stormy state.
        corr_len is the correlation length denoting how long the system remains in a given state.
        states is a list of the two possible states of the system: calm and stormy.
            calm: 0
            stormy: 1
        
        corr_len is the inverse of the spectral gap
        '''
        self.delta = 1.0 if corr_len == 0 else 1.0 - np.exp(-1.0 / corr_len) # eqn. 79
        self.p = p

        self.a = self.delta * p
        self.b = self.delta * (1 - p)
        self.corr_len = corr_len
        self.num_qubits = num_qubits
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        '''
        Reset the state of the system to its stationary state distribution
        '''
        self.states = (self.rng.random(self.num_qubits) < self.p).astype(np.int8)

    def step(self) -> list[str]:
        '''
        Utilizing vectorized operations for efficiency
        It calculates the next state of the system based on the current state and transition probabilities.
        '''
        rand = self.rng.random(self.num_qubits) # array of random numbers from 0.0 to 1.0
        # update states based on transition condition
        self.states = (np.where(self.states == 0, rand < self.a, rand >= self.b)).astype(np.int8)
        # a randomized pauli list is created
        paulis = self.rng.integers(1, 4, size=self.num_qubits)
        return np.where(self.states == 1, paulis, 0)