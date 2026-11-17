"""
Check that the hand-written TemporalStorm equals the twirled process tensor of the storm.
  1. exact: process-tensor distribution vs an independent HMM forward computation
  2. sampling: TemporalStorm and PauliCombSampler vs the exact distribution
"""
import os
import sys
from itertools import product

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from noise.process_tensor import PauliComb, exact_distribution, storm_model
from noise.temporal_storm import TemporalStorm

p, corr_len, N, shots = 0.2, 3.0, 4, 400_000
rng = np.random.default_rng(1)


def hmm_distribution(p, corr_len, N):
    '''Independent computation straight from the storm definition (forward algorithm).'''
    delta = 1.0 - np.exp(-1.0 / corr_len)
    a, b = delta * p, delta * (1 - p)
    T = np.array([[1 - a, a], [b, 1 - b]])
    emit = np.array([[1, 0, 0, 0], [0, 1 / 3, 1 / 3, 1 / 3]])  # P(e | state)
    out = np.zeros(4 ** N)
    for seq in product(range(4), repeat=N):
        alpha = np.array([1 - p, p])
        for e in seq:
            alpha = (alpha @ T) * emit[:, e]
        out[int(np.ravel_multi_index(seq, (4,) * N))] = alpha.sum()
    return out


def encode(paulis):  # paulis: (N, shots) -> base-4 index, e_1 most significant
    idx = np.zeros(paulis.shape[1], dtype=np.int64)
    for row in paulis:
        idx = idx * 4 + row
    return idx


def empirical(sampler):
    sampler.reset()
    return np.bincount(encode(np.array([sampler.step() for _ in range(N)])), minlength=4 ** N) / shots


tvd = lambda x, y: 0.5 * np.abs(x - y).sum()

A, rho0 = storm_model(p, corr_len)
P_pt = exact_distribution(rho0, N, A)
P_hmm = hmm_distribution(p, corr_len, N)
print(f"sum P = {P_pt.sum():.12f}")
print(f"[exact]    TVD(process tensor, HMM) = {tvd(P_pt, P_hmm):.2e}   (want ~1e-16)")

base = np.bincount(rng.choice(4 ** N, size=shots, p=P_hmm), minlength=4 ** N) / shots
t_base = tvd(base, P_hmm)
t_storm = tvd(empirical(TemporalStorm(p, corr_len, shots, rng)), P_hmm)
t_pt = tvd(empirical(PauliComb(A, rho0, shots, rng)), P_hmm)
print(f"[sampling] TVD to exact, {shots:,} shots, N={N}:")
print(f"           sampling-noise baseline = {t_base:.4f}")
print(f"           TemporalStorm           = {t_storm:.4f}")
print(f"           PauliComb               = {t_pt:.4f}")
ok = t_pt < 2 * t_base and t_storm < 2 * t_base and tvd(P_pt, P_hmm) < 1e-12
print("PASS" if ok else "FAIL")
