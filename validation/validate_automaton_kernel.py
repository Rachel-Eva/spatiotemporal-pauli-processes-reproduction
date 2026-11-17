"""
Exact one-round comparison of AutomatonBath with the paper's quantum QCA model, on ALL qubits at once.

After system twirling, the bath is a classical bit-string s at the start of every round, so one round is
fully described by
    (1) a bath-state kernel   T[s, x] = P(bath goes s -> x)       (128 x 128 for the 7-qubit lattice)
    (2) an emission table     w[i][k][x] = P(qubit i gets Pauli k | bath state x)

EXACT side (paper Eqs. 82, 84, 85, 87, 89, 91): built from the quantum model.
    T = S(a,b) kernel (from the Kraus operators, Eq. 84) @ |<x|W|s''>|^2, W = U_black<-red then U_red<-black,
    U = exp(-i phi sum_j X_j K_j).
    w comes from comb_operator applied to U_i = |0><0|_Ei (x) I + |1><1|_Ei (x) n.sigma (Eq. 89, 91);
    the script checks that the resulting operators are diagonal (the re-classicalisation of App. C.2).
PCA side: the same two objects computed exactly (no sampling) from AutomatonBath's own rules:
    a/b flips, black update with p_flip[k], red update with p_flip[k], emission with p_pauli.

Expected: exact phi = theta_code/2 matches to ~1e-12; phi = theta_code does not.
Only the lattice geometry is taken from the class. Needs dE = 2**n, so use distance 2 (7 qubits).
"""
import argparse
import os
import sys

import numpy as np
import stim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from noise.qca_process_tensor import Lattice, bits, exact_model, kron_all

def pca_model(L, a, b, theta):
    '''Exact (enumerated, not sampled) kernel of AutomatonBath's update rules.'''
    bt = L.bath
    n, dE = L.n, L.dE
    site = np.array([[1 - a, a], [b, 1 - b]])            # from 0: 1 w.p. a ; from 1: 0 w.p. b
    S = kron_all([site] * n)

    def half_step(targets):
        H = np.zeros((dE, dE))
        for y in range(dE):
            yb = bits(y, n)
            q = [bt.p_flip[sum(yb[i] for i in L.nbrs[j])] for j in targets]
            for pattern in range(2 ** len(targets)):
                pr, mask = 1.0, 0
                for t, j in enumerate(targets):
                    flip = (pattern >> t) & 1
                    pr *= q[t] if flip else 1 - q[t]
                    mask |= flip << (n - 1 - j)
                H[y, y ^ mask] += pr
        return H

    T = S @ half_step(L.black) @ half_step(L.red)
    w_tab = np.zeros((n, 4, dE))
    for x in range(dE):
        for i, bit in enumerate(bits(x, n)):
            w_tab[i, :, x] = [0, *bt.p_pauli] if bit else [1, 0, 0, 0]
    return T, w_tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=2)
    args = ap.parse_args()
    L = Lattice(stim.Circuit.generated("surface_code:rotated_memory_z", distance=args.distance, rounds=2))
    print(f"{L.n} bath qubits (dE={L.dE}); red={L.red}, black={L.black}")

    settings = [(0.3, 0.2, 0.4 * np.pi), (0.1, 0.5, 0.25 * np.pi), (1e-4, 0.5, 0.39 * np.pi)]
    ok = True
    for a, b, th in settings:
        L.bath.a, L.bath.b, L.bath.theta = a, b, th
        L.bath.p_flip = np.sin(np.arange(L.bath.nbr.shape[1] + 1) * th / 2) ** 2   # AutomatonBath's rule
        T_p, w_p = pca_model(L, a, b, th)
        T_h, w_h = exact_model(L, a, b, th / 2)     # rotation exp(-i (theta/2) X)
        T_f, w_f = exact_model(L, a, b, th)         # rotation exp(-i theta X): paper's text read literally
        rows = np.abs(T_h.sum(1) - 1).max()
        e_T, e_w = np.abs(T_h - T_p).max(), np.abs(w_h - w_p).max()
        m_T = np.abs(T_f - T_p).max()
        print(f"a={a:g} b={b:g} theta={th/np.pi:.2f}pi | phi=theta/2: max|dT|={e_T:.1e} max|dw|={e_w:.1e} (rows sum to 1 within {rows:.0e})"
              f" | phi=theta: max|dT|={m_T:.1e}")
        ok &= e_T < 1e-10 and e_w < 1e-10 and m_T > 1e-3
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()