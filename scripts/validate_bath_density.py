import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.memory_builder import build_base_circuit
from noise.automaton_bath import AutomatonBath

base = build_base_circuit(9, 9, 0.001)
for th in np.array([0, .2, .3, .35, .39, .45, .6]) * np.pi:
    bath = AutomatonBath(base, theta=th)
    eta = []
    for _ in range(60_000):
        bath.step()
        eta.append(bath.s[:-1].mean())
    eta = np.array(eta[10_000:])
    print(f"theta={th/np.pi:.2f}pi  <eta>={eta.mean():.2e}  N*Var={bath.num_qubits*eta.var():.3f}")