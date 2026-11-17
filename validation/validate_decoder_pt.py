"""
Decoder-level check: run TemporalStorm and the process-tensor sampler (PauliComb) through the
same d=3 memory circuit and the same matched decoder, and compare logical failure counts.
If the two samplers describe the same noise, the failure rates agree within statistical error.

    python scripts/validate_decoder_pt.py                 # defaults
    python scripts/validate_decoder_pt.py --shots 20000 --corr-lens 1 8
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from decoding.matched_decoder import create_matched_decoder
from experiments.memory_builder import build_base_circuit, compose_noisy_memory_circuit
from noise.process_tensor import PauliComb, storm_model
from noise.temporal_storm import TemporalStorm


def count_failures(base_circuit, decoder, noise, shots):
    fails = 0
    for _ in range(shots):
        shot_circuit = compose_noisy_memory_circuit(base_circuit, noise)
        detectors, actual_obs = shot_circuit.compile_detector_sampler().sample(shots=1, separate_observables=True)
        if np.any(decoder.decode_batch(detectors) != actual_obs):
            fails += 1
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=8000)
    ap.add_argument("--corr-lens", type=float, nargs="+", default=[1.0, 8.0])
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    d, p, shots = args.distance, args.p, args.shots
    rounds = 3 * d
    rng = np.random.default_rng(args.seed)
    base = build_base_circuit(d=d, rounds=rounds, p_baseline=p)
    decoder = create_matched_decoder(distance=d, rounds=rounds, p=p)   # p_marg = p, exact for the storm
    nq = base.num_qubits

    ok = True
    for xi in args.corr_lens:
        A, rho0 = storm_model(p, xi)
        f_storm = count_failures(base, decoder, TemporalStorm(p=p, corr_len=xi, num_qubits=nq, rng=rng), shots)
        f_pt = count_failures(base, decoder, PauliComb(A, rho0, nq, rng), shots)
        p1, p2 = f_storm / shots, f_pt / shots
        pool = (f_storm + f_pt) / (2 * shots)
        se = np.sqrt(2 * pool * (1 - pool) / shots)
        z = (p1 - p2) / se if se > 0 else 0.0
        print(f"xi={xi:<5g} TemporalStorm {f_storm}/{shots} ({p1:.4f})   PauliComb {f_pt}/{shots} ({p2:.4f})   z = {z:+.2f}")
        ok &= abs(z) < 3
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
