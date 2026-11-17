"""
Decoder-level check for the QCA bath: AutomatonBath vs the sampler built from the exact quantum model
(QCAComb), both through the same d=2 memory circuit and the same matched decoder.
d=2 only (7 bath qubits, dE=128); larger lattices are out of reach for an exact bath.

    python scripts/validate_decoder_automaton_pt.py
    python scripts/validate_decoder_automaton_pt.py --shots 20000 --a 0.02 --b 0.3 --theta 0.4
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from decoding.matched_decoder import create_matched_decoder
from experiments.memory_builder import build_base_circuit, compose_noisy_memory_circuit
from noise.automaton_bath import AutomatonBath
from noise.qca_process_tensor import Lattice, QCAComb, exact_model


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
    ap.add_argument("--p", type=float, default=0.001, help="baseline circuit noise")
    ap.add_argument("--a", type=float, default=0.02)
    ap.add_argument("--b", type=float, default=0.3)
    ap.add_argument("--theta", type=float, default=0.4, help="theta / pi, AutomatonBath convention")
    ap.add_argument("--shots", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    d, rounds, theta = 2, 6, args.theta * np.pi
    rng = np.random.default_rng(args.seed)
    base = build_base_circuit(d=d, rounds=rounds, p_baseline=args.p)

    L = Lattice(base)
    T, w = exact_model(L, args.a, args.b, theta / 2)          # exact quantum kernel, rotation exp(-i (theta/2) X)
    pt = QCAComb(L, T, w, rng)
    bath = AutomatonBath(base, a=args.a, b=args.b, theta=theta, rng=rng)
    decoder = create_matched_decoder(distance=d, rounds=rounds, p=args.p, p_marg=pt.p_marg)
    print(f"d={d}, rounds={rounds}, a={args.a}, b={args.b}, theta={args.theta}pi, p_marg={pt.p_marg:.4f}")

    f_bath = count_failures(base, decoder, bath, args.shots)
    f_pt = count_failures(base, decoder, pt, args.shots)
    p1, p2 = f_bath / args.shots, f_pt / args.shots
    pool = (f_bath + f_pt) / (2 * args.shots)
    se = np.sqrt(2 * pool * (1 - pool) / args.shots)
    z = (p1 - p2) / se if se > 0 else 0.0
    print(f"AutomatonBath {f_bath}/{args.shots} ({p1:.4f})   QCAComb {f_pt}/{args.shots} ({p2:.4f})   z = {z:+.2f}")
    print("PASS" if abs(z) < 3 and min(f_bath, f_pt) > 20 else "FAIL (or too few failures to compare: raise --shots / --a)")


if __name__ == "__main__":
    main()