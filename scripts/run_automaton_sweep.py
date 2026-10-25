import sys
import os
import numpy as np
import json
from multiprocessing import Pool, cpu_count

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")

from decoding.matched_decoder import create_matched_decoder
from experiments.memory_builder import build_base_circuit, compose_noisy_memory_circuit
from noise.automaton_bath import AutomatonBath

def _simulate_chunk_worker(args):
    '''
    Helps perform parallel processing.
    Divides the total number of shots into chunks for each worker to process.
    Each worker simulates a chunk of shots and returns the number of failures in that chunk.
    Returns:
        chunk_fails: int - number of failures in the chunk
        chunk_shots: int - number of shots in the chunk
    '''
    d, p, theta, p_marg, chunk_shots, seed = args
    rng = np.random.default_rng(seed)
    rounds = 3 * d
    base_circuit = build_base_circuit(d=d, rounds=rounds, p_baseline=p)
    decoder = create_matched_decoder(distance=d, rounds=rounds, p=p, p_marg=p_marg)
    bath = AutomatonBath(base_circuit, theta=theta, rng=rng)
    chunk_fails = 0
    for _ in range(chunk_shots):
        shot_circuit = compose_noisy_memory_circuit(base_circuit, bath)
        sampler = shot_circuit.compile_detector_sampler()
        detectors, actual_obs = sampler.sample(shots=1, separate_observables=True)
        predicted_obs = decoder.decode_batch(detectors)
        if np.any(predicted_obs != actual_obs):
            chunk_fails += 1
    return chunk_fails, chunk_shots

def run_single_benchmark(d: int, p: float, theta: float, num_shots: int, num_workers: int, chunk_size: int) -> dict:
    rounds = 3 * d

    chunk_sizes = [chunk_size] * (num_shots // chunk_size) # stores same chunk size for each worker
    if num_shots % chunk_size != 0:
        chunk_sizes.append(num_shots % chunk_size)
    # distribute the work across multiple workers using multiprocessing
    # gives every single worker its own random number sequence
    seeds = np.random.SeedSequence().spawn(len(chunk_sizes))

    # marginal error rate of the bath at this theta, used by the matched decoder
    p_marg = AutomatonBath(build_base_circuit(d, rounds, p), theta=theta).estimate_marginal(n_steps=100_000)
    worker_tasks = [(d, p, theta, p_marg, size, s) for size, s in zip(chunk_sizes, seeds)]

    fails = 0
    completed_shots = 0
    print(f"--> Simulating: d={d}, Nr={rounds}, p={p}, theta={theta/np.pi:.3f}pi, p_marg={p_marg:.4e}")
    print(f" Dispatching {num_shots:,} shots across {num_workers} parallel workers ({len(chunk_sizes)} chunks)...")

    with Pool(processes=num_workers) as pool:
        for chunk_fails,actual_chunk_shots in pool.imap(_simulate_chunk_worker, worker_tasks):
            fails += chunk_fails
            completed_shots += actual_chunk_shots
            print(f" Progress: {completed_shots}/{num_shots} shots completed | Current fails: {fails}", end="\r")

    p_shot = fails / num_shots

    if 2.0 * p_shot >= 1.0:
        p_round = 0.5 # above threshold so value capped
    else:
        p_round = (1.0 - (1.0 - 2 * p_shot) ** (1.0 / rounds)) / 2.0
    print(f" Result: {fails}/{num_shots} failures | p_shot = {p_shot:.4e} | p_round = {p_round:.4e}\n")

    return {
        "distance": d,
        "rounds": rounds,
        "p": p,
        "theta": theta,
        "p_marg": p_marg,
        "num_shots": num_shots,
        "num_fails": fails,
        "p_shot": p_shot,
        "p_round": p_round,
        "p_shot_err": float(np.sqrt(p_shot * (1 - p_shot) / num_shots))
    }

def save_checkpoint(results: list, filename: str):
    '''
    Performs atomic file writing.
    '''
    tmp_path = filename + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=4)
    os.replace(tmp_path, filename)

def load_checkpoint(filename: str) -> list:
    '''
    Loads results from a checkpoint file if it exists.
    '''
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []

def is_already_computed(results: list, d: int, theta: float, p: float, num_shots: int) -> bool:
    return any(
        e["distance"] == d
        and np.isclose(e["theta"], theta, rtol=1e-5)
        and np.isclose(e["p"], p)
        and e["num_shots"] == num_shots
        for e in results
    )

def main():
    p_error = 0.001
    distances = [5, 7, 9, 11, 13, 15, 17]
    thetas = [t * np.pi for t in (0.2, 0.25, 0.3, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.48)]
    num_shots = 200_000
    chunk_size = 2000          # keep num_shots a multiple of this
    output_filename = "automaton_sweep.json"
    num_workers = max(1, cpu_count() - 1) # uses all cores except one

    # --- smoke test: uncomment these, run once, then delete smoke_automaton.json ---
    # distances = [5, 7]
    # thetas = [0.2 * np.pi, 0.48 * np.pi]
    # num_shots = 5_000
    # chunk_size = 500
    # output_filename = "smoke_automaton.json"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, output_filename)
    all_results = load_checkpoint(output_path)
    print(f" Results file: {output_path}")

    print(f" Baseline p: {p_error}")
    print(f" Distances: {distances}")
    print(f" Thetas (x pi): {[round(t / np.pi, 3) for t in thetas]}")
    print(f" Shots per point: {num_shots:,}")

    total_points = len(distances) * len(thetas) # calculates total parameter combinations
    current_point = 0

    for d in distances:
        for theta in thetas:
            current_point += 1
            if is_already_computed(all_results, d, theta, p_error, num_shots):
                print(f"Skipping: d={d}, theta={theta/np.pi:.3f}pi (Already computed)")
                continue
            print(f"[{current_point}/{total_points}] Running: d={d} (Rounds={3*d}) | theta={theta/np.pi:.3f}pi")

            point_result = run_single_benchmark(
                d=d,
                p=p_error,
                theta=theta,
                num_shots=num_shots,
                num_workers=num_workers,
                chunk_size=chunk_size
            )
            # Atomic checkpointing
            all_results.append(point_result)
            save_checkpoint(all_results, output_path)
    
    print(f"\n[+] Sweep complete! Results saved to '{output_path}'")

if __name__ == "__main__":
    main()