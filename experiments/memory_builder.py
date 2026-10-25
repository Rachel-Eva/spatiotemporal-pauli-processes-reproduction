import stim
import numpy as np
from noise.temporal_storm import TemporalStorm
'''
Run a Monte Carlo simulation to prove that when the decoder is blind to time-memory, 
    it gets tricked and causes a spike in logical failures.
'''
def build_base_circuit(d: int, rounds: int, p_baseline: float) -> stim.Circuit:
    '''
    Creates a decorder for the rotated surface code.
    It utilizes the Detector error model which translates the circuit into a hypergraph.
    It generates the static graph that Pymatching uses to perform decoding.
    '''
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=rounds,
        distance=d,
        after_clifford_depolarization=p_baseline,
        after_reset_flip_probability=p_baseline,
        before_measure_flip_probability=p_baseline,
        before_round_data_depolarization=p_baseline
    )

def _inject(circuit, paulis):
    for name, code in (("X_ERROR",1),("Y_ERROR",2),("Z_ERROR",3)):
        idx = np.flatnonzero(paulis == code)
        if idx.size: # if the error list is not empty
            circuit.append(name, idx.tolist(), 1.0)

def compose_noisy_memory_circuit(base_circuit: stim.Circuit, storm: TemporalStorm) -> stim.Circuit:
    shot_circuit = stim.Circuit()
    storm.reset()
    first_tick_seen = False

    for item in base_circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            body = item.body_copy()
            for _ in range(item.repeat_count):
                _inject(shot_circuit, storm.step())
                shot_circuit += body
        else:
            shot_circuit.append(item)
            if item.name == "TICK" and not first_tick_seen: # one time noise injection for round 0
                first_tick_seen = True
                _inject(shot_circuit, storm.step())
    return shot_circuit

