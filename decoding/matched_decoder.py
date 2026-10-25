import pymatching

import sys
import os
import stim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.memory_builder import build_base_circuit

def create_matched_decoder(distance: int, rounds: int, p: float, p_marg=None) -> pymatching.Matching:
    '''
    Creates a decoder for the rotated surface code.
    It utilizes the Detector error model which translates the circuit into a hypergraph.
    It generates the static graph that Pymatching uses to perform decoding.

    p is the baseline circuit-level noise rate.
    p_marg is the marginal error rate of the correlated noise (DEPOLARIZE1 per round).
        Defaults to p, which is exact for the temporal storm.
        For the automaton bath, pass AutomatonBath.estimate_marginal().
    '''
    if p_marg is None:
        p_marg = p
    base_circuit = build_base_circuit(d = distance, rounds = rounds, p_baseline = p)
    circuit = stim.Circuit()
    n = base_circuit.num_qubits
    qubits = list(range(n))
    first_tick_seen = False
    for item in base_circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            body = stim.Circuit()
            body.append("DEPOLARIZE1", qubits, p_marg)
            body += item.body_copy()
            circuit.append(stim.CircuitRepeatBlock(item.repeat_count, body))
        else:
            circuit.append(item)
            if item.name == "TICK" and not first_tick_seen: # round 0, once only
                first_tick_seen = True
                circuit.append("DEPOLARIZE1", qubits, p_marg)
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)