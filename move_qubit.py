"""
Trains a quantum circuit to "move" a qubit, taking its input on the first qubit and outputting it on
the last qubit (optionally negating the qubit in the process). For instance, if NUM_QUBITS is 4,
then we have a circuit that looks like:
 [input]--------┌  ┐--------------
          ------└  ┘--┌  ┐--------
          ------------└  ┘--┌  ┐--
          ------------------└  ┘----[output]
"""
from optimize_qca import *

NUM_QUBITS = 3
NEGATE_QUBIT = False

if __name__ == '__main__':
  qca = [[i, i+1] for i in range(NUM_QUBITS-1)]
  msmt_qubits = [NUM_QUBITS-1]

  # recall that each entry of the training set is of the form (input, error), where input is the
  # total input to the circuit, and error indicates bad answers
  ket0s = ket('0'*NUM_QUBITS)
  ket1s = ket('1' + '0'*(NUM_QUBITS-1))
  training_set = [
    (ket0s, ket('0' if NEGATE_QUBIT else '1')),
    (ket1s, ket('1' if NEGATE_QUBIT else '0')),
  ];

  circuit, loss = optimize_qca(qca, msmt_qubits, training_set, num_iterations = 100, step_size = 0.4)
  print("Final circuit (loss = %f): " % loss)
  print(circuit)

  sim = cirq.Simulator()
  print("Final vector from |%s>:" % ('0'*NUM_QUBITS))
  print(sim.simulate(circuit, initial_state = ket0s).dirac_notation())
  print("Final vector from |%s>:" % ('1' + '0'*(NUM_QUBITS-1)))
  print(sim.simulate(circuit, initial_state = ket1s).dirac_notation())

  last_qubit = [q for q in circuit.all_qubits() if q.x == NUM_QUBITS-1][0]
  circuit.append(cirq.measure(last_qubit, key='result'))
  samples = sim.run(circuit, repetitions = 1000)
  print("Histogram from |%s>:" % ('0'*NUM_QUBITS))
  print(samples.histogram(key='result'))
