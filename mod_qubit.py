"""
Trains a single-qubit quantum circuit to modify a qubit from one value to another.
"""

from gradient3 import *

START_VALUE = ket('0')
END_VALUE = ket('+')

rot90 = np.array([
  [0, -1],
  [1,  0],
])

if __name__ == '__main__':
  qca = [[0]]
  msmt_qubits = [0]

  err_vec = np.dot(rot90, END_VALUE)
  training_set = [(START_VALUE, err_vec)]

  circuit, loss = optimize_qca(qca, msmt_qubits, training_set, num_iterations = 50, step_size = 0.4)
  print("Final circuit (loss = %f): " % loss)
  print(circuit)

  sim = cirq.Simulator()
  print("Final vector:")
  print(sim.simulate(circuit, initial_state = START_VALUE).dirac_notation())
