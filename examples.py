from gradient3 import *

if __name__ == '__main__':
  np.set_printoptions(precision=3, suppress=True)
  qubits = [cirq.LineQubit(i) for i in range(3)]
  os2 = 1/np.sqrt(2)
  ket0 = np.matrix([[1],[0]], dtype = np.complex64)
  ket1 = np.matrix([[0],[1]], dtype = np.complex64)
  ketp = os2 * (ket0 + ket1)
  ketm = os2 * (ket0 - ket1)

  U1 = unitary_to_gate(scipy.stats.unitary_group.rvs(4))(*qubits[:2])
  U2 = unitary_to_gate(scipy.stats.unitary_group.rvs(4))(*qubits[1:])
  circuit = cirq.Circuit()
  circuit.append(U1)
  circuit.append(U2)

  # try to learn to answer with the negation of the first qubit
  zero_to_one = np.kron(ket0, np.kron(ket0, ket0)),  np.array(ket1.T)
  one_to_zero = np.kron(ket1, np.kron(ket0, ket0)),  np.array(ket0.T)
  training_set = [zero_to_one, one_to_zero]
  msmt_qubits = [2]

  gss = []
  losses = []
  for i in range(50):
    gradients, loss = avg_gradient(circuit, msmt_qubits, training_set)
    gss.append(gradients)
    losses.append(loss)
    circuit = step(circuit, gradients, 1)

  simulator = cirq.Simulator()
  foo = simulator.simulate(circuit, initial_state = np.array([1,0,0,0,0,0,0,0]))
