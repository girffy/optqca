import numpy as np
import cirq
from clean import *

# 1-qubit example
'''
if __name__ == '__main__':
  np.set_printoptions(precision=3, suppress=True)
  os2 = 1/np.sqrt(2)
  qubit = cirq.LineQubit(0)
  ket0 = np.matrix([[1],[0]], dtype = np.complex64)
  ket1 = np.matrix([[0],[1]], dtype = np.complex64)
  ketp = os2 * (ket0 + ket1)
  ketm = os2 * (ket0 - ket1)
  #inp = rotket(np.random.rand()*np.pi*2)
  inp = ket0
  bad = ket0
  err_vec = bad

  #init_unitary = np.eye(2)
  init_unitary = scipy.stats.unitary_group.rvs(2)
  #init_unitary = os2*np.matrix([[1, 1], [1,-1]], dtype=np.complex64)
  #init_unitary = np.matrix([[0, 1], [1,0]], dtype=np.complex64)
  #init_unitary = np.array([[-0.040-0.348j, -0.539+0.766j], [-0.311+0.884j, -0.307+0.169j]])
  X = la.logm(init_unitary)

  def lossfn(X):
    #cout = np.matmul(la.expm(X), inp)
    #return in_prod(bad_povm, cout * cout.getH())
    return 0

  from est_grad import *
  cir = cirq.Circuit()
  cir.append(unitary_to_gate(init_unitary)(qubit))
  losses = []
  Xs = []
  gss = []
  egs = []
  for i in range(15):
    X, err = la.logm(get_unitaries(cir)[0], disp=False)
    print "\n@@@ X is: @@@"
    print X
    #losses.append(lossfn(X))
    Xs.append(X)
    eg = [est_grad(lossfn, X, .001)]
    gradients, loss = loss_gradient(cir, inp, [0], err_vec)
    losses.append(loss)
    gss.append(gradients[0])
    egs.append(eg)
    print "eg is:"
    print eg
    cir = step(cir, gradients, 1)
    #cir = step(cir, eg, 1)
  print "\nlosses:"
  print losses
'''

# 3-qubit example
'''
if __name__ == '__main__':
  np.set_printoptions(precision=3, suppress=True)
  qubits = [cirq.LineQubit(i) for i in range(3)]
  os2 = 1/np.sqrt(2)
  ket0 = np.matrix([[1],[0]], dtype = np.complex64)
  ket1 = np.matrix([[0],[1]], dtype = np.complex64)
  ketp = os2 * (ket0 + ket1)
  ketm = os2 * (ket0 - ket1)

  inp = np.kron(ketp, np.kron(ketp, ketp))
  msmt_qubits = [2]
  bad_povm = np.array([[1,0],[0,0]])

  U1 = unitary_to_gate(scipy.stats.unitary_group.rvs(4))(*qubits[:2])
  U2 = unitary_to_gate(scipy.stats.unitary_group.rvs(4))(*qubits[1:])
  #U1 = unitary_to_gate(np.eye(4))(*qubits[:2])
  #U2 = unitary_to_gate(np.eye(4))(*qubits[1:])
  circuit = cirq.Circuit()
  circuit.append([U1, U2])

  # try to learn to answer with the negation of the first qubit
  zero_to_one = (np.kron(ket0, np.kron(ket0, ket0)),
                 np.array([[0,0],[0,1]], dtype=np.complex64))
  one_to_zero = (np.kron(ket1, np.kron(ket1, ket1)),
                 np.array([[1,0],[0,0]], dtype=np.complex64))
  training_set = [zero_to_one, one_to_zero]

  gss = []
  losses = []
  #gradients, loss = loss_gradient(circuit, inp, msmt_qubits, bad_povm)
  #gradients1, loss1 = loss_gradient(circuit, zero_to_one[0], msmt_qubits, zero_to_one[1])
  #gradients2, loss2 = loss_gradient(circuit, one_to_zero[0], msmt_qubits, one_to_zero[1])
  for i in range(50):
    gradients, loss = avg_gradient(circuit, msmt_qubits, training_set)
    gss.append(gradients)
    losses.append(loss)
    circuit = step(circuit, gradients, 1)
'''

# paper example
if __name__ == '__main__':
  np.set_printoptions(precision=3, suppress=True)
  #np.set_printoptions(precision=8, suppress=False)

  os2 = 1/np.sqrt(2)
  ket0 = np.array([1,0], dtype = np.complex64)
  ket1 = np.array([0,1], dtype = np.complex64)
  ketp = os2 * (ket0 + ket1)
  ketm = os2 * (ket0 - ket1)
  #init_unitary = scipy.stats.unitary_group.rvs(2)
  msmt_qubits = [16]
  nqubits = 17

  inp = ket0
  for i in range(16):
    inp = np.kron(inp, ket0)

  qubits = [cirq.LineQubit(i) for i in range(17)]
  circuit = cirq.Circuit()
  for layer in range(6):
    for lq in range(16):
      foo = False
      # TODO: remove this
      if layer == lq == 0:
        #U = scipy.stats.unitary_group.rvs(4)
        U = np.eye(4)
      else:
        #U = np.eye(4)
        U = scipy.stats.unitary_group.rvs(4)
      gate = unitary_to_gate(U)
      op = gate(qubits[lq], qubits[16])
      circuit.append(op)

  povm_matr = .5 * np.matrix([[1,0], [0,0]], dtype=np.complex64)

  def lossfn(vec):
    shfl_fs = rekron(vec, nqubits, msmt_qubits)
    final_mq_state = partial_trace(shfl_fs, shfl_fs, len(msmt_qubits))
    loss = in_prod(povm_matr, final_mq_state)
    return loss

  gss = []
  losses = []
  circuits = []
  fus = []
  for i in range(10000):
    print("i=%s: " % i,)
    gradients, loss = loss_gradient(circuit, inp, msmt_qubits, povm_matr)
    gss.append(gradients)
    losses.append(loss)
    circuits.append(circuit)
    circuit = step(circuit, gradients, 1, aim=None)
    fus.append(get_unitaries(circuits[-1])[0])
    #print('  grad:')
    #print(gradients[0])


'''
# 1-qubit serial example
if __name__ == '__main__':
  qubit = cirq.LineQubit(0)
  numgates = 10
  #Us = [scipy.stats.unitary_group.rvs(2) for i in range(numgates)]
  Us = [np.eye(2) for i in range(numgates)]
  Us[0] = np.eye(2)
  #Us = [np.eye(2)]
  #Us = [np.eye(2), np.array([[ 0.329-0.408j, -0.848+0.075j], [ 0.372+0.766j, -0.180+0.493j]])]
  ops = [unitary_to_gate(U)(qubit) for U in Us]
  ket0 = np.array([1,0], dtype = np.complex64)
  inp = ket0
  msmt_qubits = [0]
  povm_matr = .5 * np.matrix([[1,1], [1,1]], dtype=np.complex64)
  #povm_matr = np.matrix([[1,0], [0,0]], dtype=np.complex64)

  def lossfn(Y, err=False):
    #cout = np.matmul(Us[1], la.expm(X), inp)
    cout = np.matrix(np.matmul(np.matmul(la.expm(Y), Us[1]), inp))
    ip = in_prod(povm_matr, cout.getH() * cout)
    if err:
      raise
      pass
    return ip
  X = la.logm(Us[0])
  #eg = est_grad(lossfn, X, .001)

  circuit = cirq.Circuit()
  circuit.append(ops)
  losses = []
  gss = []
  for i in range(10):
    gradients, loss = loss_gradient(circuit, inp, msmt_qubits, povm_matr)
    print 'gradient:'
    print gradients[0]
    circuit = step(circuit, gradients, 1, aim=None)
    losses.append(loss)
    gss.append(gradients)

  #g2, l2 = loss_gradient(nc, inp, msmt_qubits, povm_matr)
  #print '%s -> %s  (change: %s)' % (loss, l2, l2-loss)

  #egs = [.5 * (eg - np.matrix(eg).getH()), 0*eg]
  #anc = step(circuit, egs, .1, aim=None)

  #ag2, al2 = loss_gradient(anc, inp, msmt_qubits, povm_matr)

  #X2 = la.logm(get_unitaries(nc)[0])
  #aX2 = la.logm(get_unitaries(anc)[0])
  '''

# 2-qubit serial example
'''
if __name__ == '__main__':
  q0, q1 = cirq.LineQubit(0), cirq.LineQubit(1)
  numgates = 10
  Us = [scipy.stats.unitary_group.rvs(4) for i in range(numgates)]
  #Us = [np.eye(4) for i in range(numgates)]
  Us[0] = np.eye(4)
  #Us = [np.eye(4)]
  #Us = [np.eye(2), np.array([[ 0.329-0.408j, -0.848+0.075j], [ 0.372+0.766j, -0.180+0.493j]])]
  ops = [unitary_to_gate(U)(q0, q1) for U in Us]
  ket0 = np.array([1,0], dtype = np.complex64)
  ketp = 1/np.sqrt(2) * np.array([1,0], dtype = np.complex64)
  inp = np.kron(ket0, ket0)
  msmt_qubits = [1]
  povm_matr = .5 * np.matrix([[1,1], [1,1]], dtype=np.complex64)
  #povm_matr = np.matrix([[1,0], [0,0]], dtype=np.complex64)

  def lossfn(Y, err=False):
    #cout = np.matmul(Us[1], la.expm(X), inp)
    cout = np.matrix(np.matmul(np.matmul(la.expm(Y), Us[1]), inp))
    ip = in_prod(povm_matr, cout.getH() * cout)
    if err:
      raise
      pass
    return ip
  X = la.logm(Us[0])
  #eg = est_grad(lossfn, X, .001)

  circuit = cirq.Circuit()
  circuit.append(ops)
  losses = []
  gss = []
  for i in range(40):
    gradients, loss = loss_gradient(circuit, inp, msmt_qubits, povm_matr)
    print 'gradient:'
    print gradients[0]
    circuit = step(circuit, gradients, 1, aim=None)
    losses.append(loss)
    gss.append(gradients)

  #g2, l2 = loss_gradient(nc, inp, msmt_qubits, povm_matr)
  #print '%s -> %s  (change: %s)' % (loss, l2, l2-loss)

  #egs = [.5 * (eg - np.matrix(eg).getH()), 0*eg]
  #anc = step(circuit, egs, .1, aim=None)

  #ag2, al2 = loss_gradient(anc, inp, msmt_qubits, povm_matr)

  #X2 = la.logm(get_unitaries(nc)[0])
  #aX2 = la.logm(get_unitaries(anc)[0])
  '''
