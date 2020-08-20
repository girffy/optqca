import cirq
import qutip
import itertools
import numpy as np
import scipy
import scipy.linalg as la
import scipy.stats


from est_grad import *

def out_prod(u,v):
  return np.array(np.matmul(np.matrix(u).T, np.matrix(v).conj()))

# compute the inner product of two vectors
def in_prod(u,v):
  val = np.dot(np.array(u.flatten()), np.array(v.flatten().conj().T))
  while type(val) == np.ndarray:
    val = val[0]
  return val

def unitary_to_gate(U):
  gate = cirq.MatrixGate(U)
  return gate

def invert_circuit(circuit):
  rev = cirq.Circuit()
  for op in reversed(list(circuit.all_operations())):
    rgate = unitary_to_gate(np.matrix(op.gate._unitary_()).getH())
    rev.append(rgate(*op.qubits))
  return rev

# TODO: vectorize
def der_exp(X, M):
  """Compute <M,.> \circ exp'(X)

  Complicated helper function, which primarily computes the derivative of exp.
  Since we are composing this derivative with the functional <M,.> anyway, we
  can do it within this function. This both saves some computation (the entire
  linear map exp'(X) does not need to be computed), and also saves some
  complexity, as there is no need to explicitly represent a matrix-to-matrix
  function by means of converting between matrices and long vectors.

  Takes as input the point X in M_n at which the derivative is to be computed,
  and the linear functional to be composed with exp'(X), in the form of a matrix
  M in M_n. Computes <M,.> \circ exp'(X), which is again a linear functional on
  M_n that we represent as a matrix in M_n.

  The derivative of exp is computed using the series in the 4th equation of
  https://en.wikipedia.org/wiki/Derivative_of_the_exponential_map

  Input:
    X: The point in M_n at which to take the derivative of exp
    M: The linear functional to compose with the derivative


  Output:
    der: the matrix in M_n representing the linear functional <M,.> \circ
         exp'(X)
  """
  num_terms = 10 # TODO: how many terms are needed?
  der = 0 * X

  XH = X.getH()

  # a variable for maintaining the value of ad_{X^*}^k(e^{X^*}M) as k increases; more
  # efficient that recomputing it each time
  # TODO: can probably roll the constant factors (the -1s and factorial) into
  # this value as well
  adXkeXM = np.matmul(la.expm(XH), M)
  #print("Start der logging ------------------")
  for k in range(num_terms):
    #print(der)
    #print(adXkeXM)
    #print("")
    der += (-1)**k * adXkeXM / np.math.factorial(k+1)
    adXkeXM = XH*adXkeXM - adXkeXM*XH
  #print(der)
  #print(adXkeXM)
  #print("End der logging ------------------")

  return der

def rekron(v, n, init):
  """Reorder the kronecker product v to put init indices at the beginning

  If v = v_0 \otimes ... \otimes v_{n-1}, then rekron(v, n, [0,2]) would produce
  v_0 \otimes v_2 \otimes v_1 \otimes v_3 \otimes ... \otimes v_k. It puts all
  of the indices from init at the beginning, followed by the remaining indices.

  Input:
    v: a 1D array of length 2**n
    n: the number of 2-dimensional tensor factors (i.e. number of qubits)
    init: a list of the indices to be moved to the front

  Output:
    rkv: a modified v with the appropriate indices shifted to the front
  """
  assert(len(v) == 2**n)
  order = init + [i for i in range(n) if i not in init]

  # we kind of trick qutip into doing this for us
  # TODO: do we need to cast permuted to complex64 here?
  permuted = qutip.Qobj(v, dims=[[2]*n, [1]*n]).permute(order).full()

  return permuted

# TODO: vectorize
def partial_trace(u, v, n):
  """Traces out all but the first n qubits of uv^*

  If n is small, partial_trace is able to efficiently compute the partial trace
  of uv^* without building the entire matrix.

  Input:
    u: a vector
    v: a vector of the same length as u
    n: the number of qubits to keep

  Output:
    M: a 2^n by 2^n matrix, representing the partial trace of uv^*
  """
  assert(u.shape == v.shape)
  bs = len(u) / 2**n
  M = np.zeros([2**n, 2**n], dtype=np.complex64)
  for i in range(2**n):
    for j in range(2**n):
      M[i,j] = in_prod(u[i*bs : (i+1)*bs]  ,  v[j*bs : (j+1)*bs])

  return M

def meas_prob_grad(state, err_vec):
  ip2 = 2 * np.multiply(state, np.array(err_vec.T)[0])
  return ip2

def loss_gradient(circuit, initial_state, msmt_qubits, err_vec):
  """Compute the loss gradient of a quantum circuit

  Given a quantum circuit and a training example (a quantum input and example
  output), produces the loss gradient of the circuit with respect to this
  example. The gradient is represented as a list of gradients for each
  operation. We parameterize the unitaries U(n) by the space u(n) of
  skew-hermitian matrices via the matrix exponential, and produce a gradient
  within u(n).

  Inputs:
    circuit: a cirq.Circuit quantum circuit
    initial_state: the input quantum state to the quantum circuit
    msmt_qubits: the subset of qubits the eventual measurement is performed on
    err_vec: a binary vector on the space of msmt_qubits, indicating which
      outputs are desirable

  Output:
    gradients: the loss gradient, as a list of gradients of each operation
    loss: the current loss of the circuit
  """
  nqubits = len(circuit.all_qubits())
  if msmt_qubits != list(range(nqubits))[-len(msmt_qubits):]:
    raise Exception("msmt_qubits currently must be the last qubits")
  initial_state = initial_state.copy()
  rev = invert_circuit(circuit)
  simulator = cirq.Simulator()
  non_msmt_qubits = [i for i in range(nqubits) if i not in msmt_qubits]

  #print("Running forward pass...")
  fwd = ([initial_state] +
         [step.state() for step in
          simulator.simulate_moment_steps(circuit, initial_state=np.array(initial_state).copy())])
  final_state = fwd[-1]
  print("final_state: %s" % final_state)

  # TODO: will have to reorder this vector in the future if we allow other
  # msmt_qubits
  ext_err_vec = np.kron(np.ones(2**len(non_msmt_qubits)), err_vec).astype(np.complex64)
  ploss = (abs(in_prod(final_state, ext_err_vec)) ** 2).sum()

  # time for some weird backprop
  #print("Running backward pass...")
  err_grad = meas_prob_grad(final_state, ext_err_vec) # dloss/dphi
  emp_err_grad = None
  egnorm = np.linalg.norm(err_grad)
  # TODO: what should be my threshold here?
  if np.isclose(egnorm, 0):
    bak = [0*err_grad for i in range(len(list(circuit.all_operations())) + 1)]
  else:
    bak_init = err_grad / egnorm
    foo = simulator.simulate(rev, initial_state=bak_init.copy())
    bak = list(reversed([bak_init] +
            [step.state() for step in
             simulator.simulate_moment_steps(rev, initial_state=np.array(bak_init).copy())]))
    bak = map(lambda x: x*egnorm, bak)


  #loss = abs(fwd[-1].dot(err_fcnl))**2
  #ip = in_prod(fwd[-1], err_fcnl)
  print("end (norm %s) is:" % la.norm(fwd[-1]))
  print(fwd[-1])
  print("err_grad (norm %s) is:" % la.norm(err_grad))
  print(err_grad)
  #print("err_grad is %s" % err_grad)
  #print("emp_err_grad is %s" % emp_err_grad)
  print("ploss is %s" % ploss)

  gradients = []
  for i, op in enumerate(circuit.all_operations()):
    # TODO: remove this 1gate
    #if i != 95:
    if False:
      gradients.append(0*op.gate._unitary_())
      continue

    #print("working on gate %s" % i)
    # rekron both prev_in and next_err so the relevant qubits are at the front
    init = [q.x for q in op.qubits]
    prev_in = rekron(fwd[i], nqubits, init)
    next_err = rekron(bak[i+1], nqubits, init)


    # https://arxiv.org/pdf/1203.6151.pdf
    # TODO: there is a larger discussion of the best ways to do this in this
    # paper, but it appears that logm tends to produce skew-hermitian matrices
    # anyway
    X_init, err = la.logm(op.gate._unitary_().astype(np.complex64), disp=False)
    X = .5 * (X_init - np.matrix(X_init).getH()) # force X to be skew-hermitian just in case

    # TODO: may be able to compute partial trace more quickly using qutip
    M = partial_trace(next_err, prev_in, len(init)) # dloss/dU
    D = der_exp(X, M) # dloss/dX
    gradient = .5 * (D - D.getH()) # project Dr into u(n)

    if True:
      #print("prev_in, next_err:")
      #print(prev_in)
      #print(next_err)
      print("X is:")
      print(X)
      print("U is:")
      print(la.expm(X))
      print("M is:")
      print(M)
      print("D is:")
      print(D)
      print("gradient is:")
      print(gradient)
      #raise

    gradients.append(gradient)

  return gradients, ploss

def avg_gradient(circuit, msmt_qubits, training_set):
  """Averages the loss gradient and loss over a list of inputs/POVM matrices"""
  assert len(training_set) > 0

  grad_sum, loss_sum = loss_gradient(circuit, training_set[0][0], msmt_qubits, training_set[0][1])
  for inp, povm_matr in training_set[1:]:
    gradients, loss = loss_gradient(circuit, inp, msmt_qubits, povm_matr)
    for i, gradient in enumerate(gradients):
      grad_sum[i] += gradient
    loss_sum += loss

  scaled_gradients = map(lambda x: x/len(training_set), grad_sum)
  avg_loss = loss_sum / len(training_set)

  return scaled_gradients, avg_loss

def step(circuit, gradients, stepsize, aim=None):
  """Given a circuit and gradients for the gates, apply a step of GD

  Steps in the direction of deepest descent, with step size configurable by
  stepsize and aim.

  If aim is None, then each step is simply the gradient multiplied by stepsize.

  If aim is 'norm', then the steps are chosen so that the norm of each step is
  stepsize.

  If aim is 'decrease', then the steps are chosen so that, according to the
  first-order approximation of the loss given by the gradients, the estimated
  decrease in the loss from this step is stepsize.

  Input:
    circuit: the starting circuit from which we are stepping
    gradients: a list of the gradients for each gate in circuit
    stepsize: a real number configuring the amount to step
    aim: an optional parameter configuring how stepsize is used

  Output:
    new_circuit: a new circuit obtained by stepping
  """
  new_circuit = cirq.Circuit()
  est_increase = 0
  steps = []
  for op, gradient in zip(circuit.all_operations(), gradients):
    gnorm = np.linalg.norm(gradient)
    if np.isclose(gnorm, 0):
      step = 0*gradient
    else:
      step = -stepsize * gradient

    if aim == 'norm':
      step /= gnorm
    est_increase += in_prod(step, gradient)
    steps.append(step)

  if aim == 'decrease':
    steps = map(lambda x: x * stepsize/est_increase, steps)

  #print('steps is:')
  #print(steps)
  #raise

  for step, op, gradient in zip(steps, circuit.all_operations(), gradients):
    X, err = la.logm(op.gate._unitary_().astype(np.complex64), False)
    X += step
    U = la.expm(X)

    # out of fear of numerical error, force U to be unitary
    U, _ = la.polar(U)

    new_circuit.append(op.with_gate(unitary_to_gate(U)))
  print('est_increase: %s' % est_increase)

  return new_circuit

def get_unitaries(cir):
  return [op.gate._unitary_() for op in cir.all_operations()]

# TODO: make this output more nicely for multi qubits
def sanity_check(cir, inp, msmt_qubits, repetitions=10000):
  mcir = cir.copy()
  mcir.append(cirq.measure(*[cirq.LineQubit(i) for i in msmt_qubits]))
  simulator = cirq.Simulator()
  result = simulator.run(mcir, repetitions=repetitions)
  keys = ','.join(map(str, msmt_qubits))
  #counts = result.multi_measurement_histogram(keys=map(str,msmt_qubits))
  counts = result.multi_measurement_histogram(keys=[keys])
  freqs = {res:(1.*count/repetitions) for res,count in counts.iteritems()}
  return freqs

def vec_exp(v):
  n = int(round(np.sqrt(len(v))))
  return la.expm(v.reshape([n,n], order='F')).flatten(order='F')

def emp_dir_exp(X):
  return vecvec_grad(vec_exp, X.flatten(order='F'))

def rotket(theta):
  return np.matrix([[np.cos(theta)], [np.sin(theta)]], dtype = np.complex64)



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
  err_vec = ket0

  U1 = unitary_to_gate(scipy.stats.unitary_group.rvs(4))(*qubits[:2])
  U2 = unitary_to_gate(scipy.stats.unitary_group.rvs(4))(*qubits[1:])
  #U1 = unitary_to_gate(np.eye(4))(*qubits[:2])
  #U2 = unitary_to_gate(np.eye(4))(*qubits[1:])
  circuit = cirq.Circuit()
  circuit.append([U1, U2])

  # try to learn to answer with the negation of the first qubit
  zero_to_one = np.kron(ket0, np.kron(ket0, ket0)),  np.array(ket1.T)
  one_to_zero = np.kron(ket1, np.kron(ket0, ket0)),  np.array(ket0.T)
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
