"""
Low level tensor network manipulations.

Conventions
      2                 3 4
      |                 | |
      O                  U
      |                 | |
      1                 1 2

  1---A---3            2
      |                |
      2             1--A--3
"""
from typing import Tuple
import tensornetwork as tn
ThreeTensors = Tuple[tn.Tensor, tn.Tensor, tn.Tensor]


def leftmult(lam: tn.Tensor, gam: tn.Tensor) -> tn.Tensor:
  """
  1--lam--gam--3
          |
          2
          |
  where lam is stored 1--lam--2
  """
  return tn.linalg.operations.ncon([lam, gam], [[-1, 1],
                                                [1, -2, -3]])


def rightmult(gam: tn.Tensor, lam: tn.Tensor) -> tn.Tensor:
  """
  1--gam--lam--3
     |
     2
     |
  """
  return tn.linalg.operations.ncon([gam, lam], [[-1, -2, 1],
                                                [1, -3]])


def gauge_transform(gl: tn. Tensor, A: tn.Tensor, gr: tn.Tensor) -> tn.Tensor:
  """
          |
          2
   1--gl--A--gr--3
  """
  glA = leftmult(gl, A)
  return rightmult(glA, gr)


###############################################################################
# Chain contractors - MPS.
###############################################################################
def projdiag(A: tn.Tensor, B: tn.Tensor) -> tn.Tensor:
  """
  2   2
  |---|
  |   |
  A   B  otimes I(chi, chi)
  |   |
  |---|
  1   1
  Contract A with B to find <A|B>, and put the result on the main diagonal.
  """
  val = tn.linalg.operations.ncon([A, B], [[1, 2], [1, 2]])
  one = tn.ones(A.shape[0], dtype=A.dtype, backend=A.backend)
  return tn.diagflat(val*one)


# ***************************************************************************
# TWO SITE OPERATORS
# ***************************************************************************
def rholoc(A1: tn.Tensor, A2: tn.Tensor) -> tn.Tensor:
  """
  -----A1-----A2-----
  |    |(3)   |(4)   |
  |                  |
  |                  |
  |    |(1)   |(2)   |
  -----A1*----A2*-----
  returned as a (1:2)x(3:4) matrix.
  Assuming the appropriate Schmidt vectors have been contracted into the As,
  np.trace(np.dot(op, rholoc.T)) is the expectation value of the two-site
  operator op coupling A1 to A2.
  """
  B1 = A1.conj()
  B2 = A2.conj()
  d = A1.shape[1]
  to_contract = [A1, A2, B1, B2]
  idxs = [(1, -3, 2),
          (2, -4, 3),
          (1, -1, 4),
          (4, -2, 3)]
  return tn.linalg.operations.ncon(to_contract, idxs).reshape((d**2, d**2))


##############################################################################
# VUMPS environment
##############################################################################
def XopL(A: tn.Tensor, X: tn.Tensor) -> tn.Tensor:
  """
    |---A---2
    |   |
    X   |
    |   |
    |---A*--1
  """
  XA = leftmult(X, A)
  idx = [(1, 2, -2),
         (1, 2, -1)]
  return tn.linalg.operations.ncon([XA, A.conj()], idx)


def XopR(A: tn.Tensor, X: tn.Tensor) -> tn.Tensor:
  """
    2---A---|
        |   |
        |   X
        |   |
    1---A*--|
  """
  B = rightmult(A.conj(), X)
  idx = [(-2, 2, 1),
         (-1, 2, 1)]
  return tn.linalg.operations.ncon([A, B], idx)


def compute_hL(A_L: tn.Tensor, htilde: tn.Tensor) -> tn.Tensor:
  """
  --A_L--A_L--
  |  |____|
  |  | h  |
  |  |    |
  |-A_L*-A_L*-
  """
  A_L_d = A_L.conj()
  to_contract = [A_L, A_L, A_L_d, A_L_d, htilde]
  idxs = [(4, 2, 1),
          (1, 3, -2),
          (4, 5, 7),
          (7, 6, -1),
          (5, 6, 2, 3)]
  h_L = tn.linalg.operations.ncon(to_contract, idxs)
  return h_L


def compute_hR(A_R: tn.Tensor, htilde: tn.Tensor) -> tn.Tensor:
  """
   --A_R--A_R--
      |____|  |
      | h  |  |
      |    |  |
   --A_R*-A_R*-
  """
  A_R_d = A_R.conj()
  to_contract = [A_R, A_R, A_R_d, A_R_d, htilde]
  idxs = [(-2, 2, 1),
          (1, 3, 4),
          (-1, 5, 7),
          (7, 6, 4),
          (5, 6, 2, 3)]
  h_R = tn.linalg.operations.ncon(to_contract, idxs)
  return h_R


##############################################################################
# VUMPS heff
##############################################################################
def apply_HAc(A_C: tn.Tensor, A_L: tn.Tensor, A_R: tn.Tensor,
              H: tn.Tensor, LH: tn.Tensor, RH: tn.Tensor) -> tn.Tensor:
  """
  Apply the effective Hamiltonian for A_C.
  Args:
    A_C: Current A_C.
    A_L, A_R: From the MPS.
    H: Two-site local Hamiltonian.
    LH, RH: Environment Hamiltonians.
  Returns:
    A_C_prime: Action of the effective Hamiltonian.
  """
  to_contract_1 = [A_L, A_L.conj(), A_C, H]
  idxs_1 = [(1, 2, 4),
            (1, 3, -1),
            (4, 5, -3),
            (3, -2, 2, 5)]
  term1 = tn.linalg.operations.ncon(to_contract_1, idxs_1)

  to_contract_2 = [A_C, A_R, A_R.conj(), H]
  idxs_2 = [(-1, 5, 4),
            (4, 2, 1),
            (-3, 3, 1),
            (-2, 3, 5, 2)]
  term2 = tn.linalg.operations.ncon(to_contract_2, idxs_2)

  term3 = leftmult(LH, A_C)
  term4 = rightmult(A_C, RH.T)
  A_C_prime = term1 + term2 + term3 + term4
  return A_C_prime


def apply_Hc(C: tn.Tensor, A_L: tn.Tensor, A_R: tn.Tensor, H: tn.Tensor,
             LH: tn.Tensor, RH: tn.Tensor) -> tn.Tensor:
  """
  Apply the effective Hamiltonian for C.
  Args:
    C: Current C.
    A_L, A_R: From the MPS.
    H: Two-site local Hamiltonian.
    LH, RH: Environment Hamiltonians.
  Returns:
    C_prime: Action of the effective Hamiltonian.
  """
  A_Lstar = A_L.conj()
  A_C = rightmult(A_L, C)
  to_contract = [A_C, A_Lstar, A_R, A_R.conj(), H]
  idxs = [(4, 2, 1),
          (4, 5, -1),
          (1, 3, 6),
          (-2, 7, 6),
          (5, 7, 2, 3)]
  term1 = tn.linalg.operations.ncon(to_contract, idxs)
  term2 = LH @ C
  term3 = C @ RH.T
  C_prime = term1 + term2 + term3
  return C_prime


def twositeexpect(mps: ThreeTensors, H: tn.Tensor) -> tn.Tensor:
  """
  The expectation value of the operator H in the state represented
  by A_L, C, A_R in mps.
  Args:
    mps: The MPS.
    H: The two-site operator.

  Returns:
    out: The expectation value.
  """
  A_L, C, A_R = mps
  A_CR = leftmult(C, A_R)
  d = H.shape[0]
  rho = rholoc(A_L, A_CR).reshape((d, d, d, d))
  idxs = [(1, 2, 3, 4), (1, 2, 3, 4)]
  expect = tn.linalg.operations.ncon([rho, H], idxs)
  expect = tn.abs(expect)
  return expect


def mpsnorm(mps: ThreeTensors) -> tn.Tensor:
  """
  Norm of the MPS.
  Args:
    mps: The MPS.
  Returns:
    norm: Its norm.
  """
  A_L, C, A_R = mps
  A_CR = leftmult(C, A_R)
  rho = rholoc(A_L, A_CR)
  the_norm = tn.abs(tn.trace(rho))
  return the_norm
