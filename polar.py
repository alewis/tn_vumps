from typing import Text
import functools
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.linalg
import tensornetwork as tn

#############################################################################
# Polar decomposition
#############################################################################
def polarU(A: tn.Tensor, pivot_axis: int = -1, mode: Text = "svd",
           Niter: int = 4):
  A_mat = tn.pivot(A, pivot_axis=pivot_axis)
  if mode == "svd":
    W, _, Vh, _ = A.backend.svd(A_mat.array)
    Umat_arr = W@Vh
  elif mode == "QDWH":
    Umat_arr = polarU_QDWH(A_mat, Niter=Niter)
  else:
    raise ValueError(f"Mode {mode} was invalid.")
  U_mat = tn.Tensor(Umat_arr, backend=A.backend)
  U = U_mat.reshape(A.shape)
  return U


def polarU_QDWH(A, Niter=4):
  """
  Computes the isometric factor U in the polar decomposition, U = u @ vh
  where u and vh are the singular vector matrices in the SVD.

  This algorithm computes this factor using the "QDWH" iterative algorithm,
  (explained for example at https://sci-hub.tw/10.1137/120876605), which
  is based on an iterative procedure called "dynamically weighted Halley
  iterations". Each iteration is essentially performed by
  weighted_iterationQR. Eventually (usually after 2 iterations) we switch to
  the cheaper weighted_iterationChol, which is mathematically equivalent
  but only stable when the input is well-conditioned; the iterations
  improve the condition number.

  Compared to directly computing u @ vh via the SVD, this algorithm is
  considerably (~7-20 X) faster on the GPU, but perhaps 2X slower on the
  CPU or TPU.


  PARAMETERS
  ----------
  A: The matrix to be decomposed.
  Niter: The number of QDWH iterations.


  RETURNS
  -------
  U: The approximate polar factor.
  """
  m, n = A.shape
  if n > m:
    A = A.H
  if A.backend.name == "numpy":
    U = np_qdwh(A.array, Niter)
  elif A.backend.name == "jax":
    U = jax_qdwh(A.array, Niter)
  else:
    raise ValueError(f"QDWH not implemented for backend {A.backend.name}")
  if n > m:
    U = U.T.conj()
  return U


@functools.partial(jax.jit, static_argnums=(1,))
def jax_qdwh(A, Niter):
  """
  Implements the QDWH polar decomposition.
  """
  alpha = jnp.linalg.norm(A)
  lcond = 1E-6
  X = A / alpha

  def hl(l):
    d = (4*(1 - l**2)/(l**4))**(1/3)
    f = 8*(2 - l**2)/(l**2 * (1 + d)**(1/2))
    h = (1 + d)**(1/2) + 0.5 * (8 - 4*d + f)**0.5
    return h

  for _ in range(Niter):
    a = hl(lcond)
    b = (a - 1)**2 / 4
    c = a + b - 1
    # if c < 100:
    #   X = weighted_iterationChol(X, a, b, c)
    # else:
    #   X = weighted_iterationQR(X, a, b, c)
    X = jax.lax.cond(c < 100,
                     jax_iterationChol,
                     jax_iterationQR,
                     (X, a, b, c))
    lcond *= (a + b * lcond**2)/(1 + c * lcond**2)
  return X


@jax.jit
def jax_iterationChol(args):
  """
  One iteration of the QDWH polar decomposition, using the cheaper but
  less stable Cholesky factorization method.
  """
  X, a, b, c = args
  _, n = X.shape
  eye = jnp.eye(n)
  Z = eye + c * X.T.conj() @ X
  W = jax.scipy.linalg.cholesky(Z)
  Winv = jax.scipy.linalg.solve_triangular(W, eye, overwrite_b=True)
  XWinv = X@Winv
  X *= (b / c)
  X += (a - (b/c))*(XWinv)@(Winv.T.conj())
  return X


@jax.jit
def jax_iterationQR(args):
  """
  One iteration of the QDWH polar decomposition, using the more expensive
  but more stable QR factorization method.
  """
  X, a, b, c = args
  m, n = X.shape
  eye = jnp.eye(n)
  cX = jnp.sqrt(c)*X
  XI = jnp.vstack((cX, eye))
  Q, _ = jnp.linalg.qr(XI)
  Q1 = Q[:m, :]
  Q2 = Q[m:, :]
  X *= (b/c)
  X += (1/jnp.sqrt(c)) * (a - b/c) * Q1 @ Q2.T.conj()
  return X


def np_qdwh(A, Niter):
  """
  Implements the QDWH polar decomposition.
  """
  alpha = np.linalg.norm(A)
  lcond = 1E-6
  X = A / alpha

  def hl(l):
    d = (4*(1 - l**2)/(l**4))**(1/3)
    f = 8*(2 - l**2)/(l**2 * (1 + d)**(1/2))
    h = (1 + d)**(1/2) + 0.5 * (8 - 4*d + f)**0.5
    return h

  for _ in range(Niter):
    a = hl(lcond)
    b = (a - 1)**2 / 4
    c = a + b - 1
    if c < 100:
      X = np_iterationChol(X, a, b, c)
    else:
      X = np_iterationQR(X, a, b, c)
    lcond *= (a + b * lcond**2)/(1 + c * lcond**2)
  return X


def np_iterationChol(X, a, b, c):
  """
  One iteration of the QDWH polar decomposition, using the cheaper but
  less stable Cholesky factorization method.
  """
  _, n = X.shape
  eye = np.eye(n, dtype=X.dtype)
  Z = eye + c * X.T.conj() @ X
  W = scipy.linalg.cholesky(Z)
  Winv = scipy.linalg.solve_triangular(W, eye, overwrite_b=True)
  XWinv = X@Winv
  X *= (b / c)
  X += (a - (b/c))*(XWinv)@(Winv.T.conj())
  return X


def np_iterationQR(X, a, b, c):
  """
  One iteration of the QDWH polar decomposition, using the more expensive
  but more stable QR factorization method.
  """
  m, n = X.shape
  eye = np.eye(n)
  cX = np.sqrt(c)*X
  XI = np.vstack((cX, eye))
  Q, _ = np.linalg.qr(XI)
  Q1 = Q[:m, :]
  Q2 = Q[m:, :]
  X *= (b/c)
  X += (1/np.sqrt(c)) * (a - b/c) * Q1 @ Q2.T.conj()
  return X
