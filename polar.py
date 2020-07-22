import tensornetwork as tn
#############################################################################
# Polar decomposition
#############################################################################
def polarU(A: tn.Tensor, pivot_axis: int = -1):
  A_mat = tn.pivot(A, pivot_axis=pivot_axis)
  W, _, Vh, _ = A.backend.svd(A_mat.array)
  Umat_arr = W@Vh
  U_mat = tn.Tensor(Umat_arr, backend=A.backend)
  U = U_mat.reshape(A.shape)
  return U




#  @jax.jit
#  def polarjit(A, svd):
#      U = jax.lax.cond(svd, A, polarU_SVD,
#                            A, polarU_QDWH)
#      return U

#  @jax.jit
#  def polarU_SVD(A):

#      """
#      Compute the unitary part of the polar decomposition explitly as
#      U = u @ vH where A = u @ S @ vh is the SVD of A. This is about twice
#      as fast as polarU_QDWH on the
#      CPU or TPU but around an order of magnitude slower on the GPU.
#      """
#      a = jnp.asarray(A)
#      w, _, vh = jnp.linalg.svd(A, full_matrices=False)
#      u = w @ vh
#      return u


#  def polarU_QDWH(A, Niter=4):
#      """
#      Computes the isometric factor U in the polar decomposition, U = u @ vh
#      where u and vh are the singular vector matrices in the SVD.

#      This algorithm computes this factor using the "QDWH" iterative algorithm,
#      (explained for example at https://sci-hub.tw/10.1137/120876605), which
#      is based on an iterative procedure called "dynamically weighted Halley
#      iterations". Each iteration is essentially performed by
#      weighted_iterationQR. Eventually (usually after 2 iterations) we switch to
#      the cheaper weighted_iterationChol, which is mathematically equivalent
#      but only stable when the input is well-conditioned; the iterations
#      improve the condition number.

#      Compared to directly computing u @ vh via the SVD, this algorithm is
#      considerably (~7-20 X) faster on the GPU, but perhaps 2X slower on the
#      CPU or TPU.


#      PARAMETERS
#      ----------
#      A: The matrix to be decomposed.
#      Niter: The number of QDWH iterations.


#      RETURNS
#      -------
#      U: The approximate polar factor.
#      """
#      m, n = A.shape
#      if n > m:
#          U = polar_qdwh_transpose(A, Niter)
#      else:
#          U = polar_qdwh(A, Niter)
#      return U


#  @partial(jax.jit, static_argnums=(1,))
#  def polar_qdwh_transpose(A, Niter):
#      """
#      Handles the polar decomposition when n > m by transposing the input and
#      then the output.
#      """
#      A = A.T.conj()
#      Ud = polar_qdwh(A, Niter)
#      U = Ud.T.conj()
#      return U


#  @partial(jax.jit, static_argnums=(1,))
#  def polar_qdwh(A, Niter):
#      """
#      Implements the QDWH polar decomposition.
#      """
#      m, n = A.shape
#      alpha = jnp.linalg.norm(A)
#      lcond = 1E-6
#      k = 0
#      X = A / alpha
#      for k in range(Niter):
#          a = hl(lcond)
#          b = (a - 1)**2 / 4
#          c = a + b - 1
#          X = jax.lax.cond(c < 100, (X, a, b, c), weighted_iterationChol,
#                                    (X, a, b, c), weighted_iterationQR)
#          # if c < 100:
#          #   X = weighted_iterationChol(X, a, b, c)
#          # else:
#          #   X = weighted_iterationQR(X, a, b, c)
#          lcond *= (a + b * lcond**2)/(1 + c * lcond**2)
#      return X


#  @jax.jit
#  def hl(l):
#      d = (4*(1 - l**2)/(l**4))**(1/3)
#      f = 8*(2 - l**2)/(l**2 * (1 + d)**(1/2))
#      h = (1 + d)**(1/2) + 0.5 * (8 - 4*d + f)**0.5
#      return h


#  @jax.jit
#  def weighted_iterationChol(args):
#      """
#      One iteration of the QDWH polar decomposition, using the cheaper but
#      less stable Cholesky factorization method.
#      """
#      X, a, b, c = args
#      m, n = X.shape
#      eye = jnp.eye(n)
#      Z = eye + c * X.T.conj() @ X
#      W = jax.scipy.linalg.cholesky(Z)
#      Winv = jax.scipy.linalg.solve_triangular(W, eye, overwrite_b=True)
#      XWinv = X@Winv
#      X *= (b / c)
#      X += (a - (b/c))*(XWinv)@(Winv.T.conj())
#      return X


#  @jax.jit
#  def weighted_iterationQR(args):
#      """
#      One iteration of the QDWH polar decomposition, using the more expensive
#      but more stable QR factorization method.
#      """
#      X, a, b, c = args
#      m, n = X.shape
#      eye = jnp.eye(n)
#      cX = jnp.sqrt(c)*X
#      XI = jnp.vstack((cX, eye))
#      Q, R = jnp.linalg.qr(XI)
#      Q1 = Q[:m, :]
#      Q2 = Q[m:, :]
#      X *= (b/c)
#      X += (1/jnp.sqrt(c)) * (a - b/c) * Q1 @ Q2.T.conj()
#      return X
