import pytest
import numpy as np
import tensornetwork as tn
import jax.numpy as jnp
import tn_vumps.vumps as vumps
import tn_vumps.contractions as ct
import tn_vumps.params




import pytest
import numpy as np

import jax

import tensornetwork as tn
import tensornetwork.linalg.linalg
import tn_vumps.contractions as ct

from jax.config import config
config.update("jax_enable_x64", True)


backends = ["numpy"]
dtypes = [np.float32, np.complex64]# np.float64, np.complex64, np.complex128]
chis = [4] #[2, 3, 4]
ds = [2]#, 3]


def random_hermitian_system(backend, dtype, chi, d):
  """
  Return A_L, C, A_R representing a normalized quantum state and a Hermitian
  H.
  """
  A_1 = tn.randn((chi, d, chi), dtype=dtype, backend=backend, seed=10)
  A_L, _ = tn.linalg.linalg.qr(A_1, pivot_axis=2, non_negative_diagonal=True)
  C, A_R = tn.linalg.linalg.rq(A_L, pivot_axis=1, non_negative_diagonal=True)
  C /= tn.linalg.linalg.norm(C)
  H = tn.randn((d, d, d, d), dtype=dtype, backend=backend, seed=10)
  H = H.reshape((d*d, d*d))
  H = 0.5*(H + H.H)
  H = H.reshape((d, d, d, d))
  return (A_L, C, A_R, H)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_gauge_match_left(backend, dtype, chi, d):
  """
  Gauge matching is a null op on a system gauge matched on the LHS.
  """
  A_L, C, _, _ = random_hermitian_system(backend, dtype, chi, d)
  A_C = ct.rightmult(A_L, C)
  A_L2, _ = vumps.gauge_match(A_C, C, True)
  rtol = 2*A_L.size*C.size*A_C.size*np.finfo(dtype).eps
  np.testing.assert_allclose(A_L.array, A_L2.array, rtol=rtol)
  I = tn.eye(chi, backend=backend, dtype=dtype)
  XAL2 = ct.XopL(A_L2, I)
  np.testing.assert_allclose(XAL2.array, I.array, atol=rtol)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_gauge_match_right(backend, dtype, chi, d):
  """
  Gauge matching is a null op on a system gauge matched on the RHS.
  """
  _, C, A_R, _ = random_hermitian_system(backend, dtype, chi, d)
  A_C = ct.leftmult(C, A_R)
  _, A_R2 = vumps.gauge_match(A_C, C, True)
  rtol = 2*A_R.size*C.size*A_C.size*np.finfo(dtype).eps
  np.testing.assert_allclose(A_R.array, A_R2.array, rtol=rtol)
  I = tn.eye(chi, backend=backend, dtype=dtype)
  XAR2 = ct.XopR(A_R2, I)
  np.testing.assert_allclose(XAR2.array, I.array, atol=rtol)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_gauge_match_minimizes(backend, dtype, chi, d):
  """
  Gauge matching decreases the values of ||A_C - A_L C ||
  and ||A_C - C A_R||.
  """
  A_L, C, A_R, _ = random_hermitian_system(backend, dtype, chi, d)
  A_C = tn.randn((chi, d, chi), dtype=dtype, backend=backend)
  epsL = tn.linalg.linalg.norm(A_C - ct.rightmult(A_L, C))
  epsR = tn.linalg.linalg.norm(A_C - ct.leftmult(C, A_R))
  A_L2, A_R2 = vumps.gauge_match(A_C, C, True)
  epsL2 = tn.linalg.linalg.norm(A_C - ct.rightmult(A_L2, C))
  epsR2 = tn.linalg.linalg.norm(A_C - ct.leftmult(C, A_R2))
  assert epsL2 < epsL
  assert epsR2 < epsR


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_minimize_Hc(backend, dtype, chi, d):
  """
  The result of minimize_Hc solves the eigenproblem for Hc.
  """
  A_L, C, A_R, H = random_hermitian_system(backend, dtype, chi, d)
  LH = tn.randn((chi, chi), dtype=dtype, backend=backend)
  RH = tn.randn((chi, chi), dtype=dtype, backend=backend)
  LH = 0.5*(LH + LH.H)
  RH = 0.5*(RH + RH.H)
  params = tn_vumps.params.krylov_params(n_krylov=chi*chi)
  ev, newC = tn_vumps.vumps.minimize_Hc([A_L, C, A_R], [H, LH, RH], 1E-5,
                                        params)
  C_prime = ct.apply_Hc(newC, A_L, A_R, [H, LH, RH])
  EC = ev * newC
  np.testing.assert_allclose(C_prime.array, EC.array, rtol=1E-4, atol=1E-4)



@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_minimize_HAc(backend, dtype, chi, d):
  """
  The result of minimize_HAc solves the eigenproblem for HAc.
  """
  A_L, C, A_R, H = random_hermitian_system(backend, dtype, chi, d)
  A_C = ct.leftmult(C, A_R)
  LH = tn.randn((chi, chi), dtype=dtype, backend=backend)
  LH = 0.5*(LH + LH.H)
  RH = tn.randn((chi, chi), dtype=dtype, backend=backend)
  RH = 0.5*(RH + RH.H)
  params = tn_vumps.params.krylov_params(n_krylov=chi*chi, max_restarts=20)
  ev, newAC = tn_vumps.vumps.minimize_HAc([A_L, C, A_R], A_C, [H, LH, RH], 1E-5,
                                          params)
  AC_prime = ct.apply_HAc(newAC, A_L, A_R, H, LH, RH)
  EAC = ev * newAC
  np.testing.assert_allclose(AC_prime.array, EAC.array, rtol=1E-4, atol=1E-4)


def test_solve_for_RH():
  """
  The result of solve_for_RH is indeed a solution of the appropriate linear 
  problem.
  """
  return


def test_solve_for_LH():
  """
  The result of solve_for_LH is indeed a solution of the appropriate linear 
  problem.
  """
  return
