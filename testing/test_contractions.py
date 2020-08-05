import pytest
import numpy as np

import jax

import tensornetwork as tn
import tensornetwork.linalg.linalg
import tn_vumps.contractions as ct
import tn_vumps.polar as polar

from jax.config import config
config.update("jax_enable_x64", True)


backends = ["jax", "numpy"]
dtypes = [np.float32, np.complex64]# np.float64, np.complex64, np.complex128]
chis = [4] #[2, 3, 4]
ds = [2]#, 3]


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_leftmult(backend, dtype, chi, d):
  lam = tn.randn((chi, chi), dtype=dtype, backend=backend, seed=10)
  gam = tn.randn((chi, d, chi), dtype=dtype, backend=backend, seed=10)
  result = ct.leftmult(lam, gam)
  compare = tn.linalg.operations.ncon([lam, gam], [[-1, 1],
                                                   [1, -2, -3]])
  np.testing.assert_allclose(result.array, compare.array)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_rightmult(backend, dtype, chi, d):
  lam = tn.randn((chi, chi), dtype=dtype, backend=backend, seed=10)
  gam = tn.randn((chi, d, chi), dtype=dtype, backend=backend, seed=10)
  result = ct.rightmult(gam, lam)
  compare = tn.linalg.operations.ncon([gam, lam], [[-1, -2, 1],
                                                   [1, -3]])
  np.testing.assert_allclose(result.array, compare.array)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_gauge_transform(backend, dtype, chi, d):
  lamL = tn.randn((chi, chi), dtype=dtype, backend=backend, seed=10)
  lamR = tn.randn((chi, chi), dtype=dtype, backend=backend, seed=10)
  gam = tn.randn((chi, d, chi), dtype=dtype, backend=backend, seed=10)
  result = ct.gauge_transform(lamL, gam, lamR)
  compare = tn.linalg.operations.ncon([lamL, gam, lamR],
                                      [[-1, 1], [1, -2, 2], [2, -3]])
  rtol = (lamL.size + lamR.size + gam.size) * np.finfo(dtype).eps
  np.testing.assert_allclose(result.array, compare.array, rtol=rtol)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
def test_projdiag(backend, dtype, chi):
  A = tn.randn((chi, chi), dtype=dtype, backend=backend, seed=10)
  B = tn.randn((chi, chi), dtype=dtype, backend=backend, seed=10)
  result = ct.projdiag(A, B)
  compare_val = tn.linalg.operations.ncon([A, B], [[1, 2], [1, 2]])
  compare = compare_val * tn.eye(chi, dtype=dtype, backend=backend)
  rtol = (A.size + B.size) * np.finfo(dtype).eps
  np.testing.assert_allclose(result.array, compare.array, rtol=rtol)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_rho_loc(backend, dtype, chi, d):
  A = tn.randn((chi, d, chi), dtype=dtype, backend=backend, seed=10)
  B = tn.randn((chi, d, chi), dtype=dtype, backend=backend, seed=10)
  result = ct.rholoc(A, B)
  As = A.conj()
  Bs = B.conj()
  to_contract = [A, B, As, Bs]
  idxs = [(1, -3, 2),
          (2, -4, 3),
          (1, -1, 4),
          (4, -2, 3)]
  compare = tn.linalg.operations.ncon(to_contract, idxs).reshape((d**2, d**2))
  rtol = 2 * (A.size + B.size) * np.finfo(dtype).eps
  np.testing.assert_allclose(result.array, compare.array, rtol=rtol)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_XopL_noX(backend, dtype, chi, d):
  A = tn.randn((chi, d, chi), dtype=dtype, backend=backend, seed=10)
  X = tn.eye(chi, dtype=dtype, backend=backend)
  result = ct.XopL(A, X)
  compare = tn.linalg.operations.ncon([A, A.conj()],
                                      [[1, 3, -2],
                                       [1, 3, -1]])
  rtol = (2*A.size + X.size) * np.finfo(dtype).eps
  np.testing.assert_allclose(result.array, compare.array, rtol=rtol)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_XopL(backend, dtype, chi, d):
  A = tn.randn((chi, d, chi), dtype=dtype, backend=backend, seed=10)
  X = tn.randn((chi, chi), dtype=dtype, backend=backend, seed=10)
  result = ct.XopL(A, X)
  compare = tn.linalg.operations.ncon([A, A.conj(), X],
                                      [[1, 3, -2],
                                       [2, 3, -1],
                                       [2, 1]])
  rtol = (2*A.size + X.size) * np.finfo(dtype).eps
  np.testing.assert_allclose(result.array, compare.array, rtol=rtol)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_XopR(backend, dtype, chi, d):
  A = tn.randn((chi, d, chi), dtype=dtype, backend=backend, seed=10)
  X = tn.randn((chi, chi), dtype=dtype, backend=backend, seed=10)
  result = ct.XopR(A, X)
  compare = tn.linalg.operations.ncon([A, A.conj(), X],
                                      [[-2, 3, 1],
                                       [-1, 3, 2],
                                       [2, 1]])
  rtol = (2*A.size + X.size) * np.finfo(dtype).eps
  np.testing.assert_allclose(result.array, compare.array, rtol=rtol)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_XopR_noX(backend, dtype, chi, d):
  A = tn.randn((chi, d, chi), dtype=dtype, backend=backend, seed=10)
  X = tn.eye(chi, dtype=dtype, backend=backend)
  result = ct.XopR(A, X)
  compare = tn.linalg.operations.ncon([A, A.conj()],
                                      [[-2, 2, 1],
                                       [-1, 2, 1]])
  rtol = (2*A.size + X.size) * np.finfo(dtype).eps
  np.testing.assert_allclose(result.array, compare.array, rtol=rtol)


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
def test_mpsnorm(backend, dtype, chi, d):
  A_L, C, A_R, _ = random_hermitian_system(backend, dtype, chi, d)
  the_norm = ct.mpsnorm([A_L, C, A_R])
  rtol = (A_L.size + C.size + A_R.size) * np.finfo(dtype).eps
  np.testing.assert_allclose(the_norm.array, 1., rtol=rtol)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_twositeexpect(backend, dtype, chi, d):
  A_L, C, A_R, H = random_hermitian_system(backend, dtype, chi, d)
  result = ct.twositeexpect([A_L, C, A_R], H)
  to_contract = [A_L, A_L.conj(), C, C.conj(), A_R, A_R.conj(), H]
  idxs = [[9, 3, 5],
          [9, 1, 7],
          [5, 6],
          [7, 8],
          [6, 4, 10],
          [8, 2, 10],
          [1, 2, 3, 4]]
  compare = tn.abs(tn.linalg.operations.ncon(to_contract, idxs))
  rtol = (4*A_L.size + 2*C.size + H.size) * np.finfo(dtype).eps
  np.testing.assert_allclose(result.array, compare.array, rtol=rtol)



@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_compute_hL_vs_expect(backend, dtype, chi, d):
  A_L, _, _, H = random_hermitian_system(backend, dtype, chi, d)
  I = tn.eye(chi, backend=backend, dtype=dtype)
  hL = ct.compute_hL(A_L, H)
  result = tn.abs(tn.trace(hL))
  compare = ct.twositeexpect([A_L, I, A_L], H)
  rtol = (2*A_L.size + I.size + H.size) * np.finfo(dtype).eps
  np.testing.assert_allclose(result.array, compare.array, rtol=rtol)



@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_compute_hR_vs_expect(backend, dtype, chi, d):
  _, _, A_R, H = random_hermitian_system(backend, dtype, chi, d)
  I = tn.eye(chi, backend=backend, dtype=dtype)
  hR = ct.compute_hR(A_R, H)
  result = tn.abs(tn.trace(hR))
  compare = ct.twositeexpect([A_R, I, A_R], H)
  rtol = (2*A_R.size + I.size + H.size) * np.finfo(dtype).eps
  np.testing.assert_allclose(result.array, compare.array, rtol=rtol)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
def test_polar(backend, dtype, chi):
  A = tn.randn((chi, chi), backend=backend, dtype=dtype, seed=10)
  U = polar.polarU(A)
  UUdag = U @ U.H
  UdagU = U.H @ U
  eye = tn.eye(chi, dtype, backend=backend)
  atol = A.size * np.finfo(dtype).eps
  np.testing.assert_allclose(eye.array, UUdag.array, atol=atol)
  np.testing.assert_allclose(UUdag.array, eye.array, atol=atol)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("chi", chis)
@pytest.mark.parametrize("d", ds)
def test_polar_mps(backend, dtype, chi, d):
  A = tn.randn((chi, d, chi), backend=backend, dtype=dtype, seed=10)
  UR = polar.polarU(A, pivot_axis=1)
  UL = polar.polarU(A, pivot_axis=2)
  eye = tn.eye(chi, dtype, backend=backend)
  testR = ct.XopR(UR, eye)
  testL = ct.XopL(UL, eye)
  atol = A.size * np.finfo(dtype).eps
  np.testing.assert_allclose(eye.array, testR.array, atol=atol)
  np.testing.assert_allclose(eye.array, testL.array, atol=atol)
