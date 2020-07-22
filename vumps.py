import copy
import os
import importlib
import functools

from typing import Any, Text, Callable, Optional, Dict, Sequence, Tuple

import tensornetwork as tn
import tensornetwork.linalg.linalg

import tn_vumps.writer
import tn_vumps.params
import tn_vumps.benchmark as benchmark
import tn_vumps.environment as environment
import tn_vumps.contractions as ct
import tn_vumps.polar

Array = Any
MATVEC_CACHE = {"HAc": dict(),
                "Hc": dict(),
                "LH": dict(),
                "RH": dict()}

##########################################################################
##########################################################################
# Functions to handle output.
def diagnostics(mpslist: Sequence[tn.Tensor], H: tn.Tensor,
                oldE: float) -> Tuple[float, float, tn.Tensor, float]:
  """
  Makes a few computations to output during a vumps run.
  """
  t0 = benchmark.tick()
  E = ct.twositeexpect(mpslist, H).array
  dE = abs(E - oldE)
  norm = ct.mpsnorm(mpslist)
  tf = benchmark.tock(t0, dat=norm)
  return E, dE, norm, tf


def ostr(string: Text) -> Text:
  """
  Truncates to two decimal places.  """
  return '{:1.2e}'.format(string)


def output(writer: tn_vumps.writer.Writer, Niter: int, delta: float, E: float,
           dE: float, norm: float, timing_data: Optional[Dict] = None) -> None:
  """
  Does the actual outputting.
  """
  outstr = "N = " + str(Niter) + "| eps = " + ostr(delta)
  outstr += "| E = " + '{0:1.16f}'.format(E)
  outstr += "| dE = " + ostr(dE)
  # outstr += "| |B2| = " + ostr(B2)
  if timing_data is not None:
    outstr += "| dt= " + ostr(timing_data["Total"])
  writer.write(outstr)

  this_output = [Niter, E, delta, norm.array]
  writer.data_write(this_output)

  if timing_data is not None:
    writer.timing_write(Niter, timing_data)


def make_writer(outdir: Optional[Text] = None) -> tn_vumps.writer.Writer:
  """
  Initialize the Writer. Creates a directory in the appropriate place, and
  an output file with headers hardcoded here as 'headers'. The Writer,
  defined in writer.py, remembers the directory and will append to this
  file as the simulation proceeds. It can also be used to pickle data,
  notably the final wavefunction.

  PARAMETERS
  ----------
  outdir (string): Path to the directory where output is to be saved. This
                   directory will be created if it does not yet exist.
                   Otherwise any contents with filename collisions during
                   the simulation will be overwritten.

  OUTPUT
  ------
  writer (writer.Writer): The Writer.
  """

  data_headers = ["N", "E", "|B|", "<psi>"]
  if outdir is None:
    return None
  timing_headers = ["N", "Total", "Diagnostics", "Iteration",
                    "Gradient", "HAc", "Hc", "Gauge Match", "Loss",
                    "Environment", "LH", "RH"]
  writer = tn_vumps.writer.Writer(outdir, data_headers=data_headers,
                                  timing_headers=timing_headers)
  return writer


###############################################################################
# Effective environment.
###############################################################################
def solve_environment(mpslist: Sequence[tn.Tensor], delta: float,
                      fpoints: Sequence[tn.Tensor],
                      H: tn.Tensor, env_params: Dict, H_env: Optional[tn.Tensor]
                      = None) -> Tuple[tn.Tensor, Dict]:
  timing = {}
  timing["Environment"] = benchmark.tick()
  if H_env is None:
    H_env = [None, None]

  lh, rh = H_env  # lowercase means 'from previous iteration'

  A_L, _, A_R = mpslist
  rL, lR = fpoints

  timing["LH"] = benchmark.tick()
  LH = solve_for_LH(A_L, H, lR, env_params, delta, oldLH=lh)
  timing["LH"] = benchmark.tock(timing["LH"], dat=LH)

  timing["RH"] = benchmark.tick()
  RH = solve_for_RH(A_R, H, rL, env_params, delta, oldRH=rh)
  timing["RH"] = benchmark.tock(timing["RH"], dat=RH)

  H_env = [LH, RH]
  timing["Environment"] = benchmark.tock(timing["Environment"], dat=RH)
  return (H_env, timing)


def LH_matvec(v: Array, lR: Array, A_L: Array, backend: Text) -> Array:
  v, lR, A_L = [tn.Tensor(a, backend=backend) for a in [v, lR, A_L]]
  Th_v = ct.XopL(A_L, v)
  vR = ct.projdiag(v, lR)
  v = v - Th_v + vR
  return v.array


def solve_for_LH(A_L: tn.Tensor, H: tn.Tensor, lR: tn.Tensor, params: Dict,
                 delta: float, oldLH: Optional[tn.Tensor] = None) -> tn.Tensor:
  """
  Find the renormalized left environment Hamiltonian using a sparse
  solver.
  """
  tol = params["tol_coef"]*delta
  hL_bare = ct.compute_hL(A_L, H)
  hL_div = ct.projdiag(hL_bare, lR)
  hL = hL_bare - hL_div
  matvec_args = [lR, A_L]
  n_krylov = min(params["n_krylov"], hL.size**2)
  backend = A_L.backend.name
  if backend not in MATVEC_CACHE["LH"]:
    mv = functools.partial(LH_matvec, backend=backend)
    MATVEC_CACHE["LH"][backend] = mv
  mv = MATVEC_CACHE["LH"][backend]
  LH, _ = tn.linalg.krylov.gmres(mv,
                                 hL,
                                 A_args=matvec_args,
                                 x0=oldLH,
                                 tol=tol,
                                 num_krylov_vectors=params["n_krylov"],
                                 maxiter=params["max_restarts"])
  return LH


def RH_matvec(v: Array, rL: Array, A_R: Array, backend: Text) -> Array:
  v, rL, A_R = [tn.Tensor(a, backend=backend) for a in [v, rL, A_R]]
  Th_v = ct.XopR(A_R, v)
  Lv = ct.projdiag(rL, v)
  v = v - Th_v + Lv
  return v.array


def solve_for_RH(A_R: tn.Tensor, H: tn.Tensor, rL: tn.Tensor, params: Dict,
                 delta: float, oldRH: Optional[tn.Tensor] = None) -> tn.Tensor:
  """
  Find the renormalized right environment Hamiltonian using a sparse
  solver.
  """
  tol = params["tol_coef"]*delta
  hR_bare = ct.compute_hR(A_R, H)
  hR_div = ct.projdiag(rL, hR_bare)
  hR = hR_bare - hR_div
  matvec_args = [rL, A_R]
  n_krylov = min(params["n_krylov"], hR.size**2)
  backend = A_R.backend.name
  if backend not in MATVEC_CACHE["RH"]:
    mv = functools.partial(RH_matvec, backend=backend)
    MATVEC_CACHE["RH"][backend] = mv
  mv = MATVEC_CACHE["RH"][backend]
  RH, _ = tn.linalg.krylov.gmres(mv,
                                 hR,
                                 A_args=matvec_args,
                                 x0=oldRH,
                                 tol=tol,
                                 num_krylov_vectors=params["n_krylov"],
                                 maxiter=params["max_restarts"])
  return RH


###############################################################################
###############################################################################
# Gradient.
###############################################################################
###############################################################################
def minimum_eigenpair(matvec: Callable, mv_args: Sequence, guess: tn.Tensor,
                      tol: float, max_restarts: int = 100,
                      n_krylov: int = 40, reorth: bool = True, n_diag: int = 10,
                      verbose: bool
                      = True) -> Tuple[tn.Tensor, tn.Tensor, float]:
  eV = guess
  arrays = [a.array for a in mv_args]
  for _ in range(max_restarts):
    out = tn.linalg.krylov.eigsh_lanczos(matvec,
                                         backend=eV.backend,
                                         args=mv_args,
                                         initial_state=eV,
                                         numeig=1,
                                         num_krylov_vecs=n_krylov,
                                         ndiag=n_diag,
                                         reorthogonalize=reorth)
    ev, eV = out
    ev = ev[0]
    eV = eV[0]
    Ax = tn.Tensor(matvec(eV.array, *arrays), backend=guess.backend)
    e_eV = ev * eV
    rho = tn.norm(tn.abs(Ax - e_eV))
    err = rho  # / jnp.linalg.norm(e_eV)
    if err <= tol:
      return (ev, eV, err)
  if verbose:
    print("Warning: eigensolve exited early with error=", err)
  return (ev, eV, err)


###############################################################################
# HAc
def HAc_matvec(A_C: Array, A_L: Array, A_R: Array, H: Array, LH: Array,
               RH: Array, backend: Text):
  arrays = [A_C, A_L, A_R, H, LH, RH]
  A_C, A_L, A_R, H, LH, RH = [tn.Tensor(a, backend=backend) for a in arrays]
  result = ct.apply_HAc(A_C, A_L, A_R, H, LH, RH)
  return result.array


def minimize_HAc(mpslist: Sequence[tn.Tensor], A_C: tn.Tensor,
                 Hlist: Sequence[tn.Tensor],
                 delta: float, params: Dict) -> Tuple[float, tn.Tensor]:
  """
  The dominant (most negative) eigenvector of HAc.
  """
  A_L, _, A_R = mpslist
  backend = A_C.backend.name
  tol = params["tol_coef"]*delta
  mv_args = [A_L, A_R, *Hlist]
  if backend not in MATVEC_CACHE["HAc"]:
    mv = functools.partial(HAc_matvec, backend=backend)
    MATVEC_CACHE["HAc"][backend] = mv
  mv = MATVEC_CACHE["HAc"][backend]
  n_krylov = min(params["n_krylov"], A_C.size**2)
  ev, newA_C, _ = minimum_eigenpair(mv, mv_args, A_C, tol,
                                    max_restarts=params["max_restarts"],
                                    n_krylov=n_krylov,
                                    reorth=params["reorth"],
                                    n_diag=params["n_diag"])

  return ev, newA_C

###############################################################################
# Hc
def Hc_matvec(C: Array, A_L: Array, A_R: Array, H: Array, LH: Array,
              RH: Array, backend: Text) -> Array:
  arrays = [C, A_L, A_R, H, LH, RH]
  C, A_L, A_R, H, LH, RH = [tn.Tensor(a, backend=backend) for a in arrays]
  result = ct.apply_Hc(C, A_L, A_R, [H, LH, RH])
  return result.array


def minimize_Hc(mpslist: Sequence[tn.Tensor], Hlist: Sequence[tn.Tensor],
                delta: float, params: Dict) -> Tuple[float, tn.Tensor]:
  A_L, C, A_R = mpslist
  tol = params["tol_coef"]*delta
  mv_args = [A_L, A_R, *Hlist]
  backend = A_L.backend.name
  if backend not in MATVEC_CACHE["Hc"]:
    mv = functools.partial(Hc_matvec, backend=backend)
    MATVEC_CACHE["Hc"][backend] = mv
  mv = MATVEC_CACHE["Hc"][backend]
  n_krylov = min(params["n_krylov"], C.size**2)
  ev, newC, _ = minimum_eigenpair(mv, mv_args, C, tol,
                                  max_restarts=params["max_restarts"],
                                  n_krylov=n_krylov,
                                  reorth=params["reorth"],
                                  n_diag=params["n_diag"])
  return ev, newC


def gauge_match(A_C: tn.Tensor, C: tn.Tensor,
                svd: bool = True) -> Tuple[tn.Tensor, tn.Tensor]:
  """
  Return approximately gauge-matched A_L and A_R from A_C and C
  using a polar decomposition.

  A_L and A_R are chosen to minimize ||A_C - A_L C|| and ||A_C - C A_R||.
  The respective solutions are the isometric factors in the
  polar decompositions of A_C C\dag and C\dag A_C.

  PARAMETERS
  ----------
  A_C (chi, d, chi)
  C (chi, chi)     : MPS tensors.
  svd (bool)      :  Toggles whether the SVD or QDWH method is used for the
                     polar decomposition. In general, this should be set
                     False on the GPU and True otherwise.

  RETURNS
  -------
  A_L, A_R (d, chi, chi): Such that A_L C A_R minimizes ||A_C - A_L C||
                          and ||A_C - C A_R||, with A_L and A_R
                          left (right) isometric.
  """
  _ = svd
  UC = tn_vumps.polar.polarU(C) # unitary
  UAc_l = tn_vumps.polar.polarU(A_C) # left isometric
  A_L = ct.rightmult(UAc_l, UC.H)

  UAc_r = tn_vumps.polar.polarU(A_C, pivot_axis=1) # right isometric
  A_R = ct.leftmult(UC.H, UAc_r)
  return (A_L, A_R)


def apply_gradient(iter_data, H, heff_krylov_params, gauge_via_svd):
  """
  Apply the MPS gradient.
  """
  timing = {}
  timing["Gradient"] = benchmark.tick()
  mpslist, a_c, _, H_env, delta = iter_data
  LH, RH = H_env
  Hlist = [H, LH, RH]
  timing["HAc"] = benchmark.tick()
  _, A_C = minimize_HAc(mpslist, a_c, Hlist, delta, heff_krylov_params)
  timing["HAc"] = benchmark.tock(timing["HAc"], dat=A_C)

  timing["Hc"] = benchmark.tick()
  _, C = minimize_Hc(mpslist, Hlist, delta, heff_krylov_params)
  timing["Hc"] = benchmark.tock(timing["Hc"], dat=C)

  timing["Gauge Match"] = benchmark.tick()
  A_L, A_R = gauge_match(A_C, C, svd=gauge_via_svd)
  timing["Gauge Match"] = benchmark.tock(timing["Gauge Match"], dat=A_L)

  timing["Loss"] = benchmark.tick()
  eL = tn.norm(A_C - ct.rightmult(A_L, C))
  eR = tn.norm(A_C - ct.leftmult(C, A_R))
  delta = max(eL, eR)
  timing["Loss"] = benchmark.tock(timing["Loss"], dat=delta)

  newmpslist = [A_L, C, A_R]
  timing["Gradient"] = benchmark.tock(timing["Gradient"], dat=C)
  return (newmpslist, A_C, delta, timing)
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# Main loop and friends.
def vumps_approximate_tm_eigs(C):
  """
  Returns the approximate transfer matrix dominant eigenvectors,
  rL ~ C^dag C, and lR ~ C Cdag = rLdag, both trace-normalized.
  """
  rL = C.H @ C
  rL /= tn.trace(rL)
  lR = rL.H
  return (rL, lR)


def vumps_initialization(d: int, chi: int, dtype=None, backend=None):
  """
  Generate a random uMPS in mixed canonical forms, along with the left
  dominant eV L of A_L and right dominant eV R of A_R.

  PARAMETERS
  ----------
  d: Physical dimension.
  chi: Bond dimension.
  dtype: Data dtype of tensors.

  RETURNS
  -------
  mpslist = [A_L, C, A_R]: Arrays. A_L and A_R have shape (d, chi, chi),
                           and are respectively left and right orthogonal.
                           C is the (chi, chi) centre of orthogonality.
  A_C (array, (d, chi, chi)) : A_L @ C. One of the equations vumps minimizes
                               is A_L @ C = C @ A_R = A_C.
  fpoints = [rL, lR] = C^dag @ C and C @ C^dag respectively. Will converge
                       to the left and right environment Hamiltonians.
                       Both are chi x chi.
  """
  A_1 = tn.randn((chi, d, chi), dtype=dtype, backend=backend)
  A_L, _ = tn.linalg.linalg.qr(A_1, pivot_axis=2, non_negative_diagonal=True)
  C, A_R = tn.linalg.linalg.rq(A_L, pivot_axis=1, non_negative_diagonal=True)
  C /= tn.linalg.linalg.norm(C)
  A_C = ct.rightmult(A_L, C)
  L0, R0 = vumps_approximate_tm_eigs(C)
  fpoints = (L0, R0)
  mpslist = [A_L, C, A_R]
  return (mpslist, A_C, fpoints)


def vumps_iteration(iter_data, H, heff_params, env_params, gauge_via_svd):
  """
  One main iteration of VUMPS.
  """
  timing = {}
  timing["Iteration"] = benchmark.tick()
  mpslist, A_C, delta, grad_time = apply_gradient(iter_data, H, heff_params,
                                                  gauge_via_svd)
  timing.update(grad_time)
  fpoints = vumps_approximate_tm_eigs(mpslist[1])
  _, _, _, H_env, _ = iter_data
  H_env, env_time = solve_environment(mpslist, delta, fpoints, H,
                                      env_params, H_env=H_env)
  iter_data = [mpslist, A_C, fpoints, H_env, delta]
  timing.update(env_time)
  timing["Iteration"] = benchmark.tock(timing["Iteration"], dat=H_env[0])
  return (iter_data, timing)


def vumps(H: tn.Tensor, chi: int, delta_0: float = 0.1,
          out_directory: Text = "./vumps",
          vumps_params: Optional[Dict] = None,
          heff_params: Optional[Dict] = None,
          env_params: Optional[Dict] = None
          ) -> Tuple[Sequence, Dict]:
  """
  Find the ground state of a uniform two-site Hamiltonian
  using Variational Uniform Matrix Product States. This is a gradient
  descent method minimizing the distance between a given MPS and the
  best approximation to the physical ground state at its bond dimension.

  This interface function initializes vumps from a random initial state.

  PARAMETERS
  ----------
  H (array, (d, d, d, d)): The Hamiltonian whose ground state is to be found.
  chi (int)              : MPS bond dimension.
  delta_0 (float)        : Initial value for the gradient norm. The
                           convergence thresholds of the various solvers at
                           the initial step are proportional to this, via
                           coefficients in the Krylov and solver param dicts.
  out_directory (str)    : Output is saved here.

  The following arguments are bundled together by initialization functions
  in examples.vumps.params.

  vumps_params (dict)    : Hyperparameters for the vumps solver. Formed
                           by 'vumps_params'.
  heff_params (dict)     : Hyperparameters for an eigensolve of certain
                           'effective Hamiltonians'. Formed by
                           'krylov_params()'.
  env_params (dict)      : Hyperparameters for a linear solve that finds
                           the effective Hamiltonians. Formed by
                           'solver_params()'.

  RETURNS
  -------
  """
  if vumps_params is None:
    vumps_params = tn_vumps.params.vumps_params()
  if heff_params is None:
    heff_params = tn_vumps.params.krylov_params()
  if env_params is None:
    env_params = tn_vumps.params.gmres_params()


  writer = make_writer(out_directory)
  d = H.shape[0]
  writer.write("vuMPS, a love story.")
  writer.write("**************************************************************")
  mpslist, A_C, fpoints = vumps_initialization(d, chi, H.dtype,
                                               backend=H.backend)
  H_env, env_init_time = solve_environment(mpslist, delta_0,
                                           fpoints, H, env_params)
  iter_data = [mpslist, A_C, fpoints, H_env, delta_0]
  writer.write("Initial solve time: " + str(env_init_time["Environment"]))
  return vumps_work(H, iter_data, vumps_params, heff_params, env_params, writer)


def vumps_work(H: tn.Tensor, iter_data: Sequence, vumps_params: Dict,
               heff_params: Dict, env_params: Dict,
               writer: tn_vumps.writer.Writer,
               Niter0: int = 1) -> Tuple[Sequence, Dict]:
  """
  Main work loop for vumps. Should be accessed via one of the interface
  functions above.

  PARAMETERS
  ----------
  H

  """
  checkpoint_every = vumps_params["checkpoint_every"]
  max_iter = vumps_params["max_iter"]

  t_total = benchmark.tick()
  # mpslist, A_C, fpoints, H_env, delta
  mpslist, _, _, _, delta = iter_data
  E = ct.twositeexpect(mpslist, H).array
  writer.write("Initial energy: " + str(E))
  writer.write("And so it begins...")
  for Niter in range(Niter0, vumps_params["max_iter"]+Niter0):
    dT = benchmark.tick()
    timing = {}
    oldE = E
    iter_data, iter_time = vumps_iteration(iter_data, H, heff_params,
                                           env_params,
                                           vumps_params["gauge_via_svd"])
    mpslist, _, _, _, delta = iter_data
    timing.update(iter_time)

    E, dE, norm, tD = diagnostics(mpslist, H, oldE)
    timing["Diagnostics"] = tD
    timing["Total"] = benchmark.tock(dT, dat=iter_data[1])
    output(writer, Niter, delta, E, dE, norm, timing)

    if delta <= vumps_params["gradient_tol"]:
      writer.write("Convergence achieved at iteration " + str(Niter))
      break

    if checkpoint_every is not None and (Niter+1) % checkpoint_every == 0:
      writer.write("Checkpointing...")
      to_pickle = [H, iter_data, vumps_params, heff_params, env_params]
      to_pickle.append(Niter)
      writer.pickle(to_pickle, Niter)

  if Niter == max_iter - 1:
    writer.write("Maximum iteration " + str(max_iter) + " reached.")
  t_total = benchmark.tock(t_total, dat=mpslist[0])
  writer.write("The main loops took " + str(t_total) + " seconds.")
  writer.write("Simulation finished. Pickling results.")
  to_pickle = [H, iter_data, vumps_params, heff_params, env_params, Niter]
  writer.pickle(to_pickle, Niter)
  return (iter_data, timing)
###############################################################################
###############################################################################
