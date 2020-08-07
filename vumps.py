import copy
import os
import importlib
import functools
import time

from typing import Any, Text, Callable, Optional, Dict, Sequence, Tuple

import tensornetwork as tn
import tensornetwork.linalg.linalg
import tensornetwork.linalg.krylov

import tn_vumps.writer
import tn_vumps.params
import tn_vumps.benchmark as benchmark
import tn_vumps.contractions as ct
import tn_vumps.polar

Array = Any
BENCHMARKER = benchmark.Benchmarker()
timed = functools.partial(benchmark.timed, benchmarker=BENCHMARKER)

##########################################################################
##########################################################################
# Functions to handle output.
@timed
def diagnostics(mpslist: Sequence[tn.Tensor], H: tn.Tensor,
                oldE: float) -> Tuple[float, float, tn.Tensor, float]:
  """
  Makes a few computations to output during a vumps run.
  """
  E = ct.twositeexpect(mpslist, H).array
  dE = abs(E - oldE)
  norm = ct.mpsnorm(mpslist)
  return E, dE, norm


def ostr(string: Text) -> Text:
  """
  Truncates to two decimal places.  """
  return '{:1.2e}'.format(string)


def output(writer: tn_vumps.writer.Writer, Niter: int, delta: float, E: float,
           dE: float, norm: float) -> None:
  """
  Does the actual outputting.
  """
  outstr = "N = " + str(Niter) + "| eps = " + ostr(delta)
  outstr += "| E = " + '{0:1.16f}'.format(E)
  outstr += "| dE = " + ostr(dE)
  outstr += "| dt= " + ostr(BENCHMARKER.benchmarks["vumps_iteration"][-1])
  writer.write(outstr)
  this_output = [Niter, E, delta, norm.array]
  writer.data_write(this_output)


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
  writer = tn_vumps.writer.Writer(outdir, data_headers=data_headers)
  return writer


###############################################################################
# Effective environment.
###############################################################################
@timed
def solve_environment(mpslist: Sequence[tn.Tensor], delta: float,
                      fpoints: Sequence[tn.Tensor],
                      H: tn.Tensor, env_params: Dict, H_env: Optional[tn.Tensor]
                      = None) -> tn.Tensor:
  if H_env is None:
    H_env = [None, None]
  lh, rh = H_env  # lowercase means 'from previous iteration'
  A_L, _, A_R = mpslist
  rL, lR = fpoints
  LH = solve_for_LH(A_L, H, lR, env_params, delta, oldLH=lh)
  RH = solve_for_RH(A_R, H, rL, env_params, delta, oldRH=rh)
  H_env = [LH, RH]
  return H_env


def LH_matvec(v: tn.Tensor, lR: tn.Tensor, A_L: tn.Tensor) -> tn.Tensor:
  Th_v = ct.XopL(A_L, v)
  vR = ct.projdiag(v, lR)
  return v - Th_v + vR


@timed
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
  LH, _ = tn.linalg.krylov.gmres(LH_matvec,#mv,
                                 hL,
                                 A_args=matvec_args,
                                 x0=oldLH,
                                 tol=tol,
                                 num_krylov_vectors=n_krylov,
                                 maxiter=params["max_restarts"])
  benchmark.block_until_ready(LH)
  return LH


def RH_matvec(v: tn.Tensor, rL: tn.Tensor, A_R: tn.Tensor) -> tn.Tensor:
  Th_v = ct.XopR(A_R, v)
  Lv = ct.projdiag(rL, v)
  return v - Th_v + Lv


@timed
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
  RH, _ = tn.linalg.krylov.gmres(RH_matvec,
                                 hR,
                                 A_args=matvec_args,
                                 x0=oldRH,
                                 tol=tol,
                                 num_krylov_vectors=n_krylov,
                                 maxiter=params["max_restarts"])
  benchmark.block_until_ready(RH)
  return RH


###############################################################################
###############################################################################
# Gradient.
###############################################################################
###############################################################################
@timed
def minimum_eigenpair(matvec: Callable, mv_args: Sequence, guess: tn.Tensor,
                      tol: float, max_restarts: int = 100,
                      n_krylov: int = 40, reorth: bool = True, n_diag: int = 10,
                      verbose: bool
                      = True) -> Tuple[tn.Tensor, tn.Tensor, float]:
  eV = guess
  for _ in range(max_restarts):
    out = tn.linalg.krylov.eigsh_lanczos(matvec,
                                         backend=eV.backend,
                                         args=mv_args,
                                         x0=eV,
                                         numeig=1,
                                         num_krylov_vecs=n_krylov,
                                         ndiag=n_diag,
                                         reorthogonalize=reorth)
    ev, eV = out
    ev = ev[0]
    eV = eV[0]
    Ax = matvec(eV, *mv_args)
    e_eV = ev * eV
    rho = tn.norm(tn.abs(Ax - e_eV))
    err = rho  # / jnp.linalg.norm(e_eV)
    if err <= tol:
      return (ev, eV, err)
  if verbose:
    print("Warning: eigensolve exited early with error=", err)
  benchmark.block_until_ready(eV)
  return (ev, eV, err)


###############################################################################
# HAc
@timed
def minimize_HAc(mpslist: Sequence[tn.Tensor], A_C: tn.Tensor,
                 Hlist: Sequence[tn.Tensor],
                 delta: float, params: Dict) -> Tuple[float, tn.Tensor]:
  """
  The dominant (most negative) eigenvector of HAc.
  """
  A_L, _, A_R = mpslist
  tol = params["tol_coef"]*delta
  mv_args = [A_L, A_R, *Hlist]
  n_krylov = min(params["n_krylov"], A_C.size**2)
  ev, newA_C, _ = minimum_eigenpair(ct.apply_HAc, mv_args, A_C, tol,
                                    max_restarts=params["max_restarts"],
                                    n_krylov=n_krylov,
                                    reorth=params["reorth"],
                                    n_diag=params["n_diag"])
  return ev, newA_C

###############################################################################
# Hc
@timed
def minimize_Hc(mpslist: Sequence[tn.Tensor], Hlist: Sequence[tn.Tensor],
                delta: float, params: Dict) -> Tuple[float, tn.Tensor]:
  A_L, C, A_R = mpslist
  tol = params["tol_coef"]*delta
  mv_args = [A_L, A_R, *Hlist]
  n_krylov = min(params["n_krylov"], C.size**2)
  ev, newC, _ = minimum_eigenpair(ct.apply_Hc, mv_args, C, tol,
                                  max_restarts=params["max_restarts"],
                                  n_krylov=n_krylov,
                                  reorth=params["reorth"],
                                  n_diag=params["n_diag"])
  return ev, newC


@timed
def gauge_match(A_C: tn.Tensor, C: tn.Tensor,
                mode: Text = "svd") -> Tuple[tn.Tensor, tn.Tensor]:
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
  UC = tn_vumps.polar.polarU(C, mode=mode) # unitary
  UAc_l = tn_vumps.polar.polarU(A_C, mode=mode) # left isometric
  A_L = ct.rightmult(UAc_l, UC.H)

  UAc_r = tn_vumps.polar.polarU(A_C, pivot_axis=1, mode=mode) # right isometric
  A_R = ct.leftmult(UC.H, UAc_r)
  return (A_L, A_R)


@timed
def apply_gradient(iter_data, H, heff_krylov_params, gauge_match_mode):
  """
  Apply the MPS gradient.
  """
  mpslist, a_c, _, H_env, delta = iter_data
  LH, RH = H_env
  Hlist = [H, LH, RH]
  _, A_C = minimize_HAc(mpslist, a_c, Hlist, delta, heff_krylov_params)
  _, C = minimize_Hc(mpslist, Hlist, delta, heff_krylov_params)
  A_L, A_R = gauge_match(A_C, C, mode=gauge_match_mode)
  eL = tn.norm(A_C - ct.rightmult(A_L, C))
  eR = tn.norm(A_C - ct.leftmult(C, A_R))
  delta = max(eL, eR)
  newmpslist = [A_L, C, A_R]
  return (newmpslist, A_C, delta)
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# Main loop and friends.
@timed
def vumps_approximate_tm_eigs(C):
  """
  Returns the approximate transfer matrix dominant eigenvectors,
  rL ~ C^dag C, and lR ~ C Cdag = rLdag, both trace-normalized.
  """
  rL = C.H @ C
  rL /= tn.trace(rL)
  lR = rL.H
  return (rL, lR)


@timed
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
  benchmark.block_until_ready(R0)
  return (mpslist, A_C, fpoints)


@timed
def vumps_iteration(iter_data, H, heff_params, env_params, gauge_match_mode):
  """
  One main iteration of VUMPS.
  """
  mpslist, A_C, delta = apply_gradient(iter_data, H, heff_params,
                                       gauge_match_mode)
  fpoints = vumps_approximate_tm_eigs(mpslist[1])
  _, _, _, H_env, _ = iter_data
  H_env = solve_environment(mpslist, delta, fpoints, H, env_params, H_env=H_env)
  iter_data = [mpslist, A_C, fpoints, H_env, delta]
  return iter_data


@timed
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
  BENCHMARKER.clear()

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
  H_env = solve_environment(mpslist, delta_0, fpoints, H, env_params)
  iter_data = [mpslist, A_C, fpoints, H_env, delta_0]
  return vumps_work(H, iter_data, vumps_params, heff_params, env_params, writer)


@timed
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

  # mpslist, A_C, fpoints, H_env, delta
  mpslist, _, _, _, delta = iter_data
  E = ct.twositeexpect(mpslist, H).array
  writer.write("Initial energy: " + str(E))
  writer.write("And so it begins...")
  for Niter in range(Niter0, vumps_params["max_iter"]+Niter0):
    BENCHMARKER.increment_timestep()
    oldE = E
    iter_data = vumps_iteration(iter_data, H, heff_params, env_params,
                                vumps_params["gauge_match_mode"])
    mpslist, _, _, _, delta = iter_data
    E, dE, norm = diagnostics(mpslist, H, oldE)
    output(writer, Niter, delta, E, dE, norm)

    if delta <= vumps_params["gradient_tol"]:
      writer.write("Convergence achieved at iteration " + str(Niter))
      break

    if checkpoint_every is not None and (Niter+1) % checkpoint_every == 0:
      checkpoint(writer, H, iter_data, vumps_params, heff_params, env_params,
                 Niter)

  if Niter == max_iter - 1:
    writer.write("Maximum iteration " + str(max_iter) + " reached.")
  t_total = sum(BENCHMARKER.benchmarks["vumps_iteration"])
  writer.write("The main loops took " + str(t_total) + " seconds.")
  checkpoint(writer, H, iter_data, vumps_params, heff_params, env_params, Niter)
  return (iter_data, BENCHMARKER.benchmarks)

def checkpoint(writer, H, iter_data, vumps_params, heff_params, env_params,
               Niter):
  """
  Checkpoints the simulation.
  """
  writer.write("Checkpointing...")
  to_pickle = [H, iter_data, vumps_params, heff_params, env_params]
  to_pickle.append(Niter)
  writer.pickle(to_pickle, Niter)
###############################################################################
###############################################################################
