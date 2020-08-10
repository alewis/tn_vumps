import functools

from typing import Any, Text, Callable, Optional, Dict, Sequence, Tuple, Union
import numpy as np

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

TwoTensors = Tuple[tn.Tensor, tn.Tensor]
ThreeTensors = Tuple[tn.Tensor, tn.Tensor, tn.Tensor]
DtypeType = Any


##########################################################################
##########################################################################
# Functions to handle output.
def ostr(instr: Text) -> Text:
  """
  Truncates to two decimal places.
  Args:
    instr: string to truncate.
  Returns:
    The truncated string.
  """
  return '{:1.2e}'.format(instr)


@timed
def output(writer: tn_vumps.writer.Writer, Niter: int, mps: ThreeTensors,
           H: tn.Tensor, delta: float, oldE: float) -> float:
  """
  Computes energy and norm, and handles output of observables.
  Args:
    writer: the Writer.
    Niter: iteration number.
    mps: The MPS.
    H: The Hamiltonian.
    delta: Error estimate.
    oldE: MPS energy from last iteration.
  Returns:
    E: The MPS energy.
  """
  E = ct.twositeexpect(mps, H)
  dE = tn.abs(E - oldE).array
  E = E.array
  norm = ct.mpsnorm(mps).array
  delta_str = ostr(delta)
  Estr = '{0:1.16f}'.format(E)
  dEstr = ostr(dE)
  dtstr = ostr(BENCHMARKER.benchmarks["vumps_iteration"][-1])
  outstr = (f"N = {Niter} | eps = {delta_str} | E = {Estr} | dE = {dEstr} | "
            f"dt = {dtstr}")
  writer.write(outstr)
  this_output = [Niter, E, delta, norm]
  writer.data_write(this_output)
  return E


def make_writer(outdir: Optional[Text] = None) -> tn_vumps.writer.Writer:
  """
  Initialize the Writer. Creates a directory in the appropriate place, and
  an output file with headers hardcoded here as 'headers'. The Writer,
  defined in writer.py, remembers the directory and will append to this
  file as the simulation proceeds. It can also be used to pickle data,
  notably the final wavefunction.

  Args:
    outdir: Path to the directory where output is to be saved. This
      directory will be created if it does not yet exist. Otherwise any contents
      with filename collisions during the simulation will be overwritten.

  Returns:
    writer: The Writer.
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
def solve_environment(mps: ThreeTensors, delta: float, fpoints: TwoTensors,
                      H: tn.Tensor, env_params: Dict,
                      H_env: Optional[TwoTensors] = None) -> TwoTensors:
  """
  Solves the linear systems for the left and right environment Hamiltonians.
  Args:
    mps: The MPS.
    delta: Gradient estimate.
    fpoints: right fixed point of A_L, left fixed point of A_R.
    H: The two-site local Hamiltonian.
    env_params: Params for the environment linear solver.
    H_env: Optional guesses for LH and RH.
  Returns:
    (LH, RH): The environment Hamiltonians.
  """
  if H_env is None:
    lh = None
    rh = None
  else:
    lh, rh = H_env
  A_L, _, A_R = mps
  rL, lR = fpoints
  LH = solve_for_LH(A_L, H, lR, env_params, delta, oldLH=lh)
  RH = solve_for_RH(A_R, H, rL, env_params, delta, oldRH=rh)
  return LH, RH


def LH_matvec(v: tn.Tensor, lR: tn.Tensor, A_L: tn.Tensor) -> tn.Tensor:
  """
  The linear operator used as A in Ax = b to solve for LH. Defining this
  function inline would result in greatly degraded performance with Jax due to
  extra traces.
  Args:
    v: Current guess solution for LH.
    lR: left fixed point of A_R.
    A_L: A_L from the MPS.
  Returns:
    The action of the linear operator.
  """
  Th_v = ct.XopL(A_L, v)
  vR = ct.projdiag(v, lR)
  return v - Th_v + vR


@timed
def solve_for_LH(A_L: tn.Tensor, H: tn.Tensor, lR: tn.Tensor, env_params: Dict,
                 delta: float, oldLH: Optional[tn.Tensor] = None) -> tn.Tensor:
  """
  Finds the renormalized left environment Hamiltonian.
  Args:
    A_L: A_L from the MPS.
    H: The two-site local Hamiltonian.
    lR: left fixed point of A_R.
    env_params: params for the linear solver.
    delta: Current vuMPS error; used to set solver tolerance.
    oldLH: LH from the previous iteration.
  Returns:
    The left environment Hamiltonian.
  """
  tol = env_params["tol_coef"]
  if env_params["dynamic_tolerance"]:
    tol *= delta
  hL_bare = ct.compute_hL(A_L, H)
  hL_div = ct.projdiag(hL_bare, lR)
  hL = hL_bare - hL_div
  matvec_args = [lR, A_L]
  n_krylov = min(env_params["n_krylov"], hL.size**2)
  LH, _ = tn.linalg.krylov.gmres(LH_matvec,
                                 hL,
                                 A_args=matvec_args,
                                 x0=oldLH,
                                 tol=tol,
                                 num_krylov_vectors=n_krylov,
                                 maxiter=env_params["max_restarts"])
  benchmark.block_until_ready(LH)
  return LH


def RH_matvec(v: tn.Tensor, rL: tn.Tensor, A_R: tn.Tensor) -> tn.Tensor:
  """
  The linear operator used as A in Ax = b to solve for RH. Defining this
  function inline would result in greatly degraded performance with Jax due to
  extra traces.
  Args:
    v: Current guess solution for LH.
    rL: right fixed point of A_L.
    A_R: A_R from the MPS.
  Returns:
    The action of the linear operator.
  """
  Th_v = ct.XopR(A_R, v)
  Lv = ct.projdiag(rL, v)
  return v - Th_v + Lv


@timed
def solve_for_RH(A_R: tn.Tensor, H: tn.Tensor, rL: tn.Tensor, env_params: Dict,
                 delta: float, oldRH: Optional[tn.Tensor] = None) -> tn.Tensor:
  """
  Finds the renormalized right environment Hamiltonian.
  Args:
    A_R: A_R from the MPS.
    H: The two-site local Hamiltonian.
    rL: right fixed point of A_L.
    env_params: params for the linear solver.
    delta: Current vuMPS error; used to set solver tolerance.
    oldRH: RH from the previous iteration.
  Returns:
    The right environment Hamiltonian.
  """
  tol = env_params["tol_coef"]
  if env_params["dynamic_tolerance"]:
    tol *= delta
  hR_bare = ct.compute_hR(A_R, H)
  hR_div = ct.projdiag(rL, hR_bare)
  hR = hR_bare - hR_div
  matvec_args = [rL, A_R]
  n_krylov = min(env_params["n_krylov"], hR.size**2)
  RH, _ = tn.linalg.krylov.gmres(RH_matvec,
                                 hR,
                                 A_args=matvec_args,
                                 x0=oldRH,
                                 tol=tol,
                                 num_krylov_vectors=n_krylov,
                                 maxiter=env_params["max_restarts"])
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
  """
  Finds the eigenpair of the Hermitian matvec with the most negative eigenvalue
  by the explicitly restarted Lanczos method.
  Args:
    matvec: The function x = matvec(x, *mv_args) representing the operator.
    mv_args: A list of fixed positional arguments to matvec.
    guess: Guess eigenvector.
    tol: The degree to which the returned eigenpair is allowed to deviated from
      solving the eigenvalue equation.
    max_restarts: The Krylov space will be rebuilt at most this many times even
      if tol is not achieved.
    n_krylov: Size of the Krylov space to build.
    reorth: If True the Krylov space will be explicitly reorthogonalized at
      each solver iteration.
    n_diag: An argument to TN Lanczos that currently has no effect.
    verbose: If True a warning will be printed to console if tol was not
      reached.
    Returns:
      ev, eV, err: The eigenvalue, vector, and error.
  """
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
  return ev, eV, err


###############################################################################
# HAc
@timed
def minimize_HAc(mps: ThreeTensors, A_C: tn.Tensor, Hlist: ThreeTensors,
                 delta: float, heff_params: Dict) -> Tuple[float, tn.Tensor]:
  """
  Minimizes the effective Hamiltonian HAc to find the new A_C.
  Args:
    mps: The MPS.
    A_C: The current A_C.
    Hlist: The environment and local Hamiltonians, (H, LH, RH).
    delta: Current vuMPS error. Used to set the solver tolerance in
      dynamic_tolerance mode. Otherwise has no effect.
    heff_params: Params for the eigensolver.
  Returns:
    ev, newA_C: The eigenvalue, and the updated A_C.
  """
  A_L, _, A_R = mps
  tol = heff_params["tol_coef"]
  if heff_params["dynamic_tolerance"]:
    tol *= delta
  mv_args = [A_L, A_R, *Hlist]
  n_krylov = min(heff_params["n_krylov"], A_C.size**2)
  ev, newA_C, _ = minimum_eigenpair(ct.apply_HAc, mv_args, A_C, tol,
                                    max_restarts=heff_params["max_restarts"],
                                    n_krylov=n_krylov,
                                    reorth=heff_params["reorth"],
                                    n_diag=heff_params["n_diag"])
  return ev, newA_C


###############################################################################
# Hc
@timed
def minimize_Hc(mps: ThreeTensors, Hlist: ThreeTensors, delta: float,
                heff_params: Dict) -> Tuple[float, tn.Tensor]:
  """
  Minimizes the effective Hamiltonian Hc to find the new C.
  Args:
    mps: The MPS.
    Hlist: The environment and local Hamiltonians, (H, LH, RH).
    delta: Current vuMPS error. Used to set the solver tolerance in
      dynamic_tolerance mode. Otherwise has no effect.
    heff_params: Params for the eigensolver.
  Returns:
    ev, newC: The eigenvalue, and the updated C.
  """
  A_L, C, A_R = mps
  tol = heff_params["tol_coef"]
  if heff_params["dynamic_tolerance"]:
    tol *= delta
  mv_args = [A_L, A_R, *Hlist]
  n_krylov = min(heff_params["n_krylov"], C.size**2)
  ev, newC, _ = minimum_eigenpair(ct.apply_Hc, mv_args, C, tol,
                                  max_restarts=heff_params["max_restarts"],
                                  n_krylov=n_krylov,
                                  reorth=heff_params["reorth"],
                                  n_diag=heff_params["n_diag"])
  return ev, newC


@timed
def gauge_match(A_C: tn.Tensor, C: tn.Tensor, mode: Text = "svd") -> TwoTensors:
  """
  Return approximately gauge-matched A_L and A_R from A_C and C
  using a polar decomposition.

  A_L and A_R are chosen to minimize ||A_C - A_L C|| and ||A_C - C A_R||.
  The respective solutions are the isometric factors in the
  polar decompositions of A_C @ C.H and C.H @ A_C.

  Args:
    A_C: Current estimate of A_C.
    C: C from the MPS.
    mode: Chooses the algorithm for the polar decomposition. See the docstring
     of vumps_params in params.py.

  Returns:
    A_L, A_R: Such that A_L C A_R minimizes ||A_C - A_L C|| and ||A_C - C A_R||,
      with A_L (A_R) left (right) isometric.
  """
  UC = tn_vumps.polar.polarU(C, mode=mode)  # unitary
  UAc_l = tn_vumps.polar.polarU(A_C, mode=mode)  # left isometric
  A_L = ct.rightmult(UAc_l, UC.H)
  UAc_r = tn_vumps.polar.polarU(A_C, pivot_axis=1, mode=mode)  # right iso
  A_R = ct.leftmult(UC.H, UAc_r)
  return A_L, A_R


@timed
def vumps_delta(mps: ThreeTensors, A_C: tn.Tensor, oldA_L: tn.Tensor,
                mode: Text):
  """
  Estimate the current vuMPS error.
  Args:
    mps: The MPS.
    A_C: Current A_C.
    oldA_L: A_L from the last iteration.
    mode: gradient_estimate_mode in vumps_params. See that docstring for
      details.
  """
  if mode == "gauge mismatch":
    A_L, C, A_R = mps
    eL = tn.norm(A_C - ct.rightmult(A_L, C))
    eR = tn.norm(A_C - ct.leftmult(C, A_R))
    delta = max(eL, eR)
  elif mode == "null space":
    A_Ldag = tn.pivot(oldA_L, pivot_axis=2).H
    N_Ldag = tn_vumps.polar.null_space(A_Ldag)
    N_L = N_Ldag.H
    A_Cmat = tn.pivot(A_C, pivot_axis=2)
    B = N_L @ A_Cmat
    delta = tn.norm(B)
  else:
    raise ValueError("Invalid mode {mode}.")
  return delta




@timed
def apply_gradient(iter_data: Dict, H: tn.Tensor, heff_params: Dict,
    vumps_params: Dict) -> Tuple[ThreeTensors, tn.Tensor, float]:
  """
  Move a step along the MPS gradient.
  Args:
    iter_data: Data from this vuMPS iteration.
    H: The two-site local Hamiltonian.
    heff_params: Params for the eigensolver.
    vumps_params: Params for the vumps run.
  Returns:
    newmps: The updated MPS.
    A_C: The updated A_C.
    delta: New error estimate.
  """
  mps = iter_data["mps"]
  delta = iter_data["delta"]
  Hlist = [H, *iter_data["H_env"]]
  _, A_C = minimize_HAc(mps, iter_data["A_C"], Hlist, delta, heff_params)
  _, C = minimize_Hc(mps, Hlist, delta, heff_params)
  A_L, A_R = gauge_match(A_C, C, mode=vumps_params["gauge_match_mode"])
  newmps = (A_L, C, A_R)
  delta = vumps_delta(newmps, A_C, mps[0],
                      mode=vumps_params["gradient_estimate_mode"])
  return newmps, A_C, delta
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# Main loop and friends.
@timed
def vumps_approximate_tm_eigs(C) -> TwoTensors:
  """
  Returns the approximate transfer matrix dominant eigenvectors,
  rL ~ C^dag C, and lR ~ C Cdag = rLdag, both trace-normalized.
  Args:
    C: From the MPS.
  Returns:
    rL, lR: right fixed point of L, left fixed point of R.
  """
  rL = C.H @ C
  rL /= tn.trace(rL)
  lR = rL.H
  return rL, lR


@timed
def vumps_initialization(d: int, chi: int, dtype: Optional[DtypeType] = None,
  backend: Optional[Text] = None) -> Tuple[ThreeTensors, tn.Tensor, TwoTensors]:
  """
  Generate a random uMPS in mixed canonical forms, along with the left
  dominant eV L of A_L and right dominant eV R of A_R.

  Args:
    d: Physical dimension.
    chi: Bond dimension.
    dtype: Data dtype of tensors.
    backend: The backend.
  Returns:
    mps = (A_L, C, A_R): A_L and A_R have shape (chi, d, chi), and are
      respectively left and right isometric. C is the (chi, chi)
      centre of orthogonality.
    A_C: A_L @ C. One of the equations vumps minimizes is
      A_L @ C = C @ A_R = A_C.
    fpoints = [rL, lR]: C^dag @ C and C @ C^dag respectively. Will converge
      to the left and right fixed points of A_R and A_L. Both are chi x chi.
  """
  A_1 = tn.randn((chi, d, chi), dtype=dtype, backend=backend)
  A_L, _ = tn.linalg.linalg.qr(A_1, pivot_axis=2, non_negative_diagonal=True)
  C, A_R = tn.linalg.linalg.rq(A_L, pivot_axis=1, non_negative_diagonal=True)
  C /= tn.linalg.linalg.norm(C)
  A_C = ct.rightmult(A_L, C)
  L0, R0 = vumps_approximate_tm_eigs(C)
  fpoints = (L0, R0)
  mps = [A_L, C, A_R]
  benchmark.block_until_ready(R0)
  return (mps, A_C, fpoints)


@timed
def vumps_iteration(iter_data: Dict, H: tn.Tensor,
                    params: Tuple[Dict, Dict, Dict]) -> Dict:
  """
  One main iteration of VUMPS.
  Args:
    iter_data: Bundles data passed between iterations.
    H: The two-site local Hamiltonian.
    params: vumps_params, heff_params, env_params.
  Returns:
    Updated iter_data.
  """

  mps, A_C, delta = apply_gradient(iter_data, H, params["heff"],
      params["vumps"])
  fpoints = vumps_approximate_tm_eigs(mps[1])
  H_env = solve_environment(mps, delta, fpoints, H, params["env"],
                            H_env=iter_data["H_env"])
  return {"mps": mps, "A_C": A_C, "fpoints": fpoints, "H_env": H_env,
          "delta": delta}


@timed
def vumps(H: tn.Tensor, chi: int,
          out_directory: Text = "./vumps",
          vumps_params: Optional[Dict] = None,
          heff_params: Optional[Dict] = None,
          env_params: Optional[Dict] = None
          ) -> Tuple[Dict, Dict]:
  """
  Find the ground state of a uniform two-site Hamiltonian
  using Variational Uniform Matrix Product States. This is a gradient
  descent method minimizing the distance between a given MPS and the
  best approximation to the physical ground state at its bond dimension.

  This interface function initializes vumps from a random initial state.

  Args:
    H: The two-site local Hamiltonian whose ground state is to be found.
    chi: MPS bond dimension.
    out_directory: Output is saved here.

    The remaining arguments are generated by and documented in the functions
    found in params.py.
    vumps_params: Additional parameters pertaining to the vuMPS solver.
    heff_params: Additional parameters pertaining to the sparse eigensolver
      that finds the effective Hamiltonians.
    env_params: Additional parameters pertaining to the linear solver that
      finds the environment Hamiltonians.

  Returns:
    iter_data: Bundles the data carried between vuMPS iterations.
      - mps = (A_L, C, A_R): The MPS tensors.
      - A_C: The estimate of A_C. Once converged, and only then, we have
         A_C = A_L @ C = C @ A_R.
      -fpoints = (rL, lR): The right dominant eigentensor of A_L, and the left
         dominant eigentensor of A_R.
      -H_env = (LH, RH): The left and right environment Hamiltonians.
      -delta: The error estimate of the vuMPS run. The exact meaning of this
         quantity depends on the choice of
         vumps_params["gradient_estimate_mode"], but when small it will be
         proportional to the MPS gradient.
    timings: A dictionary of timings-per-timestep for each of various functions.
  """
  BENCHMARKER.clear()

  if vumps_params is None:
    vumps_params = tn_vumps.params.vumps_params()
  if heff_params is None:
    heff_params = tn_vumps.params.krylov_params()
  if env_params is None:
    env_params = tn_vumps.params.gmres_params()
  params = {"vumps": vumps_params, "heff": heff_params, "env": env_params}

  writer = make_writer(out_directory)
  d = H.shape[0]
  writer.write("vuMPS, a love story.")
  writer.write("**************************************************************")
  mps, A_C, fpoints = vumps_initialization(d, chi, H.dtype,
      backend=H.backend.name)
  delta_0 = vumps_params["delta_0"]
  H_env = solve_environment(mps, delta_0, fpoints, H, env_params)
  iter_data = {"mps": mps,
               "A_C": A_C,
               "fpoints": fpoints,
               "H_env": H_env,
               "delta": delta_0}
  return vumps_work(H, iter_data, params, writer)


@timed
def vumps_work(H: tn.Tensor, iter_data: Dict, params: Dict,
               writer: tn_vumps.writer.Writer,
               Niter0: int = 1) -> Tuple[Dict, Dict]:
  """
  Main work loop for vumps. Should be accessed via one of the interface
  functions above.

  Args:
    H: The two-site local Hamiltonian.
    iter_data: Carries data between runs.
    params: Nested Dict of vuMPS params.
    writer: The Writer.
    Niter0: Numbers the initial iteration.
  Returns:
    iter_data: Of the final iteration.
    benchmarks: A Dict of timing data.
  """
  checkpoint_every = params["vumps"]["checkpoint_every"]
  max_iter = params["vumps"]["max_iter"]
  gradient_tol = params["vumps"]["gradient_tol"]
  oldE = ct.twositeexpect(iter_data["mps"], H).array
  writer.write(f"Initial energy: {oldE}.\n And so it begins...")
  for Niter in range(Niter0, max_iter + Niter0):
    BENCHMARKER.increment_timestep()
    iter_data = vumps_iteration(iter_data, H, params)
    delta = iter_data["delta"]
    oldE = output(writer, Niter, iter_data["mps"], H, delta, oldE)

    if delta <= gradient_tol:
      writer.write(f"Convergence achieved at iteration {Niter}.")
      break

    if checkpoint_every is not None and (Niter+1) % checkpoint_every == 0:
      checkpoint(writer, H, iter_data, params, Niter)

  if Niter == max_iter - 1:
    writer.write(f"Maximum iteration {max_iter} reached.")
  t_total = sum(BENCHMARKER.benchmarks["vumps_iteration"])
  writer.write(f"The main loops took {t_total} seconds.")
  checkpoint(writer, H, iter_data, params, Niter)
  return iter_data, BENCHMARKER.benchmarks


def checkpoint(writer: tn_vumps.writer.Writer, H: tn.Tensor, iter_data: Dict,
               params: Dict[Text, Dict], Niter: int):
  """
  Checkpoints the simulation.
  Args:
    writer: The Writer.
    H : The Hamiltonian.
    iter_data: iter_data from the vuMPS run.
    params: The vuMPS params.
    Niter: Iteration count.
  Returns:
    None
  """
  writer.write("Checkpointing...")
  to_pickle = [H, iter_data, params]
  to_pickle.append(Niter)
  writer.pickle(to_pickle, Niter)
###############################################################################
###############################################################################
