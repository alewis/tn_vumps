"""
Dictionaries bundling together parameters for vumps runs.
"""
from typing import Dict, Text


def krylov_params(n_krylov: int = 40, n_diag: int = 10,
                  dynamic_tolerance: bool = True, tol_coef: float = 0.01,
                  max_restarts: int = 10, reorth: bool = True) -> Dict:
  """
  Bundles parameters for the Lanczos eigensolver. These control
  the expense and accuracy of minimizing the effective Hamiltonians, which
  is typically the dominant expense.

  Args:
    n_krylov: Size of the Krylov subspace. This will be truncated if it
      exceeds the size of the relevant linear operators.
    n_diag: An argument to the TN Lanczos solver that presently has
      no effect.
    dynamic_tolerance: If True, the solver tolerance will be
      tol_coef * the vuMPS error estimate, and thus will vary throughout
      the simulation. Otherwise, it will be fixed and equal to tol_coef.
    tol_coef: If dynamic_tolerance is True, this number times the vuMPS
      error estimate will be used as the solver tolerance. Otherwise,
      this number will be used as the solver tolerance.
    max_restarts: The Krylov subspace will be rebuilt at most this many times
      per solve.
    reorth: If True the solver reorthogonalizes the Lanczos vectors at each
      iteration. This is more expensive, especially for large n_krylov and low
      chi, but may be necessary for vuMPS to converge.
  Returns:
    A dict bundling the given arguments.
  """
  return {"n_krylov": n_krylov, "n_diag": n_diag, "reorth": reorth,
          "dynamic_tolerance": dynamic_tolerance,
          "tol_coef": tol_coef, "max_restarts": max_restarts}


def gmres_params(n_krylov: int = 40, dynamic_tolerance: bool = True,
                 tol_coef: float = 0.01, max_restarts: int = 10) -> Dict:
  """
  Bundles parameters for the gmres solver. These control the expense and
  accuracy of the left and right environment Hamiltonians.
  Args:
    n_krylov: Size of the Krylov subspace. GMRES will exit early if
      convergence is reached before building the full subspace. This will be
      truncated if it exceeds the size of the relevant linear operators.
    dynamic_tolerance: If True, the solver tolerance will be
      tol_coef * the vuMPS error estimate, and thus will vary throughout
      the simulation. Otherwise, it will be fixed and equal to tol_coef.
    tol_coef: If dynamic_tolerance is True, this number times the vuMPS
      error estimate will be used as the solver tolerance. Otherwise,
      this number will be used as the solver tolerance.
    max_restarts: The Krylov subspace will be rebuilt at most this many times
      per solve.
  Returns:
    A dict bundling the given arguments.
  """
  return {"n_krylov": n_krylov,
          "dynamic_tolerance": dynamic_tolerance,
          "tol_coef": tol_coef, "max_restarts": max_restarts}


def vumps_params(gradient_tol: float = 1E-3, max_iter: int = 200,
                 gauge_match_mode: Text = "svd",
                 gradient_estimate_mode: Text = "null space",
                 delta_0: float = 0.1,
                 checkpoint_every: int = 500) -> Dict:
  """
  Bundles parameters for the vumps solver itself.

  Args:
    gradient_tol: Convergence is declared once the gradient norm is
      at least this small.
    max_iter: vuMPS ends after this many iterations even if unconverged.
    gauge_match_mode: Chooses the algorithm used to compute the three
      polar decompositions needed at the gauge matching step of each
      iteration.
      The following options are supported:
        x  svd (default): The decompositions are computed from an SVD.
                          Typically faster on the CPU.
        x  QDWH         : The decompositions are computed using the QDWH
                          algorithm. Typically faster on the GPU.
                          Theoretically this should also be faster on the
                          TPU.
    gradient_estimate_mode: Chooses the means by which the MPS gradient
      (the error esimate used by vuMPS) is estimated. The following
      options are supported:
        x null space (default): Computes the gradient via the MPS null
                                space. This is more accurate and tends to
                                improve stability, particularly when
                                dynamic_tolerance is used. However, it also
                                performs an extra SVD per iteration. This is
                                a negligable expense on the CPU, but can be
                                very large on the GPU.
        x gauge mismatch      : Estimates the gradient as largest of
                                A_C - A_L @ C and A_C - C @ A_R. This is
                                much cheaper, especially on the GPU, but
                                tends to degrade stability.
      "gauge mismatch" tends to result in much larger gradient estimates
      than null_space. One should therefore consider adjusting the solver
      tolerances when comparing the two.
  delta_0: The gradient estimate is initialized to this value.
  checkpoint_every: Simulation data is pickled at this periodicity. Note
    this has no effect on the observables, which are always computed once
    per iteration.

  Returns:
    A dict bundling the given arguments.
  """
  return {"gradient_tol": gradient_tol, "max_iter": max_iter,
          "gauge_match_mode": gauge_match_mode,
          "gradient_estimate_mode": gradient_estimate_mode,
          "delta_0": delta_0,
          "checkpoint_every": checkpoint_every}
