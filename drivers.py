"""
Interface functions for VUMPS.
"""
from typing import Optional, Dict, Text, Tuple, Any
import numpy as np
import pickle as pkl
import tn_vumps.matrices as mat
import tn_vumps.vumps as vumps
import tensornetwork as tn


def runvumps(H: tn.Tensor, bond_dimension: int,
             out_directory: Text = "./vumps_output",
             vumps_params: Optional[Dict] = None,
             heff_params: Optional[Dict] = None,
             env_params: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """
  Performs a vumps simulation of some two-site local Hamiltonian H.

  Args:
    H: The Hamiltonian to be simulated.
    bond_dimension: Bond dimension of the MPS.
    out_directory: Output is saved here. The directory is created
      if it doesn't exist.
    vumps_params: Hyperparameters for the vumps solver. Formed
      by 'vumps_params'.
    heff_params: Hyperparameters for an eigensolve of certain
      'effective Hamiltonians'. Formed by 'krylov_params()'.
    env_params: Hyperparameters for a linear solve that finds the effective
      Hamiltonians. Formed by 'solver_params()'.
  Returns:
    iter_data, timings.
    iter_data: Dictionary of data carried between vuMPS iterations.
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
  out = vumps.vumps(H, bond_dimension, out_directory=out_directory,
                    vumps_params=vumps_params, heff_params=heff_params,
                    env_params=env_params)
  return out


def vumps_XX(bond_dimension: int,
             out_directory: Text = "./vumps",
             backend: Text = "numpy", dtype: Any = np.float32,
             vumps_params: Optional[Dict] = None,
             heff_params: Optional[Dict] = None,
             env_params: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """
  Performs a vumps simulation of the XX model,
  H = XX + YY.
  Args:
    bond_dimension: Of the MPS.
    out_directory: Location of output.
    backend: The backend, "numpy" or "jax".
    dtype: The dtype.
    vumps_params: For the vumps solver.
    heff_params: For the Lanczos solver.
    env_params: For the linear solver.
  Returns:
    iter_data, timings.
  """
  H = mat.H_XX(backend=backend, dtype=dtype)
  out = runvumps(H, bond_dimension,
                 out_directory=out_directory,
                 vumps_params=vumps_params, heff_params=heff_params,
                 env_params=env_params)
  return out


def vumps_ising(J: float, h: float, bond_dimension: int,
                out_directory: Text = "./vumps",
                backend: Text = "numpy", dtype: Any = np.float32,
                vumps_params: Optional[Dict] = None,
                heff_params: Optional[Dict] = None,
                env_params: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """
  Performs a vumps simulation of the Ising model,
  H = J * XX + h * ZI
  Args:
    J, h: Hamiltonian parameters.
    bond_dimension: Of the MPS.
    out_directory: Location of output.
    backend: The backend, "numpy" or "jax".
    dtype: The dtype.
    vumps_params: For the vumps solver.
    heff_params: For the Lanczos solver.
    env_params: For the linear solver.
  Returns:
    iter_data, timings.
  """
  H = mat.H_ising(h, J=J, backend=backend, dtype=dtype)
  out = runvumps(H, bond_dimension,
                 out_directory=out_directory,
                 vumps_params=vumps_params, heff_params=heff_params,
                 env_params=env_params)
  return out


def vumps_from_checkpoint(checkpoint_path: Text,
                          out_directory: Text = "./vumps_load",
                          new_vumps_params: Optional[Dict] = None,
                          new_heff_params: Optional[Dict] = None,
                          new_env_params: Optional[Dict] = None):
  """
  Resumes a vuMPS simulation from checkpointed data, optionally with new
  simulation parameters.

  Args:
    checkpoint_path: Path to the checkpoint .pkl file.
    out_directory: Where to save the new data.
    new_vumps_params, new_heff_params, new_env_params: Optional new
      parameter dicts.
  Returns:
    iter_data, timings.
  """
  writer = vumps.make_writer(out_directory)
  with open(checkpoint_path, "rb") as f:
    chk = pkl.load(f)

  H, iter_data, vumps_params, heff_params, env_params, Niter = chk
  if new_vumps_params is not None:
    vumps_params = {**vumps_params, **new_vumps_params}

  if new_heff_params is not None:
    heff_params = {**heff_params, **new_heff_params}

  if new_env_params is not None:
    env_params = {**env_params, **new_env_params}

  out = vumps.vumps_work(H, iter_data, vumps_params, heff_params, env_params,
                         writer, Niter0=Niter)
  return out
