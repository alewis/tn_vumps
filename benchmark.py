import time
import functools

class Benchmarker:
  def __init__(self):
    self.clear()

  def increment_timestep(self):
    """
    Adds a new timestep to TIMINGS. The timestep counter is incremented by
    1. Any entries in the dict  have a new [0,] appended.
    """
    self.timestep += 1
    for key in self.benchmarks:
      this_list = self.benchmarks[key]
      this_list = this_list + [0.,]
      self.benchmarks[key] = this_list

  def benchmark(self, name, dt):
    if name not in self.benchmarks:
      init = [0. for _ in range(self.timestep + 1)]
      self.benchmarks[name] = init
    self.benchmarks[name][-1] += dt

  def clear(self):
    self.timestep = 0
    self.benchmarks = dict()


def timed(func, benchmarker=None):
  """
  Logs the execution time of the decorated function within TIMINGS. If this
  function has been executed already, the execution time is added to the
  relevant entry at the relevant timestep. Otherwise a new entry is created,
  consisting of [0,...] up until the present timestep, followed by the
  execution time.
  """
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    dt = time.perf_counter() - start
    if benchmarker is not None:
      benchmarker.benchmark(func.__name__, dt)
    return result
  return wrapper

def block_until_ready(tensor):
  if tensor.backend.name == "jax":
    _ = tensor.array.block_until_ready()
