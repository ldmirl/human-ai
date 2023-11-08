# NOTE: Kindly stolen from https://github.com/openai/gym/blob/master/gym/envs/registration.py
import re
import importlib
from typing import Optional, Tuple

ENV_ID_RE = re.compile(
    r"^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)

def parse_env_id(id: str) -> Tuple[Optional[str], str, Optional[int]]:
  """Parse environment ID string format.

  This format is true today, but it's *not* an official spec.
  [namespace/](env-name)-v(version)    env-name is group 1, version is group 2
  2016-10-31: We're experimentally expanding the environment ID format
  to include an optional namespace.

  :param id: The environment id to parse
  :returns: A tuple of environment namespace, environment name and version number
  :raises RunetimeError:  If the environment id does not a valid environment regex
  """
  match = ENV_ID_RE.fullmatch(id)
  if not match:
    raise RuntimeError(
      f"Malformed environment ID: {id}."
      f"(Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))"
    )
  namespace, name, version = match.group("namespace", "name", "version")
  if version is not None:
      version = int(version)

  return namespace, name, version

def get_env_id(ns: Optional[str], name: str, version: Optional[int]) -> str:
  """
  Get the full env ID given a name and (optional) version and namespace. 
  Inverse of :meth:`parse_env_id`.

  :param ns: The environment namespace
  :param name: The environment name
  :param version: The environment version
  :returns: The environment id
  """
  full_name = name
  if version is not None: full_name += f"-v{version}"
  if ns is not None: full_name = ns + "/" + full_name
  return full_name

def load(name: str) -> callable:
  """Loads an environment with name and returns an environment creation function

  :param name: The environment name
  :returns: Calls the environment constructor
  """
  mod_name, attr_name = name.split(":")
  mod = importlib.import_module(mod_name)
  fn = getattr(mod, attr_name)
  return fn

# Global registry of environments. Meant to be accessed through `register` and `make`
registry = {}

def register(env_id: str, entry_point, **kwargs):
  global registry
  ns_id, name, version = parse_env_id(env_id)
  full_id = get_env_id(ns_id, name, version)
  registry[full_id] = {"entry_point": entry_point, "kwargs": kwargs}


def make(env_id, **kwargs):
  """Create an environment according to the given ID.
  To find all available environments use `multigrid.envs.registry.keys()` for all valid ids.
  
  :param env_id: Name of the environment. Optionally, a module to import can be included, eg. 'module:Env-v0'
  :param kwargs: Additional arguments to pass to the environment constructor.
  :returns: An instance of the environment.
  :raises VaueError: If the ``env_id`` doesn't exist then an error is raised
  """
  module, env_id = (None, env_id) if ":" not in env_id else env_id.split(":")
  if module is not None:
    try: importlib.import_module(module)
    except ModuleNotFoundError as e:
      raise ModuleNotFoundError(
        f"{e}. Environment registration via importing a module failed. "
        f"Check whether '{module}' contains env registration and can be imported."
      )
  spec_ = registry.get(env_id)
  _kwargs = spec_['kwargs'].copy()
  _kwargs.update(kwargs)
  if spec_['entry_point'] is None: 
    raise ValueError(f"{spec_.id} registered but entry_point is not specified")
  elif callable(spec_['entry_point']):
    creator = spec_['entry_point']
  else: # Assume it's a string
    creator = load(spec_['entry_point'])
  env = creator(**_kwargs)
  return env