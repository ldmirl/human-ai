"""Imports all environments so that they register themselves with the API.

This protocol is the same as OpenAI Gym, and allows all environments to be
simultaneously registered with multigrid as a package.
"""
from multigrid.envs.registration import registry, register, make

from multigrid.envs.lumberjacks import *
from multigrid.envs.plate_path import *
from multigrid.envs.predator_prey import *
