from typing import List
import dm_env
import pygame
import numpy as np
import pathlib

import multigrid
from multigrid import Actions
from multigrid.wrappers import FlatObsWrapper, DMEnvWrapper

from torchmarl import utils
from torchmarl.modules import VDN
from torchmarl.buffers import restart, transit, terminate


def parse_arguments(parser):
  parser.add_argument("--add-agent-id", action="store_true", default=True)
  parser.add_argument("--add-prev-action", action="store_true", default=True)
  parser.add_argument("--device-id", type=int, default=0)
  # VDN specific arguments
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--buffer-size", type=int, default=5000)
  parser.add_argument("--gamma", type=float, default=0.99)
  parser.add_argument("--lr", type=float, default=5e-4)
  parser.add_argument("--optim-alpha", type=float, default=0.99)
  parser.add_argument("--optim-eps", type=float, default=1e-5)
  parser.add_argument("--h-dim", type=int, default=64)
  parser.add_argument("--eps-max", type=float, default=1.0)
  parser.add_argument("--eps-min", type=float, default=0.05)
  parser.add_argument("--eps-decay", type=float, default=50000)
  parser.add_argument("--grad-norm-clip", type=float, default=10)
  parser.add_argument("--double-q", action="store_true", default=True)
  parser.add_argument("--target-update-freq", type=int, default=200)
  parser.add_argument("--use-state", action="store_true", default=False)
  parser.add_argument("--embed-dim", type=int, default=64)  
  return parser

class ManualControl:
  def __init__(self, env: dm_env.Environment, actor) -> None:
    self.env = env
    self.actor = actor
    self.closed = False
    self.agent_idx = 0

    self.avail_actions = np.ones((env.n_agents, env.n_actions))
    self.avail_actions = np.expand_dims(self.avail_actions, axis=0)

    self.initialized = False
    self.cout = open("out-experiment-2.txt", "w")
  
  def reset(self):
    timestep = self.env.reset() 
    self.env.render()
    
    # NOTE RAFAEL: Uncomment this if you want to play music
    # Motivational music: thescript.mp3
    # Neutral music: debussy.mp3
    if not self.initialized:
      pygame.mixer.music.load(path / "music/debussy.mp3")
      pygame.mixer.music.play(-1)
      self.initialized = True
    
    self.actor.init_hidden()
    return timestep

  def step(self, actions: List[Actions]):
    timestep, info = self.env.step(actions)
    reward = sum(timestep.reward)
    print(f"step={self.env.step_count}, reward={reward}")

    if timestep.last():
      print("terminated!")
      self.cout.write(f"terminated! \n")
      timestep = self.reset()
      timestep = restart(timestep.observation, self.avail_actions, None)
      self.actor.init_hidden()
      return timestep
    else:
      self.env.render()
    
    timestep = transit(timestep.reward, timestep.observation, self.avail_actions, None)
    return timestep

  def run(self):
    """Start the window display with blocking event loop."""
    timestep = self.reset()
    prev_action = np.zeros((env.n_agents, env.n_actions))
    eye = np.eye(env.n_agents)
    timestep = restart(timestep.observation, self.avail_actions, None)
    while not self.closed:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.closed = True
          self.env.close()
          break
        if event.type == pygame.KEYDOWN:
          event.key = pygame.key.name(int(event.key))
          if timestep.observation.ndim != 3:
            observation = np.concatenate([timestep.observation, prev_action, eye], axis=1)
            observation = np.expand_dims(observation, axis=0)
            timestep = timestep._replace(observation=observation)
          self.closed, timestep, actions = self.key_handler(event, timestep)
          prev_action[np.arange(env.n_agents), actions] = 1.

  def key_handler(self, event, timestep):
    key: str = event.key
    print("pressed", key)

    if key == "escape":
      self.env.close()
      return True, timestep, None
    if key == "backspace":
      self.reset()
      return False, timestep, None
    
    key_to_action = {
      "left": Actions.LEFT,
      "right": Actions.RIGHT,
      "up": Actions.FORWARD,
      "space": Actions.TOGGLE,
      "pageup": Actions.PICKUP,
      "pagedown": Actions.DROP,
      "tab": Actions.PICKUP,
      "left shift": Actions.DROP,
      "enter": Actions.DONE,
    }
    actions = None # NOTE: In case user presses a key not in key_to_action
    if key in key_to_action.keys():
      action = key_to_action[key]
      actions = self.actor.select_actions(timestep, env.step_count, False)
      actions = actions.squeeze()
      actions[self.agent_idx] = action
      self.cout.write(f"{self.env.step_count}, {actions}, IDX {self.agent_idx} \n")
      timestep = self.step(actions)
    
    # NOTE: DO NOT LET USER CHANGE AGENT INDEX
    # if key in [str(_) for _ in range(1, self.env.n_agents + 1)]:
    #   self.agent_idx = int(key) - 1
    return False, timestep, actions

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=None,
  )
  parser.add_argument(
    "--tile-size", type=int, help="size at which to render tiles", default=32
  )
  parser.add_argument(
    "--agent-pov",
    action="store_true",
    help="draw the agent sees (partially observable view)",
  )
  parser.add_argument(
    "--agent-view-size",
    type=int,
    default=5,
    help="set the number of grid spaces visible in agent-view ",
  )
  parser.add_argument(
    "--screen-size",
    type=int,
    default="640",
    help="set the resolution for pygame rendering (width and height)",
  )
  parser = parse_arguments(parser)
  args = parser.parse_args()

  # NOTE RAFAEL: Just uncomment the env you want to be bundling 
  # (only the call line -- keep the object attributes below it).
  # env = multigrid.PredatorPrey( 
  # env = multigrid.PlatePath(
  env = multigrid.LumberJack(
    tile_size=args.tile_size,
    agent_pov=args.agent_pov,
    agent_view_size=args.agent_view_size,
    screen_size=args.screen_size, 
    render_mode="human" 
  )
  env = FlatObsWrapper(env)
  env = DMEnvWrapper(env)

  obs_shape = 75 # NOTE: HARD CODED FOR NOW! DO NOT CHANGE!
  if args.add_agent_id: obs_shape += env.n_agents
  if args.add_prev_action: obs_shape += env.n_actions
  device = utils.device(device_id=args.device_id)

  actor = VDN(
    in_dim=obs_shape,
    h_dim=args.h_dim,
    out_dim=env.n_actions,
    n_agents=env.n_agents,
    lr=args.lr,
    gamma=args.gamma,
    grad_norm_clip=args.grad_norm_clip,
    buffer_maxsize=args.buffer_size,
    batch_size=args.batch_size,
    optim_alpha=args.optim_alpha,
    optim_eps=args.optim_eps,
    eps_max=args.eps_min,
    eps_min=args.eps_min,
    eps_decay=args.eps_decay,
    double_q=args.double_q,
    target_update_freq=args.target_update_freq,
    use_state=args.use_state,
    state_shape=obs_shape,
    embed_dim=args.embed_dim,
    device=device,
  )

  
  # NOTE RAFAEL: Normally this should run, but just in case, you need to consider the path
  # in respect to the bundled folder _internal (see examples)
  path = pathlib.Path(__file__).parents[1] # use 2 for building // 0 for testing
  
  # NOTE RAFAEL: Just uncomment the env you want to be bundling
  # actor.load(path / "checkpoints/PredatorPrey-v0", 11285, 491588)
  # actor.load(path / "checkpoints/PlatePath-v0", 11224, 1112894)
  actor.load(path / "checkpoints/LumberJack-NoIndicator-v0", 11600, 1160000)

  manual_control = ManualControl(env, actor)
  manual_control.run()
