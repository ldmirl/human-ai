from typing import List
import pygame

import multigrid
from multigrid import Actions

class ManualControl:
  def __init__(self, env) -> None:
    self.env = env
    self.closed = False
    self.agent_idx = 0
  
  def reset(self):
    self.env.reset()
    self.env.render()

  def step(self, actions: List[Actions]):
    _, reward, terminated, _ = self.env.step(actions)
    print(f"step={self.env.step_count}, reward={reward}")

    if all(terminated):
      print("terminated!")
      self.reset()
    else:
      self.env.render()

  def run(self):
    """Start the window display with blocking event loop."""
    self.reset()
    while not self.closed:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.closed = True
          self.env.close()
          break
        if event.type == pygame.KEYDOWN:
          event.key = pygame.key.name(int(event.key))
          self.closed = self.key_handler(event)

  def key_handler(self, event):
    key: str = event.key
    print("pressed", key)

    if key == "escape":
      self.env.close()
      return True
    if key == "backspace":
      self.reset()
      return False
    
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
    if key in key_to_action.keys():
      action = key_to_action[key]
      actions = [Actions.DONE for _ in range(self.env.n_agents)]
      actions[self.agent_idx] = action
      self.step(actions)
    
    if key in [str(_) for _ in range(1, self.env.n_agents + 1)]:
      self.agent_idx = int(key) - 1
    return False

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--env-id",
    type=str,
    help="multigrid environment to load",
    choices=multigrid.envs.registry.keys(),
    default="FourRooms-v0"
  )
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
  args = parser.parse_args()

  env = multigrid.make(
    env_id=args.env_id, 
    tile_size=args.tile_size,
    agent_pov=args.agent_pov,
    agent_view_size=args.agent_view_size,
    screen_size=args.screen_size, 
    render_mode="human") 
  manual_control = ManualControl(env)
  manual_control.run()