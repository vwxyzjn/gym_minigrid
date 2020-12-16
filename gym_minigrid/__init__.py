from pathlib import Path
import sys

# Add `gym_minigrid` parent directory to system path
path = str(Path(__file__).parent.parent.resolve().absolute())
sys.path.insert(0, path)

# Import the envs module so that envs register themselves
import gym_minigrid.envs

# Import wrappers so it's accessible when installing with pip
import gym_minigrid.wrappers
