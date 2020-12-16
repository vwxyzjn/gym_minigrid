import gym
import gym_minigrid
from PIL import Image
import numpy as np
from gym.wrappers import TimeLimit, Monitor

def one_hot(a, size):
    b = np.zeros((size))
    b[a] = 1
    return b

class FlattenObs(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        o = self.env.reset()
        obs = np.append(o['image'].flatten()/255., [one_hot(o['direction'], 4)])
        self.observation_space = gym.spaces.Box(0, 1, obs.shape)

    def reset(self, **kwargs):
        o = super().reset(**kwargs)
        obs = np.append(o['image'].flatten()/255., [one_hot(o['direction'], 4)])
        return obs

    def step(self, action):
        o, reward, done, info = super().step(action)
        obs = np.append(o['image'].flatten()/255., [one_hot(o['direction'], 4)])
        return obs, reward, done, info
    
env = gym.make("MiniGrid-MinimapForFalcon-v0")
env = FlattenObs(env)
# g = env.reset()
# i = env.render('rgb_array')
# Image.fromarray(i)

env = Monitor(env, f'videos', force=True)
env.action_space.seed(0)
env.reset()
for i in range(10000):
    # env.render()
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    if done:
        env.reset()
env.close()