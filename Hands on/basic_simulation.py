# Car Pole
import gym

env = gym.make('CartPole-v0')

env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

# Enviroments
from gym import envs

print(envs.registry.all())

# Car racing
import gym

# pip install gym[box2d]

env = gym.make('CarRacing-v0')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())