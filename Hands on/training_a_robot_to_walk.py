import gym

env = gym.make('BipedalWalker-v2')

for episode in range(100):
    observation = env.reset()
    for i in range(10000):
        env.render()
        
        print(observation)
        
        action = env.action_space.sample()

        observation, reward, done, infor = env.step(action)

        if done:
            print("{} timesteps taken for the Episode".format(i+1))
            break