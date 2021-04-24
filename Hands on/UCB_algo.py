import numpy as np
import gym_bandits
import gym
import math
import random

env = gym.make("BanditTenArmedGaussian-v0")

# Number of rounds (iterations)
num_rounds = 20000

# Count of number of times an arm was pulled
count = np.zeros(10)

# Sum of rewards of each arm
sum_rewards = np.zeros(10)

# Q value which is the average reward
Q = np.zeros(10)

def UCB(iters):
    
    ucb = np.zeros(10)

    # explore all the arms
    if iters < 10:
        return i
    else:
        for arm in range(10):
            # calculate upper bound
            upper_bound = math.sqrt((2 * math.log(sum(count))) / count[arm])

            # add upper bound to the Q value
            ucb[arm] = Q[arm] + upper_bound
        
        # return the arm which has maximum value
        return (np.argmax(ucb))

for i in range(num_rounds):
    
    # Select the arm using epsilon greedy
    arm = UCB(i)

    # Get the reward
    observation, reward, done, info = env.step(arm)

    # Update the count of that arm
    count[arm] += 1

    # Sum the rewards obtained from the arm
    sum_rewards[arm] += reward
    
    # Calculate Q value which is the average rewards of the arm
    Q[arm] = sum_rewards[arm] / count[arm]

print("The optimal arm is {}".format(np.argmax(Q)))