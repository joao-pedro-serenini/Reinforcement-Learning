import gym
import random
env = gym.make("Taxi-v3")

alpha = 0.85
gamma = 0.90
epsilon = 0.8

# Initialize Q table as dictionary for storing the state-action values
Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s, a)] = 0.0
        
# Now, we define a function called epsilon_greedy for performing action according epsilon greedy policy
def epsilon_greedy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q[(state, x)])

for i in range(4000):
    # Store cumulative reward of each episodes
    r = 0
    
    # For every interation initialize the state
    state = env.reset()

    # Pick up the action using epsilon greedy policy
    action = epsilon_greedy(state, epsilon)

    while True:
        env.render()

        # Perform the action in the state and move to the next state
        nextstate, reward, done, _ = env.step(action)

        # Pick up the next action usgin epsilon greedy policy
        nextaction = epsilon_greedy(nextstate, epsilon)

        # Calculate Q value of the previous state using our update rule
        Q[(state, action)] += alpha * (reward + gamma * Q[(nextstate, nextaction)] - Q[(state, action)])

        # Update our state and action with next action and next state
        action = nextaction
        state = nextstate
        r += reward

        # Break if we are at the terminal state of the episode
        if done:
            break
    
    print("total reward: ", r)

env.close()