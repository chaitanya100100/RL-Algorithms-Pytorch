
# coding: utf-8

# In[1]:


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import pickle

print("gym", gym.__version__)
print("pytorch", torch.__version__)


# In[2]:


""" 
    A small fully connected network to estimate action probabilities.
    Input : a 1D vector of state
    Output : Probability distribution over action space
"""
class PolicyNetworkFC(nn.Module):
    def __init__(self, num_inp, num_out):
        super(PolicyNetworkFC, self).__init__()
        self.fc1 = nn.Linear(num_inp, 64)
        self.fc2 = nn.Linear(64, num_out)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# In[3]:



def policy_gradient(env, 
                    policy, 
                    optimizer,
                    num_episodes=100,
                    gamma = 0.95,
                    render=False
                   ):
    """
        This function implements policy gradient algorithm.
        
        Returns:
            episodic_rewards : a list of episodic rewards
    """

    episodic_rewards = []
    
    # loop for episodes
    for episode in range(num_episodes):
        obs = env.reset()
        rewards = []
        log_probs = []
        
        # loop for steps
        for t in range(10000): # to avoid infinite loop
            
            # predict action probabilities
            inp = torch.from_numpy(obs).float().unsqueeze(0)
            action_prob = policy(inp)[0]
            
            # choose action and store log probability
            catg = Categorical(action_prob)
            action = catg.sample()
            log_probs.append(catg.log_prob(action))
            action = action.item()
            
            # take the chosen action and get the next state and reward
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if render:
                env.render()
            if done:
                break
        
        print("Episode {:4d} lasted for {:6d} steps".format(episode, t))
        episodic_rewards.append(t)
        
        # convert singular rewards to cumulative rewards
        for idx in range(len(rewards) - 2, -1, -1):            
            rewards[idx] = rewards[idx] + rewards[idx+1] * gamma
        
        
        # a common baseline as average of rewards
        # uncomment these lines for learning with baselines
       
        #rewards = torch.tensor(rewards)
        #rewards = rewards - rewards.mean()
        #rewards = rewards - 10
        
        loss = []
        for r, lp in zip(rewards, log_probs):
            loss.append(-r*lp)
        loss = torch.stack(loss).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return episodic_rewards
        


# In[4]:


# hyper params
num_episodes = 300
lr = 0.005
gamma = 0.99
render = False

all_runs = []

# 3 runs of training
for i in range(3):

    # setup gym environment, policy network and optimizer
    env = gym.make("CartPole-v0")
    policy = PolicyNetworkFC(env.observation_space.shape[0], env.action_space.shape[0])
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)


    # main learning function
    episodic_rewards = policy_gradient(env=env, 
                                    policy=policy, 
                                    optimizer=optimizer,
                                    num_episodes=num_episodes,
                                    gamma=gamma,
                                    render=render
                                   )

    # it's good practice to close the environment 
    env.close()
    
    all_runs.append(episodic_rewards)


# In[5]:


with open("PG.pkl", "wb") as f:
    pickle.dump(all_runs, f)


# In[6]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.plot(episodic_rewards)
plt.show()

