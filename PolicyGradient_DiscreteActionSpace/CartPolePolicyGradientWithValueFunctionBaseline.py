
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


""" 
    A small fully connected network to estimate value function
    Input : a 1D vector of state
    Output : a single value denoting state value
"""
class ValueFunctionNetworkFC(nn.Module):
    def __init__(self, num_inp):
        super(ValueFunctionNetworkFC, self).__init__()
        self.fc1 = nn.Linear(num_inp, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[4]:



def policy_gradient(env, 
                    policy_estimator,
                    value_estimator,
                    policy_optimizer,
                    value_optimizer,
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
        state_values = []
        
        # loop for steps
        for t in range(10000): # to avoid infinite loop
            
            # predict action probabilities
            inp = torch.from_numpy(obs).float().unsqueeze(0)
            action_prob = policy_estimator(inp)[0]
            
            # predict and store state value function
            vs = value_estimator(inp)[0]
            state_values.append(vs)
            
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
        
        # scaling of rewards to have mean zero and std 1
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
                
        policy_loss = []
        value_loss = []
        for r, lp, vs in zip(rewards, log_probs, state_values):
            r = r - vs.item()
            policy_loss.append(-r*lp)
            #value_loss.append(F.smooth_l1_loss(vs, torch.tensor([r])))
            value_loss.append((vs - torch.tensor([r]))**2)
            
        policy_loss = torch.stack(policy_loss).sum()
        value_loss = torch.stack(value_loss).sum()
        
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        policy_optimizer.step()
        value_optimizer.step()
        
    return episodic_rewards
        


# In[5]:


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
    policy_estimator = PolicyNetworkFC(env.observation_space.shape[0], env.action_space.shape[0])
    value_estimator = ValueFunctionNetworkFC(env.observation_space.shape[0])
    policy_optimizer = torch.optim.Adam(policy_estimator.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_estimator.parameters(), lr=lr)

    # main learning function
    episodic_rewards = policy_gradient(env=env, 
                                        policy_estimator=policy_estimator, 
                                        value_estimator=value_estimator, 
                                        policy_optimizer=policy_optimizer,
                                        value_optimizer=value_optimizer,
                                        num_episodes=num_episodes,
                                        gamma=gamma,
                                        render=render
                                       )


    # it's good practice to close the environment 
    env.close()
    
    all_runs.append(episodic_rewards)


# In[6]:


with open("PG_value_bl_scaled.pkl", "wb") as f:
    pickle.dump(all_runs, f)


# In[7]:


# %matplotlib inline
# import matplotlib.pyplot as plt
# plt.plot(episodic_rewards)
# plt.show()

