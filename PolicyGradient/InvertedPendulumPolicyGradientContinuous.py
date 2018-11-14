
# coding: utf-8

# In[1]:


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
import pickle

print("gym", gym.__version__)
print("pytorch", torch.__version__)


# In[2]:


""" 
    A small fully connected network to estimate action mean.
    Input : a 1D vector of state
    Output : mu and sigma for action space
"""
class PolicyNetworkFC(nn.Module):
    def __init__(self, num_inp, num_out):
        super(PolicyNetworkFC, self).__init__()
        self.fc1 = nn.Linear(num_inp, 64)
        self.fc2 = nn.Linear(64, num_out)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
            action_mu_sig = policy(inp)[0]
            
            # choose action and store log probability
            anor = Normal(action_mu_sig[0], action_mu_sig[1])
            #anor = Normal(action_mu_sig[0], torch.tensor(0.05))
            action = anor.sample()
            log_probs.append(anor.log_prob(action))
        
            # take the chosen action and get the next state and reward
            obs, reward, done, info = env.step(action.item())
            rewards.append(reward)
            if render:
                env.render()
            if done:
                break
        
        print("Episode {:4d} lasted for {:6.2f} steps".format(episode, sum(rewards) ) )
        episodic_rewards.append(t)

        # convert singular rewards to cumulative rewards
        for idx in range(len(rewards) - 2, -1, -1):            
            rewards[idx] = rewards[idx] + rewards[idx+1] * gamma
        
        
        # a common baseline as average of rewards
        # uncomment these lines for learning with baselines
        #rewards = torch.tensor(rewards)
        #rewards = rewards - rewards.mean()
        #rewards = rewards - 100
        
        # some post-processing on rewards (optional)
        #rewards = torch.tensor(rewards)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
        
        loss = []
        for r, lp in zip(rewards, log_probs):
            loss.append(-r*lp)
        loss = torch.stack(loss).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return episodic_rewards
        


# In[7]:


# hyper params
num_episodes = 5000
lr = 0.0005
gamma = 0.99
render = False

# setup gym environment, policy network and optimizer
env = gym.make("InvertedPendulum-v2")
policy = PolicyNetworkFC(env.observation_space.shape[0], 2*env.action_space.shape[0])
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


# In[9]:


with open("Cont_PG.pkl", "wb") as f:
    pickle.dump(episodic_rewards, f)


# In[8]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.plot(episodic_rewards)
plt.show()

