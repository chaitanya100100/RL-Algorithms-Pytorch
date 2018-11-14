
# coding: utf-8

# In[1]:


import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import itertools
import pickle

print("gym", gym.__version__)
print("pytorch", torch.__version__)

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(device)


# In[2]:


""" 
    A small fully connected network to estimate action probabilities.
    Input : a 1D vector of state
    Output : Probability distribution over action space
"""
class PolicyNetworkFC(nn.Module):
    def __init__(self, num_inp, num_out):
        super(PolicyNetworkFC, self).__init__()
        self.fc1 = nn.Linear(num_inp, 96)
        self.fc2 = nn.Linear(96, num_out)
    
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
        self.fc1 = nn.Linear(num_inp, 96)
        self.fc2 = nn.Linear(96, 1)
    
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
    obs = env.reset()
    episode_rew = 0
    nums = np.zeros(env.action_space.n)

    
    # loop for episodes
    while len(episodic_rewards) < num_episodes:
        rewards = []
        log_probs = []
        state_values = []
        
        # loop for steps
        for t in itertools.count(): # to avoid infinite loop
            
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
            nums[action] += 1
            
            # take the chosen action and get the next state and reward
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            episode_rew += reward
            
            if render:
                env.render()
            
            # make end of life as end of episode, it helps to learn better (src : DQN paper)
            if done or abs(reward) > 0.01:
                break
        
        if done:
            print("Episode {:4d}\tReward {:6.2f}".format(len(episodic_rewards), episode_rew))
            print(nums)
            obs = env.reset()
            nums = np.zeros(env.action_space.n)
            episodic_rewards.append(episode_rew)
            episode_rew = 0
        
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
        


# In[ ]:


# atari specific preprocess
def preprocess(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    img = Image.fromarray(img)
    resized_screen = img.resize((84, 110), Image.BILINEAR)
    resized_screen = np.array(resized_screen)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84*84])
    return x_t.astype(np.float32) / 255.0
    #return np.append(x_t, 1).astype(np.float32) / 255.0

# pong specific preprocess
def preprocess2(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    #return np.append(I.astype(np.float32).ravel(), 1.0)
    return I.astype(np.float32).ravel()

class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=1, shape=(80*80))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return preprocess2(obs), reward, done, info

    def _reset(self):
        return preprocess2(self.env.reset())


# In[ ]:


# hyper params
num_episodes = 20000
lr = 0.0001
gamma = 0.99
render = False



# setup gym environment, policy network and optimizer
env = gym.make("Pong-v0")
env = ProcessFrame84(env)

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


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pickle
with open("Pong_PG_nonlinear_with_baseline.pkl", "wb") as f:
    pickle.dump(episodic_rewards, f)


episodic_rewards = np.array(episodic_rewards)
K = 50

plt.plot(episodic_rewards)
plt.plot(np.convolve(episodic_rewards, np.ones(K)/K, mode='valid'))
plt.title("Pong Nonlinear Policy Gradient with Value function Baseline")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.show()

