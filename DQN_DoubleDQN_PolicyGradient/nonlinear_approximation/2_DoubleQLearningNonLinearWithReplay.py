
# coding: utf-8

# In[1]:


import numpy as np
import random
from PIL import Image
import gym
import itertools
from gym import wrappers
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)
print(gym.__version__)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

print(device)


# In[2]:


gamma = 0.99
learning_rate = 0.0005
explore_till = 500000
eps = 0.1

replay_buffer_size = 100000
learning_starts = 500
batch_size = 8


# In[3]:


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

# wrapper for preprocess
class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=1, shape=(80*80))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return preprocess2(obs), reward, done, info

    def _reset(self):
        return preprocess2(self.env.reset())

# replay buffer class
class ReplayBufferSingle(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = 0
        self.num = 0
        self.action = np.empty([self.capacity,], dtype=np.int)
        self.reward = np.empty([self.capacity,], dtype=np.float32)
        self.obs = None
        self.next_obs = None
        self.done = np.empty([self.capacity,], dtype=np.bool)
        
        
    def store(self, obs, action, reward, done, next_obs):
        if self.obs is None:
            self.obs = np.empty([self.capacity]+list(obs.shape), dtype=np.float32)
            self.next_obs = np.empty([self.capacity]+list(obs.shape), dtype=np.float32)
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.done[self.idx] = done
        
        self.num = min(self.capacity, self.num+1)
        self.idx += 1
        if self.idx == self.capacity:
            self.idx = 0

    def sample(self, batch_size):
        cand = []
        while len(cand) < batch_size:
            x = random.randint(0, self.num - 2)
            if x not in cand:
                cand.append(x)
                
        cand = np.array(cand)
        batch_obs = self.obs[cand]
        batch_action = self.action[cand]
        batch_reward = self.reward[cand]
        batch_done = self.done[cand].astype(np.float32)
        batch_next_obs = self.next_obs[cand]
        
        #return batch_obs[0], batch_action[0], batch_reward[0], batch_done[0], batch_next_obs[0]
        return batch_obs, batch_action, batch_reward, batch_done, batch_next_obs


# linear scheduler for epsilon
class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


# In[4]:


class QNet(nn.Module):
    def __init__(self, num_inp, num_out):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(num_inp, 96)
        self.fc2 = nn.Linear(96, 32)
        self.fc3 = nn.Linear(32, num_out)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[5]:


env = gym.make("Pong-v0")
env = ProcessFrame84(env)
env = wrappers.Monitor(env, "./tmp_doubleqlearning_nonlinear_replay", force=True)

eps_scheduler = LinearSchedule(explore_till, eps)
replay_buffer = ReplayBufferSingle(capacity=replay_buffer_size)

# two q networks and their optimizers
qnet1 = QNet(env.observation_space.shape[0], env.action_space.n).to(device)
qnet2 = QNet(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer1 = torch.optim.SGD(qnet1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.SGD(qnet2.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

obs = env.reset()
rewards = []
num_episodes = 0
episodic_rewards = []

for t in itertools.count():

    # choose action using both q networks with eps greedy policy
    q_val = qnet1(torch.from_numpy(obs).type(torch.float).unsqueeze(0).to(device))[0] +             qnet2(torch.from_numpy(obs).type(torch.float).unsqueeze(0).to(device))[0]
    if random.random() > eps_scheduler.value(t):
        action = q_val.argmax().item()
    else:
        action = np.random.randint(env.action_space.n)
        
    next_obs, reward, done, _ = env.step(action)
    rewards.append(reward)
    
    # store the transition in replay buffer
    replay_buffer.store(obs, action, reward, done, next_obs)
    obs = next_obs
    
    # reset env if done
    if done:
        obs = env.reset()
        print("Episode {:4d}\tReward : {:6.2f}\tSteps : {:4d}\tEpsilon : {:6.6f}".format(
            num_episodes, sum(rewards), len(rewards), eps_scheduler.value(t)))
        num_episodes += 1
        episodic_rewards.append(sum(rewards))
        rewards = []

    if len(episodic_rewards) == 1000:
        break

    if t < learning_starts:
        continue
    
    # sample a batch from replay buffer to update
    b_obs, b_action, b_reward, b_done, b_next_obs = replay_buffer.sample(batch_size)
    
    b_obs = torch.tensor(b_obs).to(device)
    b_action = torch.tensor(b_action).to(device)
    b_reward = torch.tensor(b_reward).to(device)
    b_done = torch.tensor(b_done).to(device)
    b_next_obs = torch.tensor(b_next_obs).to(device)
    
    # we update each q network with 0.5 probability
    if random.random() > 0.5:
        q1_val = qnet1(b_obs)
        next_q2_val = qnet2(b_next_obs)
        target = (b_reward + gamma * next_q2_val.max() * (1-b_done)).detach()
        loss = criterion(target, q1_val.gather(1, b_action.unsqueeze(1)).squeeze())

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        
    else:
        q2_val = qnet2(b_obs)
        next_q1_val = qnet1(b_next_obs)
        target = (b_reward + gamma * next_q1_val.max() * (1-b_done) ).detach()
        loss = criterion(target, q2_val.gather(1, b_action.unsqueeze(1)).squeeze())

        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

            
env.close()


# In[6]:


# -------------------------
# Plot the results
# -------------------------

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pickle
with open("DoubleQL_nonlinear_with_replay.pkl", "wb") as f:
    pickle.dump(episodic_rewards, f)


episodic_rewards = np.array(episodic_rewards)
K = 10

plt.plot(episodic_rewards)
plt.plot(np.convolve(episodic_rewards, np.ones(K)/K, mode='valid'))
plt.title("Pong Nonlinear Double Q-Learning with Replay Buffer")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.show()

