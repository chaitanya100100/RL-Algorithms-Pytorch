
# coding: utf-8

# In[1]:


import numpy as np
import random
from PIL import Image
import gym
import itertools
from gym import wrappers
from gym import spaces


# In[2]:


gamma = 0.99
learning_rate = 0.001
explore_till = 500000
eps = 0.1


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
    return x_t.astype(np.float) / 255.0
    #return np.append(x_t, 1).astype(np.float) / 255.0

# pong specific preprocess
def preprocess2(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    #return np.append(I.astype(np.float).ravel(), 1.0)
    return I.astype(np.float).ravel()

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


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


# In[4]:


env = gym.make("Pong-v0")
env = ProcessFrame84(env)
# env = wrappers.Monitor(env, "./tmp_qlearning", force=True)

eps_scheduler = LinearSchedule(explore_till, eps)

obs = env.reset()
rewards = []
num_episodes = 0
W = np.zeros([env.action_space.n, env.observation_space.shape[0]], dtype=np.float)

episodic_rewards = []

for t in itertools.count():
    
    # estimate q values
    q_val = np.matmul(W, obs)
    
    # eps greedy step
    if random.random() > eps_scheduler.value(t):
        action = np.argmax(q_val)
    else:
        action = np.random.randint(env.action_space.n)
        
    next_obs, reward, done, _ = env.step(action)
    rewards.append(reward)
    
    
    # target q values
    if done:
        target = reward
    else:
        next_q_val = np.matmul(W, next_obs)
        target = reward + gamma * np.max(next_q_val)
    
    # update equation
    W[action] = W[action] + learning_rate * (target - q_val[action]) * obs
    
    obs = next_obs

    # reset the env if done
    if done:
        obs = env.reset()
        print("Episode {:4d}\tReward : {:6.2f}\tSteps : {:4d}\tEpsilon : {:6.6f}".format(
            num_episodes, sum(rewards), len(rewards), eps_scheduler.value(t)))
        num_episodes += 1
        episodic_rewards.append(sum(rewards))
        rewards = []

    if len(episodic_rewards) == 1000:
        break
        
env.close()


# In[5]:


# -------------------------
# Plot the results
# -------------------------

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pickle
with open("QL_without_replay.pkl", "wb") as f:
    pickle.dump(episodic_rewards, f)


episodic_rewards = np.array(episodic_rewards)
K = 10

plt.plot(episodic_rewards)
plt.plot(np.convolve(episodic_rewards, np.ones(K)/K, mode='valid'))
plt.title("Pong Linear Q-Learning Without Replay Buffer")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.show()

