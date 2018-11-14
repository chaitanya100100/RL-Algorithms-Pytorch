
# coding: utf-8

# In[1]:


import numpy as np
import pickle as pkl


# In[9]:


x = pkl.load(open("./ddqn/statistics.pkl", "rb"))


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

mean_episode_rewards = np.array(x['mean_episode_rewards'])
plt.plot(mean_episode_rewards)
plt.title("Double DQN trained for 10^7 frames")
plt.xlabel("Frames")
plt.ylabel("Episodic Rewards")
plt.savefig("ddqn.png")
plt.show()


# In[4]:


x = pkl.load(open("./dqn_6/statistics.pkl", "rb"))


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

mean_episode_rewards = np.array(x['mean_episode_rewards'])
plt.plot(mean_episode_rewards)
plt.title("DQN trained for 10^6 frames")
plt.xlabel("Frames")
plt.ylabel("Episodic Rewards")
plt.savefig("dqn_6.png")
plt.show()


# In[6]:


x = pkl.load(open("./dqn/statistics.pkl", "rb"))


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

mean_episode_rewards = np.array(x['mean_episode_rewards'])
plt.plot(mean_episode_rewards)
plt.title("DQN trained for 10^7 frames")
plt.xlabel("Frames")
plt.ylabel("Episodic Rewards")
plt.savefig("dqn.png")
plt.show()

