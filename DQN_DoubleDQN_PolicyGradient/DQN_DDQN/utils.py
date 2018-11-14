import numpy as np
import random

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
    
def sample_n_unique(sampling_f, n):
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBufferSingle(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = 0
        self.num = 0
        self.action = np.empty([self.capacity,], dtype=np.int)
        self.reward = np.empty([self.capacity,], dtype=np.float)
        self.obs = None
        self.done = np.empty([self.capacity,], dtype=np.bool)
        
        
    def store(self, obs, action, reward, done):
        if not self.obs:
            self.obs = np.empty([self.capacity]+list(obs.shape), dtype=np.int)
        self.obs[self.idx] = obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.done[self.idx] = done
        
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
        batch_done = self.done[cand].astype(np.float)
        batch_next_obs = self.obs[cand+1]
        
        return batch_obs, batch_action, batch_reward, batch_done, batch_next_obs


class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def _encode_sample(self, idxes):
        obs_batch = np.concatenate([self._encode_observation(idx)[np.newaxis, :] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[np.newaxis, :] for idx in idxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        assert(batch_size + 1 <= self.num_in_buffer)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len

        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0)
        else:
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def store_frame(self, frame):

        # transpose image frame into (img_c, img_h, img_w)
        frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)

        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done 
        
        