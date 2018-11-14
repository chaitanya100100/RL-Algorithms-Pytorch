import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import os

from utils import ReplayBuffer


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": [],
}

def q_learning_fun(
    env,
    q_func,
    optimizer_spec,
    exploration,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    max_learning_steps=1000000,
    frame_history_len=4,
    target_update_freq=10000
    ):



    if not os.path.isdir("./dqn"):
        os.mkdir("./dqn")

    img_h, img_w, img_c = env.observation_space.shape
    input_arg = frame_history_len * img_c
    num_actions = env.action_space.n

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            with torch.no_grad():
                ret = model(obs).data.max(1)[1].cpu()
                return ret
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    # Initialize target q function and q function
    Q = q_func(input_arg, num_actions).type(dtype)
    target_Q = q_func(input_arg, num_actions).type(dtype)
    

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    save_best_mean_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    SAVE_EVERY_N_STEPS = 2000000

    for t in count():
        ### Check stopping criterion
        if env.get_total_steps() >= max_learning_steps:
            break

        last_idx = replay_buffer.store_frame(last_obs)
        recent_observations = replay_buffer.encode_recent_observation()

        # Choose random action if not learning
        if t > learning_starts:
            action = select_epilson_greedy_action(Q, recent_observations, t)[0]
        else:
            action = random.randrange(num_actions)
        obs, reward, done, _ = env.step(action)
        # clip rewards between -1 and 1
        reward = max(-1.0, min(reward, 1.0))
        # Store other info in replay memory
        replay_buffer.store_effect(last_idx, action, reward, done)
        # Resets the environment when reaching an episode boundary.
        if done:
            obs = env.reset()
        last_obs = obs

        if (t > learning_starts and
                t % learning_freq == 0):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
            # Convert numpy nd_array to torch variables for calculation
            obs_batch = torch.from_numpy(obs_batch).type(dtype) / 255.0
            act_batch = torch.from_numpy(act_batch).long()
            rew_batch = torch.from_numpy(rew_batch)
            next_obs_batch = torch.from_numpy(next_obs_batch).type(dtype) / 255.0
            not_done_mask = torch.from_numpy(1 - done_mask).type(dtype)

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()

            # update rule
            cur_all_Q_values = Q(obs_batch)
            cur_act_Q_values = cur_all_Q_values.gather(1, act_batch.unsqueeze(1)).squeeze()
            next_all_target_Q_values = target_Q(next_obs_batch).detach()
            next_max_target_Q_values = next_all_target_Q_values.max(1)[0]
            
            next_max_target_Q_values = not_done_mask * next_max_target_Q_values
            
            target = rew_batch + (gamma * next_max_target_Q_values)
            error = target - cur_act_Q_values


            clipped_error = error.clamp(-1, 1)

            d_error = clipped_error * -1.0

            optimizer.zero_grad()

            cur_act_Q_values.backward(d_error.data)


            optimizer.step()
            num_param_updates += 1

            # Periodically update the target network
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())

        # Log progress and save of statistics
        episode_rewards = env.get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        Statistic["mean_episode_rewards"].append(mean_episode_reward)
        Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            # Dump statistics to pickle
            with open('./dqn/statistics.pkl', 'wb') as f:
                pickle.dump(Statistic, f)
                print("Saved to %s" % './dqn/statistics.pkl')

            if save_best_mean_reward < best_mean_episode_reward:
                save_best_mean_reward = best_mean_episode_reward
                torch.save(Q.state_dict(), './dqn/best_model.pth')


        if t % SAVE_EVERY_N_STEPS == 0:
            torch.save(Q.state_dict(), './dqn/n_steps_%d.pth' % t)
