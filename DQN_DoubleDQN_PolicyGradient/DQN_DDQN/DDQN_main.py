import gym
import torch.optim as optim

from mymodel import QNet
from DDQN import OptimizerSpec, double_q_learning_fun
from utils import LinearSchedule
from atari_wrappers import wrap_deepmind
from gym import wrappers


BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 300000
LEARNING_STARTS = 5000
MAX_LEARNING_STEPS = 10000000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 1000
LEARNING_RATE = 0.0005
ALPHA = 0.95
EPS = 0.01

optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

exploration_schedule = LinearSchedule(2000000, 0.05)

env = gym.make("PongNoFrameskip-v4")
env = wrap_deepmind(env)
# it is necessary to wrap monitor last to check stopping criterion by total number of steps
env = wrappers.Monitor(env, "./ddqn/tmp", force=True)


double_q_learning_fun(
    env=env,
    q_func=QNet,
    optimizer_spec=optimizer_spec,
    exploration=exploration_schedule,
    replay_buffer_size=REPLAY_BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    learning_starts=LEARNING_STARTS,
    learning_freq=LEARNING_FREQ,
    max_learning_steps=MAX_LEARNING_STEPS,
    frame_history_len=FRAME_HISTORY_LEN,
    target_update_freq=TARGER_UPDATE_FREQ,
)
