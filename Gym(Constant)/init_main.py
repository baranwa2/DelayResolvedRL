import gym
from delayed_env import DelayedEnv
import wandb

"""Code adapted from https://openreview.net/forum?id=j1RMMKeP2gR"""
'''HyperParameters'''
# MountainCar-v0
# Number of Runs:10 \\
# Number of Frames: 1 Million \\
# Batch Size: 32 \\
# $\gamma$: 0.99 \\
# Learning Rate: 1e-3 \\
# Learning Rate: 1e-3 \\
# $\epsilon$-Start: 1.0 \\
# $\epsilon$-Stop: 1e-4 \\
# $\epsilon$-Decay: 1e-4 \\
# Hidden Units: [200, 200] (Action delays)\\
# Hidden Units: [200]  (Observation delays)\\
# Forward Model: [200, 200] \\
# Replay Buffer Size: 1000 \\
# Target Network Frequency Update: 300 \\

# Acrobot-v1
# Number of Runs:10 \\
# Number of Frames: 1 Million \\
# Batch Size: 32 \\
# $\gamma$: 0.99 \\
# Learning Rate: 1e-3 \\
# $\epsilon$-Start: 1.0 \\
# $\epsilon$-Stop: 1e-4 \\
# $\epsilon$-Decay: 1e-4 \\
# Hidden Units: [200] \\
# Forward Model: [200] \\
# Replay Buffer Size: 1000 \\
# Target Network Frequency Update: 25 \\

# Reshaped CartPole-v0
# Number of Runs:10 \\
# Number of Frames: 1 Million \\
# Batch Size: 32 \\
# $\gamma$: 0.99 \\
# Learning Rate: 1e-3 \\
# $\epsilon$-Start: 1.0 \\
# $\epsilon$-Stop: 1e-4 \\
# $\epsilon$-Decay: 1e-4 \\
# Hidden Units: [200] \\
# Forward Model: [24, 24] \\
# Replay Buffer Size: 1000 \\
# Target Network Frequency Update: 300 \\

# CartPole-v0
# Number of Runs:10 \\
# Number of Frames: 1 Million \\
# Batch Size: 32 \\
# $\gamma$: 0.99 \\
# Learning Rate: 1e-3 \\
# $\epsilon$-Start: 1.0 \\
# $\epsilon$-Stop: 1e-4 \\
# $\epsilon$-Decay: 1e-4 \\
# Hidden Units: [200] \\
# Forward Model: [200] \\
# Replay Buffer Size: 1000 \\
# Target Network Frequency Update: 25 \\


def init_main(algorithm, delay, seed):
    hyperparameter_defaults = dict(
        is_delayed_agent=False,
        double_q=True,
        delay_value=delay,
        epsilon_decay=1e-4,
        epsilon_min=0.001, #0.001
        learning_rate=0.001, #0.005, #mountainCar: 0.0001
        seed=seed,
        epsilon=1.0,
        use_m_step_reward=False,
        use_latest_reward=False,
        use_reward_shaping=False,
        physical_noise_std_ratio=0.0, #0.1
        env_name='CartPole-v0', #'CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'
        train_freq=1,
        target_network_update_freq=300,
        use_learned_forward_model=True,
        agent_type=algorithm, # 'augmented' (DRDQN), 'delayed' (https://openreview.net/forum?id=j1RMMKeP2gR), 'oblivious' (DQN), 'delay' (https://ieeexplore.ieee.org/document/5650345)
        total_steps=3000,
    )
    # Pass your defaults to wandb.init
    wandb.init(project='delay-rl', name='DQN_{}_delay_{}_seed_{}'.format(algorithm, delay, seed),
                         config=hyperparameter_defaults)
    config = wandb.config
    if 'CartPole' in config.env_name or 'Acrobot' in config.env_name:
        orig_env = gym.make(config.env_name)
        orig_env.seed(seed)
    else:
        orig_env = gym.make(config.env_name)
        orig_env.seed(seed)
    # orig_env = DiscretizeActions(orig_env) # for mujoco envs
    delayed_env = DelayedEnv(orig_env, config.delay_value)
    state_size = orig_env.observation_space.shape
    if not delayed_env.is_atari_env:
        state_size = state_size[0]
    action_size = orig_env.action_space.n
    done = False
    batch_size = 32
    return config, delayed_env, state_size, action_size, done, batch_size





