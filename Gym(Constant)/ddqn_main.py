import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Suppress Tensorflow Messages
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from dqn_agents import DDQNAgent, DDQNPlanningAgent, update_loss, reshape_state
from init_main import init_main
import wandb
from tqdm import tqdm
import argparse

import socket

os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_MODE"] = "dryrun"

parser = argparse.ArgumentParser()
# parser.add_argument("--algorithm", default='oblivious', help="algorithm")
# parser.add_argument("--algorithm", default='delayed', help="algorithm")
# parser.add_argument("--algorithm", default='delay', help="algorithm")
parser.add_argument("--algorithm", default='augmented', help="algorithm")
parser.add_argument("--delay", default=0, help="environment delay")
parser.add_argument("--seed", default=0, help="seed")
args = parser.parse_args()
algorithm = args.algorithm
delay = int(args.delay)
seed = int(args.seed)

AVERAGE_OVER_LAST_EP = 1
# EPISODES = 3500
SAVE_PATH = 'saved_agents'
EP_LEN_LIMIT = int(1e4)
EVAL_FREQ = 5


def init_episode(delayed_env, agent, augment_state, state_size):
    ep_reward = 0
    ep_reshaped_reward = 0
    state = delayed_env.reset()
    state = massage_state(state, augment_state, delayed_env, state_size)
    agent.clear_action_buffer()
    loss_dict = {}
    loss_count = 0
    ep_step = 0
    return ep_reward, ep_reshaped_reward, state, loss_dict, loss_count, ep_step

def routinely_save_agent(e, env_name):
    pass
    # agent_name = env_name + '_ddqn_delay.h5'
    # if e % 349 == 0:
    #     if not os.path.isdir(SAVE_PATH):
    #         os.makedirs(SAVE_PATH)
    #     agent_full_name = wandb.run.id + '_' + agent_name
    #     agent_path = os.path.join(SAVE_PATH, agent_full_name)
    #     agent.save(agent_path)
    #     print('saved agent to {}'.format(agent_path))


def agent_act(config, agent, state, delayed_env, eval=False):
    if config.agent_type == 'delayed':
        action = agent.act(state, pending_actions=delayed_env.get_pending_actions(), eval=eval)
    else:
        action = agent.act(state, eval)
    return action

def massage_state(state, augment_state, delayed_env, state_size):
    if augment_state:
        state = np.concatenate((state, delayed_env.get_pending_actions()))
    state = reshape_state(state, delayed_env.is_atari_env, state_size)
    return state

if __name__ == "__main__":
    config, delayed_env, state_size, action_size, done, batch_size = init_main(algorithm, delay, seed)

    score_vec = []
    # for non-atari (i.e. cartpole) env, run on CPU
    # if not delayed_env.is_atari_env:
    kwargs = {
        'action_size': action_size,
        'is_atari_env': delayed_env.is_atari_env,
        'delay_value': config.delay_value,
        'epsilon_min': config.epsilon_min,
        'epsilon_decay': config.epsilon_decay,
        'learning_rate': config.learning_rate,
        'epsilon': config.epsilon,
        'use_m_step_reward': config.use_m_step_reward,
        'use_latest_reward': config.use_latest_reward
    }
    # if not config.double_q:
    #     agent = DQNAgent(state_size=state_size, **kwargs)
    # else:
    augment_state = False
    # wandb.config.update({'augment_state': False}, allow_val_change=True)
    if config.agent_type == 'delayed':
        agent = DDQNPlanningAgent(seed=seed, state_size=state_size, env=delayed_env,
                                  use_learned_forward_model=config.use_learned_forward_model, **kwargs)
    else:
        if config.agent_type == 'augmented':
            # wandb.config.update({'augment_state': True}, allow_val_change=True)
            augment_state = True
            state_size += config.delay_value
            agent = DDQNAgent(seed=seed, state_size=state_size, **kwargs)
        elif config.agent_type == 'delay':
            agent = DDQNAgent(seed=seed, state_size=state_size, is_delayed_agent=True, **kwargs)
        # third option is 'oblivious'
        else:
            agent = DDQNAgent(seed=seed, state_size=state_size, **kwargs)

    episode = 0
    ep_reward, ep_reshaped_reward, state, loss_dict, loss_count, ep_step = init_episode(delayed_env, agent,
                                                                                        augment_state, state_size)
    total_steps_delay_dependent = int(1000000) # + config.delay_value * 10000)
    # eval_done = False
    for step_num in tqdm(range(total_steps_delay_dependent)):
        action = agent_act(config, agent, state, delayed_env, eval=False)
        next_state, reward, done, _ = delayed_env.step(action)
        ep_reward += reward
        if config.use_reward_shaping and not delayed_env.is_atari_env:
            reward = delayed_env.get_shaped_reward(next_state, reward)
        ep_reshaped_reward += reward
        next_state = massage_state(next_state, augment_state, delayed_env, state_size)
        can_memorize = ep_step > config.delay_value or not delayed_env.pretrained_agent_loaded
        if can_memorize: # otherwise, we're using expert samples initially which is unfair
            agent.memorize(state, action, reward, next_state, done)
        state = next_state
        if config.double_q and step_num % config.target_network_update_freq == 0:
            agent.update_target_model()
        if len(agent.memory) > batch_size and step_num % config.train_freq == 0:
            batch_loss_dict = agent.replay(batch_size, step_num)
            update_loss(loss_dict, batch_loss_dict)
            loss_count += 1
        ep_step += 1
        if done:
            routinely_save_agent(episode, config.env_name)
            wandb_dict = {'reward': ep_reward, 'ep_reshaped_reward': ep_reshaped_reward}
            if 'f_model_loss' in loss_dict:
                f_model_loss = loss_dict['f_model_loss'] / loss_count
                wandb_dict['f_model_loss'] = f_model_loss
            wandb.log(wandb_dict, step=step_num)
            score_vec.append(ep_reward)
            episode += 1
            ep_reward, ep_reshaped_reward, state, loss_dict, loss_count, ep_step = init_episode(delayed_env, agent, augment_state,
                                                                                       state_size)

    tot_ep_num = len(score_vec)
    avg_over = round(tot_ep_num * AVERAGE_OVER_LAST_EP)
    final_avg_score = np.mean(score_vec[-avg_over:])
    wandb.log({'final_score': final_avg_score})
