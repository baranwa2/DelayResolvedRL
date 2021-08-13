import tensorflow as tf
import numpy as np
import random
import copy
from statistics import mean
from collections import deque

GPUs = tf.config.experimental.list_physical_devices('GPU')

if GPUs:
    try:
        for gpu in GPUs:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def to_onehot(size, value):
    """1 hot encoding for observed state"""
    return np.eye(size)[value]


class Model(tf.keras.Model):
    """DQN Model"""
    def __init__(self, num_states, hidden_units, num_actions, alg, use_stochastic_delay, max_dimension):
        super(Model, self).__init__()
        if alg == 'IS':
            if use_stochastic_delay:
                self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states + 1 + max_dimension,))
            else:
                self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states + max_dimension,))
        else:
            self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, num_states, num_actions, model_params, alg_params):
        np.random.seed(alg_params['seed'])
        tf.random.set_seed(alg_params['seed'])
        random.seed(alg_params['seed'])
        self.num_actions = num_actions
        self.alg = alg_params['algorithm']
        self.batch_size = alg_params['batch_size']
        self.optimizer = tf.optimizers.Adam(alg_params['learning_rate'])
        self.use_stochastic_delay = alg_params['use_stochastic_delay']
        self.max_dimension = model_params['max_dimension']
        hidden_units = model_params['hidden_units']
        self.delay = alg_params['delay']
        self.gamma = alg_params['gamma']
        self.model = Model(num_states, hidden_units, num_actions, self.use_stochastic_delay, self.max_dimension,
                           self.alg)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = model_params['max_buffer_size']
        self.min_experiences = model_params['min_buffer_size']
        if self.alg != 'normal':
            self.action_buffer = deque(maxlen=self.max_dimension + 1)
            self.action_buffer_padded = deque(maxlen=self.max_dimension + 1)

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def fill_up_buffer(self):
        self.action_buffer_padded.clear()
        for _ in range(self.max_dimension):
            self.action_buffer_padded.append(0)

    def buffer_padding(self):
        current_length = len(self.action_buffer)
        self.action_buffer_padded = copy.deepcopy(self.action_buffer)
        for _ in range(0, self.max_dimension - current_length):
            self.action_buffer_padded.append(0)

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def play_game(global_step, env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    episode_step = 0
    last_state_observed = 0
    done = False
    observations = env.reset()
    observations_original = observations
    if env.game_name.startswith('Frozen'):
        observations = to_onehot(env.state_space.n, observations)
    if TrainNet.alg != 'normal':
        TrainNet.fill_up_buffer()
    losses = list()
    clear = False
    while not done:
        delay = env.delay
        len_buffer = len(env.state_buffer)
        if TrainNet.alg == 'normal':
            action = TrainNet.get_action(observations, epsilon)
            prev_observations = observations
            observations, reward, done = env.step(observations_original, action)
            observations_original = observations
            if env.game_name.startswith('Frozen'):
                observations = to_onehot(env.state_space.n, observations)
        else:
            if episode_step == 0:
                if env.use_stochastic_delay:
                    last_state_observed = (episode_step - env.turn_limit / 2) / env.turn_limit
                    action_state = np.append(last_state_observed, TrainNet.action_buffer_padded)
                    information_state = np.append(observations, action_state)
                    # information_state = np.append(observations, TrainNet.action_buffer_padded)
                else:
                    information_state = np.append(observations, TrainNet.action_buffer_padded)
            if TrainNet.alg == 'IS':
                action = TrainNet.get_action(information_state, epsilon)
            else:
                action = TrainNet.get_action(observations, epsilon)
            prev_observations = observations
            prev_information_state = information_state
            observations, reward, done = env.step(observations_original, action)
            observations_original = observations
            if env.game_name.startswith('Frozen'):
                observations = to_onehot(env.state_space.n, observations)
            episode_step += 1

            if env.train:
                last_state_observed = (episode_step - 1 - env.turn_limit / 2) / env.turn_limit
                TrainNet.action_buffer.append(action + 1)
                for i in range(len_buffer + 1 - delay):
                    TrainNet.action_buffer.popleft() - 1
                TrainNet.buffer_padding()
            else:
                # delayed_action = random.randint(0, TrainNet.num_actions)
                TrainNet.action_buffer.append(action + 1)
                TrainNet.buffer_padding()

            if env.delay == 0:
                delayed_action = action
            else:
                if not TrainNet.action_buffer:
                    delayed_action = random.randint(0, TrainNet.num_actions)
                else:
                    delayed_action = TrainNet.action_buffer[0]

            if delay == 0:
                delayed_action = action

            if len(TrainNet.action_buffer) == TrainNet.max_dimension + 1:
                TrainNet.action_buffer.clear()
                TrainNet.buffer_padding()
                observations = env.state_buffer.pop()
                env.state_buffer.clear()
                reward = np.sum(env.reward_buffer)
                done = env.done_buffer.pop()
                env.done_buffer.clear()
                env.reward_buffer.clear()
                clear = True

            if env.use_stochastic_delay:
                action_state = np.append(last_state_observed, TrainNet.action_buffer_padded)
                information_state = np.append(observations, action_state)
                # information_state = np.append(observations, TrainNet.action_buffer_padded)
            else:
                information_state = np.append(observations, TrainNet.action_buffer_padded)
        rewards += reward
        if done:
            episode_step = 0
            env.reset()
            if TrainNet.alg != 'normal':
                TrainNet.action_buffer.clear()
                TrainNet.buffer_padding()
        global_step += 1
        if TrainNet.alg == 'normal':
            exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        if TrainNet.alg == 'delay':
            exp = {'s': prev_observations, 'a': delayed_action, 'r': reward, 's2': observations, 'done': done}
        if TrainNet.alg == 'IS':
            exp = {'s': prev_information_state, 'a': action, 'r': reward, 's2': information_state, 'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        if global_step % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return global_step, rewards, mean(losses)


def test(env, TrainNet, logs, num_episodes):
    for _ in range(num_episodes):
        observation = env.reset()
        rewards = 0
        steps = 0
        done = False
        while not done:
            action = TrainNet.get_action(observation, 0)
            observation, reward, done, _ = env.step(action)
            steps += 1
            rewards += reward
        with open(logs['log_file_name'], "a") as f:
            print("Testing steps: {} rewards :{} ".format(steps, rewards), file=f)
        print("Testing steps: {} rewards :{} ".format(steps, rewards))


def train_agent(env, num_frames, model_params, algorithm_params, logs, verbose):
    num_actions = env.number_of_actions
    try:
        state_space = len(env.state_space.sample())
    except TypeError:
        state_space = env.state_space.n

    copy_step = model_params['copy_step']
    TrainNet = DQN(state_space, num_actions, model_params, algorithm_params)
    TargetNet = DQN(state_space, num_actions, model_params, algorithm_params)
    # N = num_episodes
    total_rewards_list = []
    total_losses_list = []
    epsilon_start = algorithm_params['start_epsilon']
    decay = algorithm_params['epsilon_decay']
    min_epsilon = algorithm_params['stop_epsilon']
    global_step = 1
    n = 0
    while True:
        epsilon = min_epsilon + (epsilon_start - min_epsilon) * np.exp(-decay * global_step)
        global_step, total_reward, losses = play_game(global_step, env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards_list.append(total_reward)
        total_losses_list.append(losses)
        total_rewards = np.array(total_rewards_list)
        total_losses = np.array(total_losses_list)
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        avg_losses = total_losses[max(0, n - 100):(n + 1)].mean()
        if n % logs['log_interval'] == 0:
            if verbose:
                with open(logs['log_file_name'], "a") as f:
                    print("episode:{}, eps:{:.3f}, avg reward (last 100):{:.2f}, avg loss:{:.2f}"
                          .format(n, epsilon, avg_rewards, avg_losses), file=f)
            if not verbose:
                print("episode:{}, eps:{:.3f}, avg reward (last 100):{:.2f}"
                      .format(n, epsilon, avg_rewards))
        # test(env, TrainNet, logs, 100)
        n += 1
        if global_step > num_frames:
            break
    env.close()
    return total_rewards, total_losses
