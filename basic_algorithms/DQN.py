import numpy as np
import os
import gym
import torch
import torch.nn as nn
import collections
import copy
import random

# hype-params
learn_freq = 5 #经验池攒一些经验再开启训练
buffer_size = 20000 #经验池大小
buffer_init_size = 200 #开启训练最低经验条数
batch_size = 32 #每次sample的数量
learning_rate = 0.001 #学习率
GAMMA = 0.99 # reward折扣因子

class Model(nn.Module):
    def __init__(self, act_dim, state_dim):
        super(Model, self).__init__()
        hidden1_size = 128
        hidden2_size = 128
        self.input_layer = nn.Linear(state_dim, hidden1_size)
        self.input_layer.weight.data.normal_(0, 0.1)
        self.hidden_layer = nn.Linear(hidden1_size, hidden2_size)
        self.hidden_layer.weight.data.normal_(0, 0.1)
        self.output_layer = nn.Linear(hidden2_size, act_dim)
        self.output_layer.weight.data.normal_(0, 0.1)

    def forward(self, state):
        h1 = nn.functional.relu(self.input_layer(state))
        h2 = nn.functional.relu(self.hidden_layer(h1))
        Q = self.output_layer(h2)
        return Q


class DQN:
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma

    def predict(self, state):
        return self.model.forward(state)  # shape: batch_size x act_dim

    def learn(self, state, action, reward, state_next, done):  # shape: batch_size x 1
        # 根据target网络求target Q
        next_values = self.target_model.forward(state_next).detach() # 阻断target梯度, shape: batch_size x act_dim
        target_value = reward + (1.0 - done)*self.gamma*next_values.max(1)[0]  # shape: batch_size x 1
        # 根据当前网络获取Q(s, a)
        curr_value = self.model.forward(state)
        action = action.unsqueeze(1)
        pred_value = torch.gather(curr_value, 1, action.long())  # batch_size x act_dim中以第二维取action对应的Q值成为batch_size x 1

        cost = self.loss(pred_value, target_value)
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()
        return cost

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict()) # 更新target网络参数


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, state_netx_batch, done_batch = [], [], [], [], []

        for exp in batch:
            s, a, r, s_next, done = exp
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            state_netx_batch.append(s_next)
            done_batch.append(done)
        return torch.from_numpy(np.array(state_batch).astype('float32')), \
               torch.from_numpy(np.array(action_batch).astype('int32')), \
               torch.from_numpy(np.array(reward_batch).astype('float32')), \
               torch.from_numpy(np.array(state_netx_batch).astype('float32')), \
               torch.from_numpy(np.array(done_batch).astype('float32'))

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, algorithm, state_dim, act_dim, epsilon=0.1, epsilon_fade=0.0):
        self.dqn = algorithm
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.steps = 0
        self.update_target_steps = 200
        self.epsilon = epsilon
        self.epsilon_fade = epsilon_fade

    def explore(self, state):
        sample = np.random.rand()
        if sample < self.epsilon:
            action = np.random.randint(self.act_dim)
        else:
            action = self.greedy(state)
        self.epsilon = max(0.01, self.epsilon - self.epsilon_fade)
        return action

    def greedy(self, state):
        state = torch.from_numpy(state)
        state = torch.tensor(state, dtype=torch.float32)
        pred_value = self.dqn.target_model.forward(state)
        values = pred_value.detach().numpy()
        values = np.squeeze(values, axis=None)
        action = np.argmax(values)  # 选择值最大的下标
        return action

    def learn(self, state, action, reward, state_next, done):
        if self.steps % self.update_target_steps == 0:
            self.dqn.update_target()
        self.steps += 1
        cost = self.dqn.learn(state, action, reward, state_next, done)
        return cost


def evaluate(env, agent, render=True):
    eval_reward = []
    for i in range(10):
        state = env.reset()
        episode_reward = 0
        while True:
            action = agent.greedy(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]

    exp_buffer = ReplayMemory(buffer_size)

    model = Model(act_dim=action_dim, state_dim=state_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=learning_rate)
    agent = Agent(algorithm, state_dim=state_dim, act_dim=action_dim, epsilon=0.1, epsilon_fade=1e-6)

    state = env.reset()
    while(len(exp_buffer)<buffer_init_size):
        action = agent.explore(state)
        state_next, reward, done, _ = env.step(action)
        exp_buffer.append((state, action, reward, state_next, done))
        state = state_next
        if done:
            state = env.reset()

    episode = 0
    while episode < 20000:
        for i in range(0, 100):
            episode += 1
            total_reward = 0
            state = env.reset()
            step = 0
            while True:
                step += 1
                action = agent.explore(state)
                state_next, reward, done, _ = env.step(action)
                # env.render()
                exp_buffer.append((state, action, reward, state_next, done))
                # train
                if len(exp_buffer) > buffer_init_size and step%learn_freq == 0:
                    (state_batch, action_batch, reward_batch, state_next_batch, done_batch) = exp_buffer.sample(batch_size)
                    loss = agent.learn(state_batch, action_batch, reward_batch, state_next_batch, done_batch)
                total_reward += reward
                state = state_next
                if done:
                    break

        eval_reward = evaluate(env, agent, render=True)
        print('episode: %d  e_greed: %.5f test_reward: %.1f' %(episode, agent.epsilon, eval_reward))
    torch.save(agent.dqn.target_model, './dqn.pkl')