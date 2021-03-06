import gym
import numpy as np
import time

class Qlearning_Agent(object):
    def __init__(self, state_dim, act_dim, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        # size of action space
        # learning rate
        # discount factor
        # probability for random action
        # Q table
        self.act_n = act_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((state_dim, act_dim))

    # epsilon-greedy to move
    def action(self, state):
        # greedy
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.greedy(state)
        # random to discover
        else:
            action = np.random.choice(self.act_n)
        return action

    # get ptimal action according to Q table and current state
    def greedy(self, state):
        Q_list = self.Q[state, :] # current state's Q value
        maxQ = np.max(Q_list) # find max Q
        action_list = np.where(Q_list == maxQ)[0]  # when exists many actions for max Q
        action = np.random.choice(action_list)
        return action

    # learning
    def iterate(self, state, action, reward, state_next, done):
        """ on-policy
            state: current state_t
            action: action_t
            reward: reward for action_t
            state_next: state_t+1
            action_next: action_t+1
            done: if episode is over
        """
        current_Q = self.Q[state, action]
        if done:
            target_Q = reward
        else:
            # Q learning
            target_Q = reward + self.gamma * np.max(self.Q[state_next, :])
        self.Q[state, action] += self.lr * (target_Q - current_Q) # update Q table


def test(env, agent):
    total_reward = 0
    state = env.reset()
    while True:
        # greedy, no need to exploit
        action = agent.greedy(state)
        state_next, reward, done, _ = env.step(action)
        total_reward += reward
        state = state_next
        #env.render()
        if done:
            break
    return total_reward

if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left

    agent = Qlearning_Agent(
        state_dim=env.observation_space.n,
        act_dim=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)
    for episode in range(500):
        total_steps = 0
        total_reward = 0

        state = env.reset()

        while True:
            action = agent.action(state)  # 根据算法选择一个动作
            state_next, reward, done, _ = env.step(action)  # 与环境进行一个交互

            # Q learning
            agent.iterate(state, action, reward, state_next, done)
            # update
            state = state_next
            total_reward += reward
            total_steps += 1
            if done:
                break
        print('Episode %s: steps = %s , reward = %.1f' % (episode, total_steps, total_reward))
        #env.render()

    test_reward = test(env, agent)
    print('test reward = %.1f' % (test_reward))