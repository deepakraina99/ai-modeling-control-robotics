import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic
from twolinkarm_env import TwoLinkArmEnv
# from temp_env import TwoLinkArmEnv
import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import csv

#Agent Model
BUFFER_SIZE  = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0

eps_start = 1.0
eps_end = 0.01
eps_decay = 1e-6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed

        #Actor Network (with Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        #Critic Network (with Target Network)
        self.critic_local = Critic(self.state_size, self.action_size, self.seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        #Noise process
        self.noise = OUNoise(action_size, random_seed)

        #Replay memory
        self.memory = ReplayBuffer(action_size,BUFFER_SIZE,BATCH_SIZE,random_seed)

    def step(self, state, action, reward, next_state, done):
        # for i in range(state.shape[0]):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        # self.actor_local.eval()
        # with torch.no_grad():
        action = self.actor_local(state).cpu().data.numpy().flatten()
            # self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
            # return np.clip(action, -np.pi, np.pi)
        return np.clip(action, -1.0, 1.0)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        # get predicted next state actions and Q vales from target models
        action_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, action_next)
        # compute Q targets
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filename):
        torch.save(self.actor_local.state_dict(), (filename + '_actor'))
        torch.save(self.actor_optimizer.state_dict(), (filename + '_actor_optimizer'))

        torch.save(self.critic_local.state_dict(), (filename + '_critic'))
        torch.save(self.critic_optimizer.state_dict(), (filename + '_critic_optimizer'))
        print('Model saved')

    def load(self, filename):
        self.actor_local.load_state_dict(torch.load((filename + '_actor')))
        self.actor_optimizer.load_state_dict(torch.load((filename + '_actor_optimizer')))
        self.actor_target.load_state_dict(torch.load((filename + '_actor')))

        self.critic_local.load_state_dict(torch.load((filename + '_critic')))
        self.critic_optimizer.load_state_dict(torch.load((filename + '_critic_optimizer')))
        self.critic_target.load_state_dict(torch.load((filename + '_critic')))
        print('Model loaded')

# Ornstein-Uhlenbeck process
class OUNoise:
    def __init__(self, size, seed, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience',field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Controller():
    def __init__(self, rand_seed = 0, rew_type = 0):
        self.env = TwoLinkArmEnv(rew_type)
        self.env.seed(rand_seed)
        self.agents = Agent(state_size = self.env.observation_space.shape[0],action_size = self.env.action_space.shape[0], random_seed = rand_seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rew_ver = rew_type

    # function for training the model
    def train(self, num_episodes = 5000, max_timesteps = 500, model_name = 'default'):
        print('Training started')
        avg_reward = []
        for i_episode in range(1,num_episodes+1):
            state = self.env.reset()
            score = 0
            rewards = []
            for t in range(max_timesteps):
                action = self.agents.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agents.step(state,action,reward, next_state, done)
                score += reward
                rewards.append(reward)
                state = next_state
                if done:
                   break
            
            scores = np.mean(rewards)
            avg_reward.append(scores)
            print('Episode {} Length {} Reward {:.2f} Avg. reward {:.2f}'.format(i_episode, len(rewards), score, scores))
                    
            if i_episode % 500 == 0:
                self.agents.save(filename='./trained_models/' + model_name + '_{}_{}'.format(i_episode, self.rew_ver))
                self.plot_reward(avg_reward)

    # function for plotting reward
    def plot_reward(self, avg_reward):
        # writing data to csv file
        myFile = open('logs/rewards_{}_{}.csv'.format(len(avg_reward), self.rew_ver), 'w')
        with myFile:
            writer = csv.writer(myFile)
            # writer.writerows(myData)
            writer.writerows(map(lambda x: [x], avg_reward))
        
        # plotting reward of all episodes
        # fig = plt.figure()
        # plt.plot(np.arange(len(avg_reward)), avg_reward)
        # plt.ylabel('Reward')
        # plt.xlabel('Episode')
        # plt.savefig('logs/rewards_{}_{}.pdf'.format(len(avg_reward), self.rew_ver))
        # plt.show()

        # plotting average reward
        mean_over = 100
        s=0
        R_mean = []
        for e in range(mean_over,len(avg_reward)):
            R_list = avg_reward[s:e]
            R_mean_item = sum(R_list)/mean_over
            R_mean.append(R_mean_item)
            s+=1
        R_mean = np.asarray(R_mean)

        fig = plt.figure()
        plt.plot(np.arange(len(R_mean)), R_mean)
        plt.ylabel('Average reward')
        plt.xlabel('Episode')
        plt.savefig('logs/avg_reward_{}_{}.pdf'.format(len(avg_reward), self.rew_ver))
        plt.show()
        
    # function for testing the model
    def test(self, num_test = 10, max_timesteps = 500, model_name = 'default'):
        print('Testing started')
        NumSuccess = 0
        self.agents.load(filename='./trained_models/' + model_name)
        for i in range(num_test):
            state = self.env.reset()
            self.agents.reset()   
            for t in range(max_timesteps+1):
                action = self.agents.act(state, add_noise=False)
                self.env.render()
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                if done:
                    break
            if t < max_timesteps:
                print('TEST: {}, SUCCESS'.format(i))
                NumSuccess += 1
            else:
                print('TEST: {}, FAIL'.format(i))
            self.env.close()
        print("Total {} success out of {} tests. Accuracy {} %".format(NumSuccess, num_test, NumSuccess/num_test))

