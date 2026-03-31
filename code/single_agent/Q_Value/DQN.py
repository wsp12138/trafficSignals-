import os
import math
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from network.BaseNet import MLP
# from common.utils import test
from common.replayBuffer import ReplayBuffer

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class DQN:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.add_agent_id = args.add_agent_id
        self.policy_net = MLP(args).to(args.device)
        self.target_net = MLP(args).to(args.device)
        self.buffer = ReplayBuffer(args)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.device = args.device
        self.frame = 0
        self.optim = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.epsilon = lambda frame_idx: args.epsilon_end + (args.epsilon_start - args.epsilon_end) * math.exp(
            -1. * frame_idx / args.epsilon_decay)

        self.model_dir = args.model_dir + '/' + args.alg_name + '/' + args.env_name + curr_time
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if args.load_model:
            self.load_model()

    def choose_action(self, obs, train=True):
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        if train:
            self.frame += 1
            if np.random.random() > self.epsilon(self.frame):
                with torch.no_grad():
                    action = self.policy_net(obs).max(0)[1].item()
            else:
                action = np.random.choice(self.args.output_dim)
        else:
            action = self.policy_net(obs).max(0)[1].item()
        return action

    def choose_action_for_mutil(self, obs, train=True):
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        inputs = []
        inputs.append(obs)
        if self.add_agent_id:
            inputs.append(torch.eye(self.n_agents).to(self.device))
        inputs = torch.cat([x for x in inputs], dim=-1)

        if train:
            self.frame += 1
            if np.random.random() > self.epsilon(self.frame):
                with torch.no_grad():

                    action = self.policy_net(inputs).max(dim=-1)[1].cpu().numpy()

                    # action = self.policy_net(obs).max(0)[1].item()
            else:
                action = [np.random.choice(self.args.output_dim) for _ in range(self.n_agents)]
        else:
            action = self.policy_net(inputs).max(dim=-1)[1].cpu().numpy()

        return action

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        q = self.policy_net(state).gather(dim=1, index=action.unsqueeze(1))
        next_q = self.target_net(next_state).max(dim=1)[0].detach()
        except_q = reward + self.gamma * next_q * (1 - done)
        loss = nn.MSELoss()(q, except_q.unsqueeze(1))
        self.optim.zero_grad()
        loss.backward()
        # 防止梯度爆炸
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

    def get_input(self, obs):

        inputs = []
        inputs.append(obs)
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.n_agents).unsqueeze(0).repeat(self.batch_size, 1, 1).to(self.device)

            inputs.append(agent_id_one_hot)
        inputs = torch.cat([x for x in inputs], dim=-1)
        return inputs

    def update_for_mutil(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        inputs_obs = self.get_input(state)

        inputs_next_obs = self.get_input(next_state)

        q = self.policy_net(inputs_obs).gather(dim=-1, index=action.unsqueeze(-1))

        next_q = self.target_net(inputs_next_obs).max(dim=-1)[0].unsqueeze(-1).detach()

        except_q = reward.unsqueeze(-1) + self.gamma * next_q * (1 - done.unsqueeze(-1))

        loss = nn.MSELoss()(q, except_q)
        self.optim.zero_grad()
        loss.backward()
        # 防止梯度爆炸
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

    def load_model(self):
        print(self.model_dir + '/policy_net_param.pkl')
        if os.path.exists(self.model_dir + '/policy_net_param.pkl'):
            path = self.model_dir + '/policy_net_param.pkl'
            self.policy_net.load_state_dict(torch.load(path))
            # self.target_net.load_state_dict(self.policy_net.state_dict())
            print('load policy net param')

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        print('save')
        torch.save(self.policy_net.state_dict(), self.model_dir + '/policy_net_param.pkl')


def train(env, args):
    print('train')
    print(f'env:{args.env_name},alg:{args.alg_name},device:{args.device}')
    agent = DQN(args)
    rewards = []

    for i_eps in range(args.episode):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.store(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
            agent.update()
            if done:
                break
        if (i_eps + 1) % args.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if (i_eps + 1) % 10 == 0:
            print('回合:{}/{},奖励:{}'.format(i_eps + 1, args.episode, ep_reward))

    # test(env, args, agent)
    env.close()
    agent.save_model()
    return rewards
