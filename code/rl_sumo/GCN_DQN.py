import os
import math
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from Net import GCN_, effect_net
from common.replayBuffer import ReplayBuffer

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class GCN_DQN:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.add_agent_id = args.add_agent_id
        self.policy_net = GCN_(args).to(args.device)
        self.target_net = GCN_(args).to(args.device)
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

    def choose_action(self, obs, edge_index, train=True):
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.float32, device=self.device)

        if train:
            self.frame += 1
            if np.random.random() > self.epsilon(self.frame):
                with torch.no_grad():
                    action = self.policy_net(obs, edge_index).max(0)[1].item()
            else:
                action = np.random.choice(self.args.output_dim)
        else:
            action = self.policy_net(obs, edge_index).max(0)[1].item()
        return action

    def choose_action_for_mutil(self, obs, edge_index, train=True):
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.int64, device=self.device)
        # batch = torch.tensor(np.array(batch), dtype=torch.int64, device=self.device)
        inputs = []
        inputs.append(obs)
        if self.add_agent_id:
            inputs.append(torch.eye(self.n_agents).to(self.device))
        inputs = torch.cat([x for x in inputs], dim=-1)
        if train:
            self.frame += 1
            if np.random.random() > self.epsilon(self.frame):
                with torch.no_grad():
                    action = self.policy_net(inputs, edge_index).max(dim=-1)[1].cpu().numpy()
                    # action = self.policy_net(obs).max(0)[1].item()
            else:
                action = [np.random.choice(self.args.output_dim) for _ in range(self.n_agents)]
        else:
            action = self.policy_net(inputs, edge_index).max(dim=-1)[1].cpu().numpy()

        return action

    def update(self, edge_index):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.int32, device=self.device)

        q = self.policy_net(state).gather(dim=1, index=action.unsqueeze(1))
        next_q = self.target_net(next_state).max(dim=1)[0].detach()
        except_q = reward + self.gamma * next_q * (1 - done)

        loss = nn.MSELoss()(q, except_q.unsqueeze(1))
        self.optim.zero_grad()
        loss.backward()

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

    def update_for_mutil(self, edge_index):
        if len(self.buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.int64, device=self.device)
        # batch = torch.tensor(np.array(batch), dtype=torch.int64, device=self.device)

        inputs_obs = self.get_input(state)
        inputs_next_obs = self.get_input(next_state)
        q = self.policy_net(inputs_obs, edge_index).gather(dim=-1, index=action.unsqueeze(-1))
        next_q = self.target_net(inputs_next_obs, edge_index).max(dim=-1)[0].unsqueeze(-1).detach()
        except_q = reward.unsqueeze(-1) + self.gamma * next_q * (1 - done.unsqueeze(-1))

        loss = nn.MSELoss()(q, except_q)
        self.optim.zero_grad()
        loss.backward()

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


class Effect:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.policy_net = effect_net(args).to(args.device)
        self.target_net = effect_net(args).to(args.device)

        self.buffer = ReplayBuffer(args)
        self.optim = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def choose_action(self, influence_infos):
        influence_infos = torch.tensor(np.array(influence_infos), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.policy_net(influence_infos).max(dim=-1)[1].cpu().numpy()
        return action

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        q = self.policy_net(state).gather(dim=-1, index=action.unsqueeze(-1))
        next_q = self.target_net(next_state).max(dim=-1)[0].unsqueeze(-1).detach()
        except_q = reward.unsqueeze(-1) + self.gamma * next_q * (1 - done.unsqueeze(-1))

        loss = nn.MSELoss()(q, except_q)
        self.optim.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
