import os
import math
import datetime

import torch.optim as optim

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.capacity = args.capacity
        self.buffer = []
        self.position = 0

    def store(self, state, action, reward, next_state, done, tau):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, tau)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, tau = zip(*batch)

        obs = np.stack([s[0] for s in state])
        next_obs = np.stack([s[0] for s in next_state])
        next_action_mask = np.stack([s[1] for s in next_state])

        obs = torch.tensor(obs, device=self.args.device, dtype=torch.float32)
        action = torch.tensor(np.array(action), device=self.args.device, dtype=torch.int64)
        reward = torch.tensor(reward, device=self.args.device, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, device=self.args.device, dtype=torch.float32)
        next_action_mask = torch.tensor(next_action_mask, device=self.args.device, dtype=torch.bool)
        done = torch.tensor(done, dtype=torch.float, device=self.args.device)
        tau = torch.tensor(tau, dtype=torch.float32, device=self.args.device)

        return obs, action, reward, next_obs, next_action_mask, done, tau

    def __len__(self):
        return len(self.buffer)


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.f1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.f2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.f3 = nn.Linear(args.hidden_dim, args.max_output_dim)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.f3(x)


class action_mask_DQN:
    def __init__(self, args):
        print(args.input_dim, 'input_dim')
        self.args = args
        self.n_agents = args.n_agents
        self.policy_net = MLP(args).to(args.device)

        self.target_net = MLP(args).to(args.device)
        self.buffer = ReplayBuffer(args)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.device = args.device
        self.frame = 0
        self.use_double_dqn = getattr(args, "use_double_dqn", True)
        self.optim = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.epsilon = lambda frame_idx: args.epsilon_end + (args.epsilon_start - args.epsilon_end) * math.exp(
            -1. * frame_idx / args.epsilon_decay)

        self.model_dir = args.model_dir + '/' + args.alg_name + '/' + args.env_name + curr_time
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 奖励归一化的运行统计量
        self.reward_running_mean = 0.0
        self.reward_running_var = 1.0
        self.reward_count = 0

        if args.load_model:
            self.load_model()

    def choose_action(self, input, train=True):
        obs = torch.tensor(np.array(input[0]), dtype=torch.float32, device=self.device)
        action_mask = torch.tensor(np.array(input[1]), dtype=torch.bool, device=self.device)

        if train:
            self.frame += 1
            if np.random.random() > self.epsilon(self.frame):
                with torch.no_grad():
                    q = self.policy_net(obs)
                    action = q.masked_fill(~action_mask, float('-inf')).max(-1)[1].item()
            else:
                avail_actions = np.where(input[1])[0]
                if len(avail_actions) > 0:
                    action = np.random.choice(avail_actions)
                else:
                    action = 0
        else:
            q = self.policy_net(obs)
            action = q.masked_fill(~action_mask, float('-inf')).max(-1)[1].item()
        return action

    def _normalize_reward(self, reward_tensor):
        """对reward做运行均值归一化，防止Q值爆炸"""
        batch_mean = reward_tensor.mean().item()
        batch_var = reward_tensor.var().item() + 1e-8

        # 更新运行统计量
        self.reward_count += 1
        alpha = max(0.01, 1.0 / self.reward_count)
        self.reward_running_mean = (1 - alpha) * self.reward_running_mean + alpha * batch_mean
        self.reward_running_var = (1 - alpha) * self.reward_running_var + alpha * batch_var

        # 归一化
        std = max(math.sqrt(self.reward_running_var), 1.0)
        normalized = (reward_tensor - self.reward_running_mean) / std
        # 裁剪归一化后的奖励，防止极端值
        return torch.clamp(normalized, -5.0, 5.0)

    def update_for_mutil(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        state, action, reward, next_state, next_state_action_mask, done, tau = self.buffer.sample(self.batch_size)

        # ===== 关键修复1: 奖励归一化 =====
        reward = self._normalize_reward(reward)

        # 当前Q值
        q = self.policy_net(state).gather(dim=1, index=action.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_target = self.target_net(next_state)
            has_valid_action = next_state_action_mask.any(dim=1)
            if self.use_double_dqn:
                next_q_policy = self.policy_net(next_state).clone()
                next_q_policy[~next_state_action_mask] = float('-inf')
                best_actions = next_q_policy.max(dim=1)[1]
                next_q_value = next_q_target.gather(dim=1, index=best_actions.unsqueeze(1)).squeeze()
            else:
                next_q_target = next_q_target.clone()
                next_q_target[~next_state_action_mask] = float('-inf')
                next_q_value = next_q_target.max(dim=1).values
            next_q_max = torch.where(has_valid_action, next_q_value, torch.tensor(0.0, device=self.device))

        tau_clip = float(getattr(self.args, "tau_discount_clip", 10.0))
        effective_tau = torch.clamp(tau, min=1.0, max=tau_clip)
        discount = torch.pow(torch.full_like(effective_tau, self.gamma), effective_tau)
        expected_q = reward + discount * next_q_max * (1 - done)

        loss = nn.SmoothL1Loss()(q, expected_q)
        self.optim.zero_grad()
        loss.backward()

        # ===== 关键修复3: 梯度裁剪 =====
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optim.step()

        return loss.item()

    def load_model(self):
        print(self.model_dir + '/dqn-param.pkl')
        if os.path.exists(self.model_dir + '/dqn-param.pkl'):
            path = self.model_dir + '/dqn-param.pkl'
            self.policy_net.load_state_dict(torch.load(path))
            print('load policy net param')

    def save_model(self, episode):
        model_dir = '../model/param_dqn'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.policy_net.state_dict(), model_dir + '/policy_net_param' + str(episode) + '.pkl')

    def save_model_(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(self.policy_net.state_dict(), model_dir + '/veh.pkl')
        print('save')
