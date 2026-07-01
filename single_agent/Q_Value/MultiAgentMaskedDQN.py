import datetime
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class ReplayBuffer:
    def __init__(self, args):
        self.capacity = args.capacity
        self.buffer = []
        self.position = 0

    def store(self, state, action_mask, action, reward, next_state, next_action_mask, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action_mask, dtype=bool),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            np.asarray(next_action_mask, dtype=bool),
            np.asarray(done, dtype=np.float32),
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        state, action_mask, action, reward, next_state, next_action_mask, done = zip(*batch)
        return (
            torch.tensor(np.stack(state), device=device, dtype=torch.float32),
            torch.tensor(np.stack(action_mask), device=device, dtype=torch.bool),
            torch.tensor(np.stack(action), device=device, dtype=torch.long),
            torch.tensor(np.stack(reward), device=device, dtype=torch.float32),
            torch.tensor(np.stack(next_state), device=device, dtype=torch.float32),
            torch.tensor(np.stack(next_action_mask), device=device, dtype=torch.bool),
            torch.tensor(np.stack(done), device=device, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.f1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.f2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.f3 = nn.Linear(args.hidden_dim, args.max_output_dim)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.f3(x)


class MultiAgentMaskedDQN:
    """Shared masked Double DQN for heterogeneous traffic-signal agents.

    Each intersection can have a different number of phases. The network emits
    Q-values for the maximum phase count, and masks prune impossible phases
    during action selection and target computation.
    """

    def __init__(self, args):
        self.args = args
        self.policy_net = MLP(args).to(args.device)
        self.target_net = MLP(args).to(args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.buffer = ReplayBuffer(args)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.device = args.device
        self.frame = 0
        self.use_double_dqn = getattr(args, "use_double_dqn", True)
        self.optim = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.epsilon = lambda frame_idx: args.epsilon_end + (args.epsilon_start - args.epsilon_end) * math.exp(
            -1.0 * frame_idx / args.epsilon_decay
        )
        self.model_dir = args.model_dir + "/" + args.alg_name + "/" + args.env_name + curr_time
        self.reward_running_mean = 0.0
        self.reward_running_var = 1.0
        self.reward_count = 0

    def choose_actions(self, obs, action_mask, train=True):
        obs_tensor = torch.tensor(np.asarray(obs), device=self.device, dtype=torch.float32)
        mask_tensor = torch.tensor(np.asarray(action_mask), device=self.device, dtype=torch.bool)
        actions = []

        if train:
            self.frame += 1

        explore = train and np.random.random() <= self.epsilon(self.frame)
        if explore:
            for mask in np.asarray(action_mask):
                valid_actions = np.where(mask)[0]
                if len(valid_actions) == 0:
                    actions.append(0)
                else:
                    actions.append(int(np.random.choice(valid_actions)))
            return np.asarray(actions, dtype=np.int64)

        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
            q_values = q_values.masked_fill(~mask_tensor, float("-inf"))
            actions = q_values.argmax(dim=-1).detach().cpu().numpy()
        return actions.astype(np.int64)

    def _normalize_reward(self, reward):
        batch_mean = reward.mean().item()
        batch_var = reward.var().item() + 1e-8
        self.reward_count += 1
        alpha = max(0.01, 1.0 / self.reward_count)
        self.reward_running_mean = (1.0 - alpha) * self.reward_running_mean + alpha * batch_mean
        self.reward_running_var = (1.0 - alpha) * self.reward_running_var + alpha * batch_var
        std = max(math.sqrt(self.reward_running_var), 1.0)
        return torch.clamp((reward - self.reward_running_mean) / std, -5.0, 5.0)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        state, action_mask, action, reward, next_state, next_action_mask, done = self.buffer.sample(
            self.batch_size, self.device
        )
        del action_mask

        reward = self._normalize_reward(reward)
        q = self.policy_net(state).gather(dim=-1, index=action.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            has_valid_action = next_action_mask.any(dim=-1)
            if self.use_double_dqn:
                next_q_policy = self.policy_net(next_state).masked_fill(~next_action_mask, float("-inf"))
                best_actions = next_q_policy.argmax(dim=-1, keepdim=True)
                next_q_target = self.target_net(next_state).gather(dim=-1, index=best_actions).squeeze(-1)
            else:
                next_q_target_all = self.target_net(next_state).masked_fill(~next_action_mask, float("-inf"))
                next_q_target = next_q_target_all.max(dim=-1).values
            next_q_target = torch.where(has_valid_action, next_q_target, torch.zeros_like(next_q_target))
            expected_q = reward + self.gamma * next_q_target * (1.0 - done)

        loss = nn.SmoothL1Loss()(q, expected_q)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optim.step()
        return float(loss.item())

    def soft_update_target(self, tau=0.005):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def save_model_(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.policy_net.state_dict(), os.path.join(model_dir, "tl.pkl"))
