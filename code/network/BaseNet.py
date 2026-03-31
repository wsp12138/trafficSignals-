import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h



class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.output_dim = args.output_dim
        self.conv1 = nn.Conv2d(args.stack_size, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # 初始化并计算全连接层输入的特征数量
        self.feature_dim = self._get_conv_output((args.stack_size, 64, 64))

        # 定义全连接层
        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, args.output_dim)

    def _get_conv_output(self, shape):
        # 计算卷积层的输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.numel()

    def forward(self, x):
        # 卷积层
        batch_size, agent_num, channel, height, width = x.size()

        x = x.view(batch_size * agent_num, channel, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 展平
        x = x.reshape(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(batch_size, agent_num, self.output_dim)

        return x


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.f1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.f2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.f3 = nn.Linear(args.hidden_dim, args.output_dim)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.f3(x)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.f2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.f3 = nn.Linear(args.hidden_dim, args.output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        action_scores = self.f3(x)
        dist = Categorical(self.softmax(action_scores))
        return dist


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.f1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.f2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.f3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.f3(x)


class ACNet(nn.Module):
    def __init__(self, args):
        super(ACNet, self).__init__()
        self.actor = Actor(args)
        self.critic = Critic(args)

    def forward(self, x):
        dist = self.actor(x)
        value = self.critic(x)
        return dist, value


class Critic_Q(nn.Module):
    def __init__(self, args):
        super(Critic_Q, self).__init__()
        self.f1 = nn.Linear(args.input_dim + args.action_dim, 400)
        self.f2 = nn.Linear(400, 300)
        self.f3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.f3(x)


class Actor_continue(nn.Module):
    def __init__(self, args):
        super(Actor_continue, self).__init__()
        self.args = args
        self.f1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.f2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.f3 = nn.Linear(args.hidden_dim, args.output_dim)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return torch.tanh(x) * self.args.max_action


class DuelingNet(nn.Module):
    def __init__(self, args):
        super(DuelingNet, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(args.input_dim, args.hidden_dim),
            nn.ReLU()
        )
        self.value_linear = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.output_dim)
        )

    def forward(self, x):
        x = self.hidden(x)
        v = self.value_linear(x)
        ad = self.advantage(x)
        return v + (ad - ad.mean())


# Actor Network
class Actor_ddpg(nn.Module):
    def __init__(self, args):
        super(Actor_ddpg, self).__init__()
        self.args = args
        self.l1 = nn.Linear(args.state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, args.action_dim)
        self.max_action = args.max_action

    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return self.args.max_action * x


# Critic Network
class Critic_ddpg(nn.Module):
    def __init__(self, args):
        super(Critic_ddpg, self).__init__()
        self.args = args
        self.l1 = nn.Linear(args.state_dim + args.action_dim, args.hidden_dim)
        self.l2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.l3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, state, action):
        x = torch.relu(self.l1(torch.cat([state, action], 1)))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.std_init = std_init
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / (self.in_features ** 0.5)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / (self.out_features ** 0.5))

    def reset_noise(self):
        epsilon_in = self._scalse_noise(self.in_features)
        epsilon_out = self._scalse_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scalse_noise(self.out_features))

    def _scalse_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())


class NoisyNet(nn.Module):
    def __init__(self, args):
        super(NoisyNet, self).__init__()
        self.args = args
        self.linear = nn.Linear(args.state_dim, args.hidden_dim)
        self.noisy1 = NoisyLinear(args.hidden_dim, args.hidden_dim)
        self.noisy2 = NoisyLinear(args.hidden_dim, args.action_dim)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy1(x))
        return self.noisy2(x)


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.args = args
        self.max_action = args.max_action
        self.f1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.f2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.mean_layer = nn.Linear(args.hidden_dim, args.output_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.output_dim))
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.f1)
            orthogonal_init(self.f2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.f1(s))
        s = self.activate_func(self.f2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist


class Critic_ppo(nn.Module):
    def __init__(self, args):
        super(Critic_ppo, self).__init__()
        self.f1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.f2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.f3 = nn.Linear(args.hidden_dim, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.f1)
            orthogonal_init(self.f2)
            orthogonal_init(self.f3)

    def forward(self, s):
        s = self.activate_func(self.f1(s))
        s = self.activate_func(self.f2(s))
        v_s = self.f3(s)
        return v_s


class Critic_TD3(nn.Module):
    def __init__(self, args):
        super(Critic_TD3, self).__init__()
        self.l1 = nn.Linear(args.input_dim + args.output_dim, args.hidden_width)
        self.l2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.l3 = nn.Linear(args.hidden_width, 1)
        # Q2
        self.l4 = nn.Linear(args.input_dim + args.output_dim, args.hidden_width)
        self.l5 = nn.Linear(args.hidden_width, args.hidden_width)
        self.l6 = nn.Linear(args.hidden_width, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class Actor_SAC(nn.Module):
    def __init__(self, args):
        super(Actor_SAC, self).__init__()
        self.max_action = args.max_action
        self.l1 = nn.Linear(args.input_dim, args.hidden_width)
        self.l2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.output_dim)
        self.log_std_layer = nn.Linear(args.hidden_width, args.output_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(
            a)  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi


class Critic_SAC(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, args):
        super(Critic_SAC, self).__init__()
        # Q1
        self.l1 = nn.Linear(args.input_dim + args.output_dim, args.hidden_width)
        self.l2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.l3 = nn.Linear(args.hidden_width, 1)
        # Q2
        self.l4 = nn.Linear(args.input_dim + args.output_dim, args.hidden_width)
        self.l5 = nn.Linear(args.hidden_width, args.hidden_width)
        self.l6 = nn.Linear(args.hidden_width, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2



