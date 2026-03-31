import argparse
import torch


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=1000, help='the number of episodes before once training')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='the name of game')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--alg_name', type=str, default='A2C', help='name of alg')
    parser.add_argument('--hidden_dim', type=int, default=16, help='hidden dim')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--test_eps', type=int, default=30, help='number of the epoch to test the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--plot_dir', type=str, default='./plot', help='plot directory of reward')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    args = parser.parse_args()
    return args


def get_DQN_args(args):
    args.capacity = 10000
    args.target_update = 4
    args.lr = 0.0001
    args.epsilon_start = 0.90  # e-greedy策略中初始epsilon
    args.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
    args.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
    args.hidden_dim = 64
    args.embed_dim = 64
    return args


def get_AC_args(args):
    args.lr = 0.01
    args.hidden_dim = 36
    args.action_dim = 1
    return args


def get_PPO_args(args):
    args = get_AC_args(args)
    args.lr = 0.001
    args.k_episode = 10
    args.lam = 0.95
    args.epsilon = 0.2
    return args


def get_DDPG_args(args):
    args = get_AC_args(args)
    args.actor_lr = 1e-4
    args.critic_lr = 1e-3
    args.capacity = 1000000
    args.tau = 0.005
    return args


def get_PPO_continue_args(args):
    args.max_train_steps = int(3e6)
    args.actor_lr = 3e-4
    args.critic_lr = 3e-4
    args.lam = 0.95
    args.epsilon = 0.2
    args.k_episode = 10
    args.use_adv_norm = True
    args.use_state_norm = True
    args.use_reward_norm = False
    args.use_reward_scaling = True
    args.entropy_coef = 0.01
    args.use_lr_decay = True
    args.use_grad_clip = True
    args.use_orthogonal_init = True
    args.set_adm_eps = True
    args.use_tanh = True
    return args


def get_vdn_args(args):
    args.buffer_size = 1000
    args.lr = 0.001
    args.epsilon = 1
    args.epsilon_decay_steps = 5000
    args.epsilon_min = 0.05
    args.alg_name = "VDN"
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps
    return args


def get_MDDDPG_args(args):
    args.hidden_dim = 128
    args.lr_a = 1e-4
    args.lr_c = 1e-3
    args.capacity = 1000000
    args.tau = 0.005
    args.noise_std_init = 0.2
    args.noise_std_min = 0.05
    args.noise_decay_steps = 3e5
    args.use_noise_decay = True
    args.use_grad_clip = True
    args.use_orthogonal_init = True
    return args


def get_qmix_args(args):
    args.qmix_hidden_dim = 32
    args.hyper_hidden_dim = 64
    args.hyper_layers_num = 1
    args.lr = 0.01
    args.buffer_size = 100000
    args.epsilon = 1
    args.epsilon_min = 0.05
    args.epsilon_decay_steps = 3000
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps
    return args
