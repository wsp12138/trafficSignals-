import argparse

import torch


def get_common_args():
    parser = argparse.ArgumentParser(description="HMC-DQN training for cooperative traffic signal control and vehicle routing.")
    parser.add_argument("--episode", type=int, default=1000, help="number of training episodes")
    parser.add_argument(
        "--dataset",
        type=str,
        default="nanchang_split1",
        choices=["nanchang_split1", "hangzhou", "hangzhou_peak"],
        help="dataset preset used by main.py; external SUMO files can be supplied with --net-file and --route-file",
    )
    parser.add_argument("--net-file", type=str, default="", help="override SUMO network file")
    parser.add_argument("--route-file", type=str, default="", help="override SUMO route file")
    parser.add_argument("--num-seconds", type=int, default=3600, help="simulation horizon in seconds")
    parser.add_argument("--use-gui", action="store_true", help="run SUMO with GUI")
    parser.add_argument("--workers", type=int, default=1, help="number of rollout workers for distributed_train.py")
    parser.add_argument(
        "--q-target",
        type=str,
        default="dqn",
        choices=["ddqn", "dqn"],
        help="target Q calculation: ddqn uses online action selection and target evaluation; dqn uses target max",
    )
    parser.add_argument(
        "--cav-penetration",
        type=float,
        default=0.2,
        help="fraction of vehicles controlled by the vehicle routing agent",
    )
    parser.add_argument("--cav-seed", type=int, default=0, help="seed for deterministic CAV subset selection")
    parser.add_argument("--env_name", type=str, default="SUMO", help="environment name")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--device", type=str, default=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    parser.add_argument("--alg_name", type=str, default="HMC-DQN", help="algorithm name")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--model_dir", type=str, default="./model", help="model output directory")
    parser.add_argument("--result_dir", type=str, default="./result", help="training log output directory")
    parser.add_argument("--load_model", action="store_true", help="load a pretrained model if available")
    return parser.parse_args()


def get_DQN_args(args):
    args.capacity = 10000
    args.target_update = 4
    args.lr = 0.0001
    args.epsilon_start = 0.90
    args.epsilon_end = 0.01
    args.epsilon_decay = 500
    args.hidden_dim = 64
    args.embed_dim = 64
    return args
