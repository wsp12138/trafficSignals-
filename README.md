# HMC-DQN

This repository contains the implementation of HMC-DQN, a cooperative reinforcement-learning framework for joint traffic signal control and vehicle routing.

Only the proposed method code is included. Baseline implementations, experimental result files, generated figures, paper drafts, model checkpoints, and SUMO datasets are intentionally excluded.

## Contents

- `main.py`: single-process HMC-DQN training entry point.
- `distributed_train.py`: multi-worker rollout and centralized update training entry point.
- `vehicle_single_env_finally.py`: vehicle-routing environment wrapper and SMDP decision logic.
- `env/wrapper_env.py`: SUMO/PettingZoo traffic-signal environment wrapper.
- `single_agent/Q_Value/DQN_action_mask.py`: masked SMDP-DQN vehicle-routing agent.
- `single_agent/Q_Value/MultiAgentMaskedDQN.py`: shared masked DQN signal-control agent.
- `common/arguments.py`: HMC-DQN argument parser.
- `common/utils.py`: observation helper utilities.
- `until.py`: SUMO road-network topology and route utilities.

## Data

The paper datasets and experimental outputs are not included in this repository.
To run training, provide SUMO files explicitly:

```bash
python main.py --net-file path/to/network.net.xml --route-file path/to/routes.rou.xml
```

or:

```bash
python distributed_train.py --net-file path/to/network.net.xml --route-file path/to/routes.rou.xml --workers 4
```

## Dependencies

Install SUMO separately and ensure `SUMO_HOME` is configured. Then install Python dependencies:

```bash
pip install -r requirements.txt
```

The code depends on `sumo-rl`, `traci`, `sumolib`, `pettingzoo`, `gymnasium`, `networkx`, `numpy`, and `torch`.

## Notes

This release is code-only. It does not include any baseline code, baseline data, trained models, logs, plots, tables, or result artifacts.
