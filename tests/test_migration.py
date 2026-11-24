import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import rlcard
from rlcard.agents import CFRAgent, DQNAgent, HumanAgent, NFSPAgent, RandomAgent, RuleBasedAgent
from rlcard.agents.dmc_agent import DMCAgent, DMCModel, DMCTrainer
from rlcard.utils import reorganize, set_seed


def test_env_and_agents_gameplay():
    set_seed(42)
    env = rlcard.make('swy-blm', config={'seed': 42})
    assert env.state_shape == [[230], [230]]
    assert env.action_shape == [[44], [44]]

    env.set_agents([RuleBasedAgent(), RandomAgent()])
    for _ in range(10):
        trajs, payoffs = env.run(is_training=False)
        assert len(trajs) == 2
        assert len(payoffs) == 2
        assert np.isclose(payoffs[0] + payoffs[1], 0.0)


def test_dmc_agent_device_normalization():
    env = rlcard.make('swy-blm')
    agent_cpu = DMCAgent(env.state_shape[0], env.action_shape[0], mlp_layers=[32, 32], device='cpu')
    assert agent_cpu.device == 'cpu'

    # Ensure cuda:0 is not double-prefixed into cuda:cuda:0 (without requiring CUDA hardware)
    from rlcard.agents.dmc_agent.model import _normalize_device
    assert _normalize_device('cpu') == 'cpu'
    assert _normalize_device('0') == 'cuda:0'
    assert _normalize_device(1) == 'cuda:1'
    assert _normalize_device('cuda:0') == 'cuda:0'
    assert _normalize_device(torch.device('cpu')) == 'cpu'


def test_dqn_nfsp_cfr_agents():
    env = rlcard.make('swy-blm', config={'seed': 7})

    dqn = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[32, 32],
        batch_size=8,
        replay_memory_init_size=16,
        device=torch.device('cpu'),
    )
    env.set_agents([dqn, RandomAgent()])
    for _ in range(5):
        trajs, payoffs = env.run(is_training=True)
        for ts in reorganize(trajs, payoffs)[0]:
            dqn.feed(ts)

    nfsp = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[32, 32],
        q_mlp_layers=[32, 32],
        batch_size=8,
        min_buffer_size_to_learn=16,
        q_batch_size=8,
        q_replay_memory_init_size=16,
        device=torch.device('cpu'),
    )
    env.set_agents([nfsp, RandomAgent()])
    for _ in range(5):
        nfsp.sample_episode_policy()
        trajs, payoffs = env.run(is_training=True)
        for ts in reorganize(trajs, payoffs)[0]:
            nfsp.feed(ts)

    cfr = CFRAgent(env)
    state, _ = env.reset()
    action, info = cfr.eval_step(state)
    assert action in state['legal_actions']
    assert 'probs' in info


def test_dmc_trainer_train_resume_and_evaluate(tmp_path):
    env = rlcard.make('swy-blm')
    savedir = str(tmp_path)
    trainer = DMCTrainer(
        env,
        cuda='',
        actor_device='cpu',
        xpid='test_dmc',
        savedir=savedir,
        total_iterations=50,
        num_actor_devices=1,
        num_actors=2,
        training_device='cpu',
        batch_size=4,
        unroll_length=6,
        num_buffers=16,
        num_threads=1,
        sync_interval=5,
        mlp_layers=[32, 32],
    )
    trainer.start()

    ckpt_path = os.path.join(savedir, 'test_dmc', 'model.tar')
    assert os.path.exists(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert 'model_state_dict' in ckpt
    assert len(ckpt['model_state_dict']) == 2

    # Resume training from checkpoint
    trainer_resume = DMCTrainer(
        env,
        cuda='',
        load_model=True,
        xpid='test_dmc',
        savedir=savedir,
        total_iterations=60,
        num_actor_devices=1,
        num_actors=2,
        training_device='cpu',
        batch_size=4,
        unroll_length=6,
        num_buffers=8,
        num_threads=2,
        mlp_layers=[32, 32],
    )
    trainer_resume.start()

    # Load weights into fresh DMCAgents and validate
    agents = [
        DMCAgent(env.state_shape[p], env.action_shape[p], mlp_layers=[32, 32], device='cpu')
        for p in range(env.num_players)
    ]
    for p in range(env.num_players):
        agents[p].load_state_dict(ckpt['model_state_dict'][p])

    from evaluate import validate
    res = validate(env, agents, RandomAgent(), M=20)
    assert 'first_avg_score' in res
    assert 'second_avg_score' in res


def test_train_rl_cli_smoke(tmp_path):
    log_dir = str(tmp_path / 'rl_run')
    proc = subprocess.run(
        [
            sys.executable,
            'train_rl.py',
            '--env', 'swy-blm',
            '--algorithm', 'dqn',
            '--num_episodes', '15',
            '--num_eval_games', '10',
            '--evaluate_every', '10',
            '--log_dir', log_dir,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert os.path.exists(os.path.join(log_dir, 'model.pth'))
    assert os.path.exists(os.path.join(log_dir, 'fig.png'))


def test_train_dmc_and_evaluate_cli_smoke(tmp_path):
    savedir = str(tmp_path / 'dmc_runs')
    proc_train = subprocess.run(
        [
            sys.executable,
            'train_dmc.py',
            '--env', 'swy-blm',
            '--xpid', 'cli_test',
            '--actor_device', 'cpu',
            '--num_actor_devices', '1',
            '--num_actors', '2',
            '--training_device', 'cpu',
            '--batch_size', '8',
            '--num_buffers', '16',
            '--num_threads', '1',
            '--sync_interval', '5',
            '--savedir', savedir,
            '--total_iterations', '50',
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc_train.returncode == 0, proc_train.stderr
    model_path = os.path.join(savedir, 'cli_test', 'model.tar')
    assert os.path.exists(model_path)

    for agent_name in ('dmc', 'dmc_search'):
        proc_eval = subprocess.run(
            [
                sys.executable,
                'evaluate.py',
                '--ai_agent', agent_name,
                '--model_path', model_path,
                '--baseline_agent', 'rule-based',
                '--total_simulations', '20',
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc_eval.returncode == 0, proc_eval.stderr
        assert 'first_avg_score' in proc_eval.stdout
