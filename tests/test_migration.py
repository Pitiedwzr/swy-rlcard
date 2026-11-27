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
    assert env.state_shape == [[264], [264]]
    assert env.action_shape == [[44], [44]]

    env_legacy = rlcard.make('swy-blm', config={'seed': 42, 'legacy_obs': True, 'win_reward_weight': 5.0})
    assert env_legacy.state_shape == [[230], [230]]
    env_legacy.set_agents([RuleBasedAgent(), RandomAgent()])
    _, shaped_payoffs = env_legacy.run(is_training=False)
    assert np.isclose(shaped_payoffs[0] + shaped_payoffs[1], 0.0)
    score_diff = env_legacy.game.final_score[0] - env_legacy.game.final_score[1]
    assert np.isclose(shaped_payoffs[0], 5.0 * np.sign(score_diff) + score_diff)

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
        augment_symmetry=True,
        aug_copies=2,
        loss_type='smooth_l1',
        loss_beta=2.0,
        snapshot_prob=0.15,
        rule_opponent_prob=0.15,
        snapshot_pool_size=2,
        snapshot_interval=10,
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


def test_swy_blm_v4_history_privacy_and_equivariance():
    from rlcard.agents.dmc_agent.equivariant_model import EquivariantDMCAgent, SwyBlmEquivariantNet
    from rlcard.agents.dmc_agent.symmetry import build_swy_blm_symmetry_tables

    set_seed(11)
    env = rlcard.make('swy-blm', config={'seed': 11, 'obs_mode': 'v4'})
    assert env.state_shape == [[318], [318]]

    agents = [
        EquivariantDMCAgent(env.state_shape[p], env.action_shape[p], d_model=32, num_blocks=2, device='cpu')
        for p in range(2)
    ]
    env.set_agents(agents)
    trajs, _ = env.run(is_training=False)

    for p in range(2):
        for idx in range(0, len(trajs[p]) - 2, 2):
            raw = trajs[p][idx]['raw_obs']
            for card, _, sec in raw['oppo_history']:
                if sec == 0:
                    assert card is None
            turn = int(raw['cur_turn'])
            oppo_priv = env.game.get_oppo_private_at_turn(p, turn)
            assert len(oppo_priv) == raw['oppo_private_count']

    obs_perms, act_perms = build_swy_blm_symmetry_tables(obs_dim=318)
    net = SwyBlmEquivariantNet(d_model=32, num_blocks=2)
    net.eval()

    sample_obs = torch.from_numpy(trajs[0][4]['obs']).float().unsqueeze(0)
    with torch.no_grad():
        wdl0, diff0, bel0, bel_mid0 = net.forward_all(sample_obs)
        rng = np.random.default_rng(123)
        for si in rng.choice(obs_perms.shape[0], size=24, replace=False):
            op = obs_perms[si]
            ap = act_perms[si]
            cp = ap[:22]
            p_obs = sample_obs[:, op]
            wdl1, diff1, bel1, bel_mid1 = net.forward_all(p_obs)
            assert torch.allclose(wdl1, wdl0[:, ap, :], atol=1e-4, rtol=1e-4)
            assert torch.allclose(diff1, diff0[:, ap], atol=1e-4, rtol=1e-4)
            assert torch.allclose(bel1, bel0[:, cp], atol=1e-4, rtol=1e-4)
            assert torch.allclose(bel_mid1, bel_mid0[:, cp], atol=1e-4, rtol=1e-4)


def test_dmc_agent_sym_avg():
    env = rlcard.make('swy-blm', config={'seed': 5, 'obs_mode': 'v2'})
    agent = DMCAgent(env.state_shape[0], env.action_shape[0], mlp_layers=[32, 32], device='cpu', sym_avg_k=8)
    state, _ = env.reset()
    action, info = agent.eval_step(state)
    assert action in state['legal_actions']
    assert 'values' in info


def test_equivariant_dmc_train_resume_and_evaluate_cli_smoke(tmp_path):
    savedir = str(tmp_path / 'eq_runs')
    common_args = [
        sys.executable,
        'train_dmc.py',
        '--env', 'swy-blm',
        '--arch', 'equivariant',
        '--d_model', '32',
        '--num_blocks', '2',
        '--xpid', 'eq_test',
        '--actor_device', 'cpu',
        '--num_actor_devices', '1',
        '--num_actors', '2',
        '--training_device', 'cpu',
        '--batch_size', '8',
        '--num_buffers', '16',
        '--num_threads', '1',
        '--sync_interval', '5',
        '--snapshot_prob', '0.0',
        '--rule_opponent_prob', '0.0',
        '--savedir', savedir,
    ]
    proc_train = subprocess.run(
        common_args + ['--total_iterations', '40'],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc_train.returncode == 0, proc_train.stderr
    model_path = os.path.join(savedir, 'eq_test', 'model.tar')
    assert os.path.exists(model_path)

    proc_resume = subprocess.run(
        common_args + ['--load_model', '--total_iterations', '50'],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc_resume.returncode == 0, proc_resume.stderr

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


def test_cli_entrypoints_no_circular_import():
    proc_import = subprocess.run(
        [sys.executable, '-c', 'from evaluate import load_dmc_agents, validate'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_import.returncode == 0, proc_import.stderr

    for script in ('evaluate.py', 'assist.py', 'evaluate_human.py'):
        proc_help = subprocess.run(
            [sys.executable, script, '--help'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc_help.returncode == 0, proc_help.stderr


def test_swy_blm_extract_state_fast_parity():
    from rlcard.agents.dmc_search_agent import _extract_obs_batch
    from rlcard.games.bailongmen.utils import _cards2array

    for obs_mode, expected_dim in (('legacy', 230), ('v2', 264), ('v4', 318)):
        env = rlcard.make('swy-blm', config={'seed': 19, 'obs_mode': obs_mode})
        env.set_agents([RandomAgent(44), RandomAgent(44)])
        for _ in range(15):
            trajs, _ = env.run(is_training=True)
            for p in range(2):
                for idx in range(0, len(trajs[p]) - 2, 2):
                    st = trajs[p][idx]
                    raw = st['raw_obs']
                    assert st['obs'].shape == (expected_dim,)
                    if obs_mode in ('legacy', 'v2'):
                        ref = _extract_obs_batch(
                            _cards2array(raw['board_public'])[None, :],
                            _cards2array(raw['cur_public'])[None, :],
                            _cards2array(raw['cur_private'])[None, :],
                            _cards2array(raw['oppo_public'])[None, :],
                            _cards2array(raw['cur_hand'])[None, :],
                            np.array([raw['oppo_private_count']], dtype=np.int64),
                            obs_dim=expected_dim,
                            cur_turn=raw['cur_turn'],
                        )[0]
                        assert np.array_equal(st['obs'], ref)
                    else:
                        assert st['obs'][304 + int(raw['cur_turn'])] == 1.0
                        assert st['obs'][316 + int(raw['player_id'])] == 1.0
        feat = env.get_action_feature(0)
        assert feat.shape == (44,)
        assert feat.flags.writeable


def test_dmc_agents_step_batch():
    from rlcard.agents.dmc_agent.equivariant_model import EquivariantDMCAgent

    set_seed(29)
    env_v4 = rlcard.make('swy-blm', config={'seed': 29, 'obs_mode': 'v4'})
    eq_agents = [
        EquivariantDMCAgent(env_v4.state_shape[p], env_v4.action_shape[p], d_model=32, num_blocks=2, device='cpu')
        for p in range(2)
    ]
    for a in eq_agents:
        a.eval()

    states = [env_v4.reset()[0] for _ in range(8)]
    obs_b = np.stack([s['obs'] for s in states]).astype(np.float32)
    utils_b = eq_agents[0]._predict_batch_utilities(obs_b)
    for b, st in enumerate(states):
        keys, vals_single = eq_agents[0].predict(st)
        assert np.allclose(utils_b[b, keys], vals_single, atol=1e-5, rtol=1e-5)
    acts_batch = eq_agents[0].step_batch(states, rng=np.random.default_rng(1))
    assert all(a in s['legal_actions'] for a, s in zip(acts_batch, states))

    env_v2 = rlcard.make('swy-blm', config={'seed': 29, 'obs_mode': 'v2'})
    mlp_agent = DMCAgent(env_v2.state_shape[0], env_v2.action_shape[0], mlp_layers=[32, 32], device='cpu')
    mlp_agent.eval()
    states_v2 = [env_v2.reset()[0] for _ in range(8)]
    assert mlp_agent.eval_step_batch(states_v2) == [mlp_agent.eval_step(s)[0] for s in states_v2]


def test_batched_trainer_and_validate(tmp_path):
    from rlcard.agents.dmc_agent.equivariant_model import EquivariantDMCAgent
    from rlcard.agents.dmc_agent.utils import validate

    set_seed(29)
    env_v4 = rlcard.make('swy-blm', config={'seed': 29, 'obs_mode': 'v4'})
    eq_agents = [
        EquivariantDMCAgent(env_v4.state_shape[p], env_v4.action_shape[p], d_model=32, num_blocks=2, device='cpu')
        for p in range(2)
    ]
    for a in eq_agents:
        a.eval()

    res = validate(env_v4, eq_agents, RuleBasedAgent(), M=16)
    assert 'first_avg_winning_rate' in res and 'second_avg_winning_rate' in res

    savedir = str(tmp_path / 'batched_eq')
    trainer = DMCTrainer(
        env_v4,
        cuda='',
        actor_device='cpu',
        xpid='batched_eq_test',
        savedir=savedir,
        total_iterations=30,
        num_actors=2,
        actor_batch_size=8,
        training_device='cpu',
        batch_size=4,
        unroll_length=6,
        num_buffers=24,
        num_threads=1,
        sync_interval=5,
        d_model=32,
        num_blocks=2,
        snapshot_prob=0.15,
        rule_opponent_prob=0.15,
        snapshot_pool_size=2,
        snapshot_interval=10,
    )
    trainer.start()
    assert os.path.exists(os.path.join(savedir, 'batched_eq_test', 'model.tar'))

