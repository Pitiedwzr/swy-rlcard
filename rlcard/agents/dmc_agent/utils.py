# Copyright 2021 RLCard Team of Texas A&M University
# Copyright 2021 DouZero Team of Kwai
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import traceback

import numpy as np
import torch
import torch.optim as optim

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

def get_batch(
    free_queue,
    full_queue,
    buffers,
    batch_size,
    lock
):
    with lock:
        indices = [full_queue.get() for _ in range(batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def create_buffers(
    T,
    num_buffers,
    state_shape,
    action_shape,
    device_iterator,
    include_aux_targets=False,
    buffer_on_cpu=False,
):
    buffers = {}
    for device in device_iterator:
        buffers[device] = []
        for player_id in range(len(state_shape)):
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                state=dict(size=(T,)+tuple(state_shape[player_id]), dtype=torch.int8),
                action=dict(size=(T,)+tuple(action_shape[player_id]), dtype=torch.int8),
            )
            if include_aux_targets:
                specs['wdl_target'] = dict(size=(T,), dtype=torch.int64)
                specs['oppo_priv'] = dict(size=(T, 22), dtype=torch.int8)
            _buffers = {key: [] for key in specs}
            target_dev = 'cpu' if (device == "cpu" or buffer_on_cpu) else ('cuda:' + str(device))
            for _ in range(num_buffers):
                for key in _buffers:
                    _buffer = torch.empty(**specs[key]).to(target_dev).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device].append(_buffers)
    return buffers

def create_optimizers(
    num_players,
    learning_rate,
    momentum,
    epsilon,
    alpha,
    learner_model
):
    optimizers = []
    for player_id in range(num_players):
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(player_id),
            lr=learning_rate,
            momentum=momentum,
            eps=epsilon,
            alpha=alpha)
        optimizers.append(optimizer)
    return optimizers

def create_schedulers(
    optimizers,
    max_learning_rate,
    total_iterations
):
    schedulers = []
    for player_id in range(len(optimizers)):
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizers[player_id],
            max_learning_rate,
            total_iterations,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='linear')
        schedulers.append(scheduler)
    return schedulers

def _configure_actor_threads():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def act(
    i,
    device,
    T,
    free_queue,
    full_queue,
    model,
    buffers,
    env,
    snapshot_models=None,
    num_snapshots_ready=None,
    snapshot_prob=0.0,
    rule_opponent_prob=0.0,
    include_aux_targets=False,
):
    try:
        _configure_actor_threads()
        log.info('Device %s Actor %i started.', str(device), i)

        # Configure environment
        env.seed(i)
        current_agents = model.get_agents()
        env.set_agents(current_agents)

        rng = np.random.default_rng(10000 + i)
        rule_agent = None
        if rule_opponent_prob > 0:
            from rlcard.agents.rule_based_agent import RuleBasedAgent
            rule_agent = RuleBasedAgent()

        if include_aux_targets:
            from rlcard.games.bailongmen.utils import _cards2array

        done_buf = [[] for _ in range(env.num_players)]
        episode_return_buf = [[] for _ in range(env.num_players)]
        target_buf = [[] for _ in range(env.num_players)]
        state_buf = [[] for _ in range(env.num_players)]
        action_buf = [[] for _ in range(env.num_players)]
        wdl_target_buf = [[] for _ in range(env.num_players)]
        oppo_priv_buf = [[] for _ in range(env.num_players)]
        size = [0 for _ in range(env.num_players)]

        while True:
            train_players = range(env.num_players)
            if rule_opponent_prob > 0 or snapshot_prob > 0:
                r = float(rng.random())
                if r < rule_opponent_prob and rule_agent is not None:
                    train_p = int(rng.integers(0, env.num_players))
                    agents = [
                        current_agents[p] if p == train_p else rule_agent
                        for p in range(env.num_players)
                    ]
                    env.set_agents(agents)
                    train_players = (train_p,)
                elif (
                    r < rule_opponent_prob + snapshot_prob
                    and snapshot_models
                    and num_snapshots_ready is not None
                    and num_snapshots_ready.value > 0
                ):
                    train_p = int(rng.integers(0, env.num_players))
                    snap_idx = int(rng.integers(0, num_snapshots_ready.value))
                    snap_agents = snapshot_models[snap_idx].get_agents()
                    agents = [
                        current_agents[p] if p == train_p else snap_agents[p]
                        for p in range(env.num_players)
                    ]
                    env.set_agents(agents)
                    train_players = (train_p,)
                else:
                    env.set_agents(current_agents)

            trajectories, payoffs = env.run(is_training=True)  # run a full game
            score_diffs = env.get_score_diff() if include_aux_targets else payoffs
            for p in train_players:
                size[p] += len(trajectories[p][:-1]) // 2
                diff = size[p] - len(target_buf[p])
                if diff > 0:
                    ep_ret = float(score_diffs[p]) if include_aux_targets else float(payoffs[p])
                    done_buf[p].extend([False for _ in range(diff-1)])
                    done_buf[p].append(True)
                    episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                    episode_return_buf[p].append(ep_ret)
                    target_buf[p].extend([ep_ret for _ in range(diff)])
                    if include_aux_targets:
                        wdl_label = 2 if ep_ret > 0 else (1 if ep_ret == 0 else 0)
                        wdl_target_buf[p].extend([wdl_label for _ in range(diff)])
                    # State and action
                    for idx in range(0, len(trajectories[p])-2, 2):
                        state = trajectories[p][idx]['obs']
                        action = env.get_action_feature(trajectories[p][idx+1])
                        state_buf[p].append(torch.from_numpy(state))
                        action_buf[p].append(torch.from_numpy(action))
                        if include_aux_targets:
                            turn = int(trajectories[p][idx]['raw_obs']['cur_turn'])
                            oppo_priv_cards = env.game.get_oppo_private_at_turn(p, turn)
                            oppo_priv_buf[p].append(torch.from_numpy(_cards2array(oppo_priv_cards)).to(torch.int8))

                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    buffers[p]['done'][index].copy_(torch.as_tensor(done_buf[p][:T], dtype=torch.bool))
                    buffers[p]['episode_return'][index].copy_(torch.as_tensor(episode_return_buf[p][:T], dtype=torch.float32))
                    buffers[p]['target'][index].copy_(torch.as_tensor(target_buf[p][:T], dtype=torch.float32))
                    buffers[p]['state'][index].copy_(torch.stack(state_buf[p][:T]))
                    buffers[p]['action'][index].copy_(torch.stack(action_buf[p][:T]))
                    if include_aux_targets:
                        buffers[p]['wdl_target'][index].copy_(torch.as_tensor(wdl_target_buf[p][:T], dtype=torch.int64))
                        buffers[p]['oppo_priv'][index].copy_(torch.stack(oppo_priv_buf[p][:T]))
                        wdl_target_buf[p] = wdl_target_buf[p][T:]
                        oppo_priv_buf[p] = oppo_priv_buf[p][T:]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    state_buf[p] = state_buf[p][T:]
                    action_buf[p] = action_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


def act_batched(
    i,
    device,
    T,
    free_queue,
    full_queue,
    model,
    buffers,
    env,
    snapshot_models=None,
    num_snapshots_ready=None,
    snapshot_prob=0.0,
    rule_opponent_prob=0.0,
    include_aux_targets=False,
    actor_batch_size=64,
):
    try:
        _configure_actor_threads()
        import rlcard
        from rlcard.games.bailongmen.utils import _cards2array

        B = max(1, int(actor_batch_size))
        log.info('Device %s BatchedActor %i started (actor_batch_size=%d).', str(device), i, B)

        env_cfg = {'obs_mode': env.obs_mode, 'win_reward_weight': env.win_reward_weight}
        envs = [rlcard.make('swy-blm', config=env_cfg) for _ in range(B)]
        for b, e in enumerate(envs):
            e.seed((i + 1) * 10000 + b)

        current_agents = model.get_agents()
        rng = np.random.default_rng(10000 + i)
        rule_agent = None
        if rule_opponent_prob > 0:
            from rlcard.agents.rule_based_agent import RuleBasedAgent
            rule_agent = RuleBasedAgent()

        obs_dim = int(env.state_shape[0][0])
        act_dim = int(env.action_shape[0][0])
        action_eye = np.eye(act_dim, dtype=np.int8)

        done_buf = [[] for _ in range(2)]
        episode_return_buf = [[] for _ in range(2)]
        target_buf = [[] for _ in range(2)]
        state_buf = [[] for _ in range(2)]
        action_buf = [[] for _ in range(2)]
        wdl_target_buf = [[] for _ in range(2)]
        oppo_priv_buf = [[] for _ in range(2)]
        size = [0, 0]

        turn_obs = [np.empty((B, 6, obs_dim), dtype=np.int8), np.empty((B, 6, obs_dim), dtype=np.int8)]
        turn_act = [np.empty((B, 6), dtype=np.int64), np.empty((B, 6), dtype=np.int64)]

        use_mixed = (rule_opponent_prob > 0 or snapshot_prob > 0)

        while True:
            if not use_mixed:
                train_players_list = [(0, 1)] * B
                slot_modes = None
            else:
                train_players_list = []
                # mode per (b, p): 'current', 'rule', or ('snap', snap_idx)
                slot_modes = [['current', 'current'] for _ in range(B)]
                ready_snaps = num_snapshots_ready.value if (snapshot_models and num_snapshots_ready is not None) else 0
                for b in range(B):
                    r = float(rng.random())
                    if r < rule_opponent_prob and rule_agent is not None:
                        tp = int(rng.integers(0, 2))
                        slot_modes[b][1 - tp] = 'rule'
                        train_players_list.append((tp,))
                    elif r < rule_opponent_prob + snapshot_prob and ready_snaps > 0:
                        tp = int(rng.integers(0, 2))
                        snap_idx = int(rng.integers(0, ready_snaps))
                        slot_modes[b][1 - tp] = ('snap', snap_idx)
                        train_players_list.append((tp,))
                    else:
                        train_players_list.append((0, 1))

            states = [e.reset()[0] for e in envs]

            for turn in range(12):
                p = turn & 1
                step_k = turn >> 1
                if not use_mixed:
                    actions = current_agents[p].step_batch(states, rng=rng)
                    for b in range(B):
                        turn_obs[p][b, step_k] = states[b]['obs']
                        turn_act[p][b, step_k] = actions[b]
                    states = [envs[b].step(actions[b], False)[0] for b in range(B)]
                else:
                    cur_idx = []
                    rule_idx = []
                    snap_groups = {}
                    for b in range(B):
                        mode = slot_modes[b][p]
                        if mode == 'current':
                            cur_idx.append(b)
                        elif mode == 'rule':
                            rule_idx.append(b)
                        else:
                            snap_groups.setdefault(mode[1], []).append(b)

                    next_states = [None] * B
                    if cur_idx:
                        cur_states = [states[b] for b in cur_idx]
                        cur_acts = current_agents[p].step_batch(cur_states, rng=rng)
                        for b, act_id in zip(cur_idx, cur_acts):
                            turn_obs[p][b, step_k] = states[b]['obs']
                            turn_act[p][b, step_k] = act_id
                            next_states[b] = envs[b].step(act_id, False)[0]
                    for b in rule_idx:
                        raw_act = rule_agent.step(states[b])
                        next_states[b] = envs[b].step(raw_act, True)[0]
                    for s_idx, b_list in snap_groups.items():
                        snap_agent = snapshot_models[s_idx].get_agent(p)
                        s_states = [states[b] for b in b_list]
                        if hasattr(snap_agent, 'step_batch'):
                            s_acts = snap_agent.step_batch(s_states, rng=rng)
                        else:
                            s_acts = [snap_agent.step(st) for st in s_states]
                        for b, act_id in zip(b_list, s_acts):
                            next_states[b] = envs[b].step(act_id, False)[0]
                    states = next_states

            for e in envs:
                e.is_over()

            order = rng.permutation(B)
            for b in order:
                e = envs[b]
                score_diffs = e.get_score_diff() if include_aux_targets else e.get_payoffs()
                for p in train_players_list[b]:
                    ep_ret = float(score_diffs[p])
                    done_buf[p].extend((False, False, False, False, False, True))
                    episode_return_buf[p].extend((0.0, 0.0, 0.0, 0.0, 0.0, ep_ret))
                    target_buf[p].extend((ep_ret, ep_ret, ep_ret, ep_ret, ep_ret, ep_ret))
                    if include_aux_targets:
                        wdl_label = 2 if ep_ret > 0 else (1 if ep_ret == 0 else 0)
                        wdl_target_buf[p].extend((wdl_label, wdl_label, wdl_label, wdl_label, wdl_label, wdl_label))
                    for step_k in range(6):
                        state_buf[p].append(turn_obs[p][b, step_k])
                        action_buf[p].append(action_eye[turn_act[p][b, step_k]])
                        if include_aux_targets:
                            mv_turn = step_k * 2 + p
                            oppo_priv_cards = e.game.get_oppo_private_at_turn(p, mv_turn)
                            oppo_priv_buf[p].append(_cards2array(oppo_priv_cards).astype(np.int8))
                    size[p] += 6

                    while size[p] > T:
                        index = free_queue[p].get()
                        if index is None:
                            return
                        buffers[p]['done'][index].copy_(torch.as_tensor(done_buf[p][:T], dtype=torch.bool))
                        buffers[p]['episode_return'][index].copy_(torch.as_tensor(episode_return_buf[p][:T], dtype=torch.float32))
                        buffers[p]['target'][index].copy_(torch.as_tensor(target_buf[p][:T], dtype=torch.float32))
                        buffers[p]['state'][index].copy_(torch.from_numpy(np.stack(state_buf[p][:T])))
                        buffers[p]['action'][index].copy_(torch.from_numpy(np.stack(action_buf[p][:T])))
                        if include_aux_targets:
                            buffers[p]['wdl_target'][index].copy_(torch.as_tensor(wdl_target_buf[p][:T], dtype=torch.int64))
                            buffers[p]['oppo_priv'][index].copy_(torch.from_numpy(np.stack(oppo_priv_buf[p][:T])))
                            del wdl_target_buf[p][:T]
                            del oppo_priv_buf[p][:T]
                        full_queue[p].put(index)
                        del done_buf[p][:T]
                        del episode_return_buf[p][:T]
                        del target_buf[p][:T]
                        del state_buf[p][:T]
                        del action_buf[p][:T]
                        size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in batched worker process %i', i)
        traceback.print_exc()
        print()
        raise e


def _run_swy_blm_eval_batched(env, p0_agent, p1_agent, M):
    import rlcard

    batch_size = min(256, M)
    env_cfg = {'obs_mode': env.obs_mode, 'win_reward_weight': env.win_reward_weight}
    envs = [rlcard.make('swy-blm', config=env_cfg) for _ in range(batch_size)]
    scores = np.zeros((M, 2), dtype=np.float64)

    p0_batched = hasattr(p0_agent, 'eval_step_batch')
    p1_batched = hasattr(p1_agent, 'eval_step_batch')

    done = 0
    while done < M:
        chunk = min(batch_size, M - done)
        for b in range(chunk):
            envs[b].seed(int(env.np_random.randint(0, 2**31 - 1)))
        states = [envs[b].reset()[0] for b in range(chunk)]
        for turn in range(12):
            agent = p0_agent if (turn & 1) == 0 else p1_agent
            is_batched = p0_batched if (turn & 1) == 0 else p1_batched
            if is_batched:
                actions = agent.eval_step_batch(states)
                use_raw = False
            else:
                use_raw = getattr(agent, 'use_raw', False)
                actions = [agent.eval_step(s)[0] for s in states]
            states = [envs[b].step(actions[b], use_raw)[0] for b in range(chunk)]
        for b in range(chunk):
            envs[b].is_over()
            scores[done + b] = envs[b].game.final_score
        done += chunk
    return scores


@torch.no_grad()
def validate(env, agents, baseline_agent, M=1000):
    res = {}

    can_batch = (
        getattr(env, 'name', None) == 'swy-blm'
        and M >= 4
        and (hasattr(agents[0], 'eval_step_batch') or hasattr(agents[1], 'eval_step_batch'))
    )

    if can_batch:
        first_score = _run_swy_blm_eval_batched(env, agents[0], baseline_agent, M)
        second_score = _run_swy_blm_eval_batched(env, baseline_agent, agents[1], M)
    else:
        # test first hand agent
        env.set_agents([agents[0], baseline_agent])
        first_score = np.zeros((M, 2))
        for i in range(M):
            env.run(is_training=False)
            first_score[i] = env.game.final_score

        # test second hand agent
        env.set_agents([baseline_agent, agents[1]])
        second_score = np.zeros((M, 2))
        for i in range(M):
            env.run(is_training=False)
            second_score[i] = env.game.final_score

    res['first_avg_score'] = float(first_score[:, 0].mean())
    res['first_avg_winning_score'] = float((first_score[:, 0] - first_score[:, 1]).mean())
    res['first_avg_winning_rate'] = float((first_score[:, 0] > first_score[:, 1]).mean())
    res['first_avg_losing_rate'] = float((first_score[:, 0] < first_score[:, 1]).mean())

    res['second_avg_score'] = float(second_score[:, 1].mean())
    res['second_avg_winning_score'] = float((second_score[:, 1] - second_score[:, 0]).mean())
    res['second_avg_winning_rate'] = float((second_score[:, 1] > second_score[:, 0]).mean())
    res['second_avg_losing_rate'] = float((second_score[:, 1] < second_score[:, 0]).mean())

    return res
