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
            for _ in range(num_buffers):
                for key in _buffers:
                    if device == "cpu":
                        _buffer = torch.empty(**specs[key]).to('cpu').share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to('cuda:'+str(device)).share_memory_()
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
