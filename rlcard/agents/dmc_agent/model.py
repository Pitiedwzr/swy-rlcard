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

import numpy as np

import torch
from torch import nn

def _normalize_device(device):
    if isinstance(device, torch.device):
        return str(device)
    device_str = str(device)
    if device_str == "cpu" or device_str.startswith("cuda"):
        return device_str
    return f"cuda:{device_str}"

class DMCNet(nn.Module):
    def __init__(
        self,
        state_shape,
        action_shape,
        mlp_layers=[512,512,512,512,512]
    ):
        super().__init__()
        input_dim = int(np.prod(state_shape) + np.prod(action_shape))
        layer_dims = [input_dim] + mlp_layers
        fc = []
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(layer_dims[-1], 1))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, obs, actions):
        obs = torch.flatten(obs, 1)
        actions = torch.flatten(actions, 1)
        x = torch.cat((obs, actions), dim=1)
        values = self.fc_layers(x).flatten()
        return values

class DMCAgent:
    def __init__(
        self,
        state_shape,
        action_shape,
        mlp_layers=[512,512,512,512,512],
        exp_epsilon=0.01,
        device="0",
        sym_avg_k=1,
    ):
        self.use_raw = False
        self.device = _normalize_device(device)
        self.net = DMCNet(state_shape, action_shape, mlp_layers).to(self.device)
        self.exp_epsilon = exp_epsilon
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.sym_avg_k = max(1, int(sym_avg_k))
        self._sym_tables = None
        self._sym_indices = None

    def step(self, state):
        action_keys, values = self.predict(state)

        if self.exp_epsilon > 0 and np.random.rand() < self.exp_epsilon:
            action = np.random.choice(action_keys)
        else:
            action_idx = np.argmax(values)
            action = action_keys[action_idx]

        return action

    def eval_step(self, state):
        action_keys, values = self.predict(state)

        action_idx = np.argmax(values)
        action = action_keys[action_idx]

        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(values[i]) for i in range(len(action_keys))}

        return action, info

    def share_memory(self):
        self.net.share_memory()

    def eval(self):
        self.net.eval()

    def parameters(self):
        return self.net.parameters()

    def _get_sym_tables(self, obs_dim):
        if self._sym_tables is None:
            from .symmetry import build_swy_blm_symmetry_tables
            obs_perms, act_perms = build_swy_blm_symmetry_tables(obs_dim=obs_dim)
            k = min(self.sym_avg_k, int(obs_perms.shape[0]))
            if k >= int(obs_perms.shape[0]):
                idx = torch.arange(obs_perms.shape[0], dtype=torch.long)
            else:
                rng = np.random.default_rng(0)
                idx = torch.from_numpy(rng.choice(obs_perms.shape[0], size=k, replace=False)).long()
            self._sym_tables = (
                obs_perms[idx].to(self.device),
                act_perms[idx].to(self.device),
            )
        return self._sym_tables

    def predict(self, state):
        # Prepare obs and actions
        obs = state['obs'].astype(np.float32)
        legal_actions = state['legal_actions']
        action_keys = np.array(list(legal_actions.keys()))
        action_values = list(legal_actions.values())
        # One-hot encoding if there is no action features
        for i in range(len(action_values)):
            if action_values[i] is None:
                action_values[i] = np.zeros(self.action_shape[0])
                action_values[i][action_keys[i]] = 1
        action_values = np.array(action_values, dtype=np.float32)

        obs_dim = int(obs.shape[0])
        if self.sym_avg_k > 1 and obs_dim in (230, 264) and int(action_values.shape[1]) == 44:
            obs_perms, act_perms = self._get_sym_tables(obs_dim)
            k = int(obs_perms.shape[0])
            num_a = len(action_keys)
            obs_t = torch.from_numpy(obs).to(self.device)
            act_t = torch.from_numpy(action_values).to(self.device)
            obs_sym = obs_t[obs_perms].unsqueeze(1).expand(-1, num_a, -1).reshape(k * num_a, obs_dim)
            act_sym = act_t[:, act_perms].permute(1, 0, 2).reshape(k * num_a, 44)
            with torch.no_grad():
                values = self.net.forward(obs_sym, act_sym).reshape(k, num_a).mean(dim=0)
            return action_keys, values.cpu().numpy()

        obs = np.repeat(obs[np.newaxis, :], len(action_keys), axis=0)

        # Predict Q values
        with torch.no_grad():
            values = self.net.forward(torch.from_numpy(obs).to(self.device),
                                      torch.from_numpy(action_values).to(self.device))

        return action_keys, values.cpu().numpy()

    @torch.no_grad()
    def eval_step_batch(self, states):
        if self.sym_avg_k > 1:
            return [self.eval_step(s)[0] for s in states]
        obs_list = []
        act_list = []
        keys_list = []
        counts = []
        act_dim = int(self.action_shape[0])
        for s in states:
            legal_actions = s['legal_actions']
            keys = list(legal_actions.keys())
            keys_list.append(keys)
            counts.append(len(keys))
            obs = s['obs'].astype(np.float32, copy=False)
            obs_list.append(np.broadcast_to(obs, (len(keys), obs.shape[0])))
            for k in keys:
                v = legal_actions[k]
                if v is None:
                    v = np.zeros(act_dim, dtype=np.float32)
                    v[k] = 1.0
                act_list.append(v)
        all_obs = np.concatenate(obs_list, axis=0)
        all_acts = np.asarray(act_list, dtype=np.float32)
        values = self.net.forward(
            torch.from_numpy(all_obs).to(self.device),
            torch.from_numpy(all_acts).to(self.device),
        ).cpu().numpy()
        actions = []
        offset = 0
        for keys, n in zip(keys_list, counts):
            best_idx = int(np.argmax(values[offset:offset + n]))
            actions.append(int(keys[best_idx]))
            offset += n
        return actions

    def step_batch(self, states, rng=None):
        actions = self.eval_step_batch(states)
        if self.exp_epsilon > 0:
            rand_fn = rng.random if rng is not None else np.random.random
            choice_fn = rng.choice if rng is not None else np.random.choice
            for b, s in enumerate(states):
                if rand_fn() < self.exp_epsilon:
                    actions[b] = int(choice_fn(list(s['legal_actions'].keys())))
        return actions

    def forward(self, obs, actions):
        return self.net.forward(obs, actions)

    def load_state_dict(self, state_dict):
        return self.net.load_state_dict(state_dict)

    def state_dict(self):
        return self.net.state_dict()

    def set_device(self, device):
        self.device = _normalize_device(device)
        self.net.to(self.device)
        self._sym_tables = None

class DMCModel:
    def __init__(
        self,
        state_shape,
        action_shape,
        mlp_layers=[512,512,512,512,512],
        exp_epsilon=0.01,
        device=0
    ):
        self.agents = []
        for player_id in range(len(state_shape)):
            agent = DMCAgent(
                state_shape[player_id],
                action_shape[player_id],
                mlp_layers,
                exp_epsilon,
                device
            )
            self.agents.append(agent)

    def share_memory(self):
        for agent in self.agents:
            agent.share_memory()

    def eval(self):
        for agent in self.agents:
            agent.eval()

    def parameters(self, index):
        return self.agents[index].parameters()

    def get_agent(self, index):
        return self.agents[index]

    def get_agents(self):
        return self.agents
