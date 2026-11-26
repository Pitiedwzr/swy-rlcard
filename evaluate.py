''' An example of evluating the trained models in RLCard
'''
import os
import argparse

import numpy as np
import torch

import rlcard
from rlcard.agents.dmc_agent.equivariant_model import EquivariantDMCAgent
from rlcard.agents.dmc_agent.model import DMCAgent
from rlcard.agents.dmc_search_agent import DMCSearchAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.rule_based_agent import RuleBasedAgent


def load_dmc_agents(model_path, device="cpu", sym_avg_k=1):
    """Load MLP (230/264-dim) or Equivariant (318-dim) DMC agents with auto-detection."""
    ckpt_states = torch.load(model_path, map_location=device, weights_only=False)
    p0_state = ckpt_states['model_state_dict'][0]
    arch = ckpt_states.get('arch')
    obs_mode = ckpt_states.get('obs_mode')
    model_cfg = ckpt_states.get('model_config', {})

    if arch is None:
        if 'reg_proj.0.weight' in p0_state:
            arch = 'equivariant'
            obs_mode = obs_mode or 'v4'
        else:
            arch = 'mlp'
            first_weight = p0_state.get('fc_layers.0.weight')
            if obs_mode is None:
                obs_mode = 'legacy' if (first_weight is not None and int(first_weight.shape[1]) == 274) else 'v2'

    env = rlcard.make('swy-blm', config={'obs_mode': obs_mode})
    base_agents = []
    if arch == 'equivariant':
        d_model = int(model_cfg.get('d_model', 128))
        num_blocks = int(model_cfg.get('num_blocks', 3))
        margin_weight = float(model_cfg.get('margin_weight', 0.02))
        for p in range(env.num_players):
            agent = EquivariantDMCAgent(
                env.state_shape[p],
                env.action_shape[p],
                d_model=d_model,
                num_blocks=num_blocks,
                margin_weight=margin_weight,
                device=device,
            )
            agent.load_state_dict(ckpt_states['model_state_dict'][p])
            agent.eval()
            base_agents.append(agent)
    else:
        mlp_layers = list(model_cfg.get('mlp_layers', [256, 256, 256, 128]))
        for p in range(env.num_players):
            agent = DMCAgent(
                env.state_shape[p],
                env.action_shape[p],
                mlp_layers=mlp_layers,
                device=device,
                sym_avg_k=sym_avg_k,
            )
            agent.load_state_dict(ckpt_states['model_state_dict'][p])
            agent.eval()
            base_agents.append(agent)

    return env, base_agents, ckpt_states


@torch.no_grad()
def validate(env, agents, baseline_agent, M=1000):
    res = {}

    # test first hand agent
    env.set_agents([agents[0], baseline_agent])
    first_score = np.zeros((M, 2))
    for i in range(M):
        env.run(is_training=False)
        first_score[i] = env.game.final_score
    
    res['first_avg_score'] = float(first_score[:, 0].mean())
    res['first_avg_winning_score'] = float((first_score[:, 0] - first_score[:, 1]).mean())
    res['first_avg_winning_rate'] = float((first_score[:, 0] > first_score[:, 1]).mean())
    res['first_avg_losing_rate'] = float((first_score[:, 0] < first_score[:, 1]).mean())

    # test second hand agent
    env.set_agents([baseline_agent, agents[1]])
    second_score = np.zeros((M, 2))
    for i in range(M):
        env.run(is_training=False)
        second_score[i] = env.game.final_score

    res['second_avg_score'] = float(second_score[:, 1].mean())
    res['second_avg_winning_score'] = float((second_score[:, 1] - second_score[:, 0]).mean())
    res['second_avg_winning_rate'] = float((second_score[:, 1] > second_score[:, 0]).mean())
    res['second_avg_losing_rate'] = float((second_score[:, 1] < second_score[:, 0]).mean())
    
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate agent against baselines")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--ai_agent', type=str, required=True)
    parser.add_argument('--baseline_agent', type=str, default="random")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--total_simulations', type=int, default=5000)
    parser.add_argument('--human_order', type=int, choices=[1, 2], default=None)
    parser.add_argument('--sym_avg_k', type=int, default=1, help='Number of symmetries averaged at inference for MLP models')
    args = parser.parse_args()

    DEVICE = "cuda:0" if args.cuda else "cpu"

    ## Prepare environment and agents
    if args.ai_agent in ('dmc', 'dmc_search'):
        env, base_agents, _ = load_dmc_agents(args.model_path, device=DEVICE, sym_avg_k=args.sym_avg_k)
        if args.ai_agent == 'dmc_search':
            ai_agents = [DMCSearchAgent(base_agents, p) for p in range(env.num_players)]
        else:
            ai_agents = base_agents
    else:
        raise NotImplementedError(f"Unsupported ai_agent: {args.ai_agent}")

    print("AI model version: {} at {}".format(ai_agents[0].__class__, args.model_path))
    
    # Prepare baseline agents
    if args.baseline_agent == "random":
        baseline_agent = RandomAgent()
    elif args.baseline_agent == "rule-based":
        baseline_agent = RuleBasedAgent()
    else:
        raise NotImplementedError
    
    res = validate(env, ai_agents, baseline_agent, M=args.total_simulations)
    print(res)