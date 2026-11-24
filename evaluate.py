''' An example of evluating the trained models in RLCard
'''
import os
import argparse

import numpy as np
import torch

import rlcard
from rlcard.agents.dmc_agent.model import DMCAgent
from rlcard.agents.dmc_search_agent import DMCSearchAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.rule_based_agent import RuleBasedAgent


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
    args = parser.parse_args()

    DEVICE = "cuda:0" if args.cuda else "cpu"

    ckpt_states = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    first_weight = ckpt_states['model_state_dict'][0].get('fc_layers.0.weight')
    legacy_obs = (first_weight is not None and int(first_weight.shape[1]) == 274)

    ## Make environment
    env = rlcard.make('swy-blm', config={'legacy_obs': legacy_obs})

    ## Prepare agents
    if args.ai_agent in ('dmc', 'dmc_search'):
        base_agents = []
        for p in range(env.num_players):
            base_agents.append(DMCAgent(env.state_shape[p], env.action_shape[p], mlp_layers=[256, 256, 256, 128], device=DEVICE))

        # load trained ai agents
        for p in range(env.num_players):
            base_agents[p].load_state_dict(ckpt_states['model_state_dict'][p])

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