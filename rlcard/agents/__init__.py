from rlcard.agents.cfr_agent import CFRAgent
from rlcard.agents.human_agent import HumanAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.rule_based_agent import RuleBasedAgent

try:
    import torch  # noqa: F401
    from rlcard.agents.dqn_agent import DQNAgent
    from rlcard.agents.nfsp_agent import NFSPAgent
except ImportError:
    pass
