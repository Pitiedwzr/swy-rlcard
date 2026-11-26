from collections import Counter, OrderedDict
import numpy as np

from rlcard.envs import Env
from rlcard.games.bailongmen.game import SwyBlmGame
from rlcard.games.bailongmen.utils import ACTION_2_ID, DECK, ID_2_ACTION
from rlcard.games.bailongmen.utils import _cards2array, _cards2table


CARD_TO_IDX = {c: i for i, c in enumerate(DECK)}
OBS_V4_DIM = 318


def one_hot(i, n):
    array = np.zeros(int(n))
    array[int(i)] = 1.
    return array
        
        
class SwyBlmEnv(Env):
    ''' Swy-blm Environment
    '''

    def __init__(self, config):
        self.name = 'swy-blm'
        self.game = SwyBlmGame()
        super().__init__(config)
        self.legacy_obs = bool(config.get('legacy_obs', False))
        obs_mode = config.get('obs_mode', None)
        if obs_mode is None:
            obs_mode = 'legacy' if self.legacy_obs else 'v2'
        if obs_mode not in ('legacy', 'v2', 'v4'):
            raise ValueError(f"Unsupported swy-blm obs_mode: {obs_mode}")
        self.obs_mode = obs_mode
        self.legacy_obs = (self.obs_mode == 'legacy')
        self.win_reward_weight = float(config.get('win_reward_weight', 0.0))
        if self.obs_mode == 'legacy':
            self.state_shape = [[230], [230]]
        elif self.obs_mode == 'v2':
            self.state_shape = [[264], [264]]
        else:
            self.state_shape = [[OBS_V4_DIM], [OBS_V4_DIM]]
        self.action_shape = [[44], [44]]
        
    def _extract_state(self, state):
        ''' Encode state

        Args:
            state (dict): dict of original state
        '''
        
        # encode cards on different sections
        board_public = _cards2array(state['board_public'])
        cur_public = _cards2array(state['cur_public'])
        cur_private = _cards2array(state['cur_private'])
        oppo_public = _cards2array(state['oppo_public'])
        cur_hand = _cards2array(state['cur_hand'])
        
        # encode empty space
        cur_public_space = one_hot(3 - len(state['cur_public']), 4)
        cur_private_space = one_hot(3 - len(state['cur_private']), 4)
        oppo_public_space = one_hot(3 - len(state['oppo_public']), 4)
        oppo_private_space = one_hot(3 - state['oppo_private_count'], 4)
        
        # encode score table
        cur_table, cur_joker = _cards2table(state['board_public'] + state['cur_public'] + state['oppo_public'] + state['cur_private'])
        oppo_table, oppo_joker = _cards2table(state['board_public'] + state['cur_public'] + state['oppo_public'])
       
        cur_joker_count = one_hot(cur_joker.sum(), 3)
        cur_suit_count = np.zeros(4 * 6)
        for i in range(4):  # four different suits
            cur_suit_count[i*6: (i+1)*6] = one_hot(cur_table[:, i].sum(), 6)
        cur_color_count = np.zeros(5 * 5)
        for i in range(5):  # five different colors
            cur_color_count[i*5: (i+1)*5] = one_hot(cur_table[i, :].sum(), 5)
            
        oppo_joker_count = one_hot(oppo_joker.sum(), 3)
        oppo_suit_count = np.zeros(4 * 6)
        for i in range(4):  # four different suits
            oppo_suit_count[i*6: (i+1)*6] = one_hot(oppo_table[:, i].sum(), 6)
        oppo_color_count = np.zeros(5 * 5)
        for i in range(5):  # five different colors
            oppo_color_count[i*5: (i+1)*5] = one_hot(oppo_table[i, :].sum(), 5)
            
        if self.obs_mode == 'legacy':
            obs = np.concatenate((board_public, cur_public, cur_private, oppo_public, cur_hand,
                                  cur_public_space, cur_private_space, oppo_public_space, oppo_private_space,
                                  cur_joker_count, cur_suit_count, cur_color_count,
                                  oppo_joker_count, oppo_suit_count, oppo_color_count))
        elif self.obs_mode == 'v2':
            seen_cards = np.clip(board_public + cur_public + cur_private + oppo_public + cur_hand, 0.0, 1.0)
            unseen_cards = 1.0 - seen_cards
            cur_turn = np.zeros(12)
            turn_idx = int(state.get('cur_turn', -1))
            if 0 <= turn_idx < 12:
                cur_turn[turn_idx] = 1.0
            obs = np.concatenate((board_public, cur_public, cur_private, oppo_public, cur_hand,
                                  cur_public_space, cur_private_space, oppo_public_space, oppo_private_space,
                                  cur_joker_count, cur_suit_count, cur_color_count,
                                  oppo_joker_count, oppo_suit_count, oppo_color_count,
                                  unseen_cards, cur_turn))
        else:
            seen_cards = np.clip(board_public + cur_public + cur_private + oppo_public + cur_hand, 0.0, 1.0)
            unseen_cards = 1.0 - seen_cards

            oppo_pub_by_step = np.zeros(6 * 22, dtype=np.float32)
            oppo_step_priv = np.zeros(6, dtype=np.float32)
            oppo_step_pub = np.zeros(6, dtype=np.float32)
            for k, (card, _, sec) in enumerate(state.get('oppo_history', [])[:6]):
                if sec == 0:
                    oppo_step_priv[k] = 1.0
                else:
                    oppo_step_pub[k] = 1.0
                    if card in CARD_TO_IDX:
                        oppo_pub_by_step[k * 22 + CARD_TO_IDX[card]] = 1.0

            cur_step_priv = np.zeros(6, dtype=np.float32)
            cur_step_pub = np.zeros(6, dtype=np.float32)
            for k, (_, _, sec) in enumerate(state.get('cur_history', [])[:6]):
                if sec == 0:
                    cur_step_priv[k] = 1.0
                else:
                    cur_step_pub[k] = 1.0

            cur_turn = np.zeros(12, dtype=np.float32)
            turn_idx = int(state.get('cur_turn', -1))
            if 0 <= turn_idx < 12:
                cur_turn[turn_idx] = 1.0
            player_oh = one_hot(int(state.get('player_id', 0)), 2)

            obs = np.concatenate((
                board_public, cur_public, cur_private, oppo_public, cur_hand, unseen_cards,
                oppo_pub_by_step,
                oppo_step_priv, oppo_step_pub,
                cur_step_priv, cur_step_pub,
                cur_public_space, cur_private_space, oppo_public_space, oppo_private_space,
                cur_turn, player_oh,
            ))
        
        # encode legal actions
        raw_legal_actions = self._get_legal_actions(state)
        legal_actions = {ACTION_2_ID[x]: one_hot(ACTION_2_ID[x], len(ID_2_ACTION)) for x in raw_legal_actions}

        extracted_state = OrderedDict({'obs': obs, 'legal_actions': legal_actions})
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = raw_legal_actions
        extracted_state['action_record'] = self.action_recorder
        return extracted_state
            
    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''
        score_diff = float(self.game.final_score[0] - self.game.final_score[1])
        reward = self.win_reward_weight * float(np.sign(score_diff)) + score_diff
        return np.array([reward, -reward], dtype=np.float32)

    def get_score_diff(self):
        '''Return unshaped final score differential [p0 - p1, p1 - p0].'''
        score_diff = float(self.game.final_score[0] - self.game.final_score[1])
        return np.array([score_diff, -score_diff], dtype=np.float32)

    def _decode_action(self, action_id):
        ''' Action id -> the action in the game. Must be implemented in the child class.

        Args:
            action_id (int): the id of the action

        Returns:
            action (string): the action that will be passed to the game engine.
        '''
        return ID_2_ACTION[action_id]

    def _get_legal_actions(self, state):
        ''' Get all legal actions for current state

        Returns:
            legal_actions (list): a list of legal actions' id
        '''
        legal_actions = self.game.compute_legal_actions(state['cur_hand'], len(state['cur_public']), len(state['cur_private']))
        return legal_actions

    def get_action_feature(self, action):
        ''' For some environments such as DouDizhu, we can have action features
        Returns:
            (numpy.array): The action features
        '''
        return one_hot(action, len(ID_2_ACTION))
        