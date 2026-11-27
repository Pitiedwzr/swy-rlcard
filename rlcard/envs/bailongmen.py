from collections import Counter, OrderedDict
import numpy as np

from rlcard.envs import Env
from rlcard.games.bailongmen.game import SwyBlmGame
from rlcard.games.bailongmen.utils import ACTION_2_ID, CARD_TO_IDX, DECK, ID_2_ACTION
from rlcard.games.bailongmen.utils import _cards2array


OBS_V4_DIM = 318
_SPACE_OH = np.eye(4, dtype=np.float32)
_ACTION_OH = np.eye(len(ID_2_ACTION), dtype=np.float32)


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

        if self.obs_mode == 'v4':
            obs = np.zeros(OBS_V4_DIM, dtype=np.float32)
            obs[0:22] = board_public
            obs[22:44] = cur_public
            obs[44:66] = cur_private
            obs[66:88] = oppo_public
            obs[88:110] = cur_hand
            seen_cards = np.clip(board_public + cur_public + cur_private + oppo_public + cur_hand, 0.0, 1.0)
            obs[110:132] = 1.0 - seen_cards

            for k, (card, _, sec) in enumerate(state.get('oppo_history', [])[:6]):
                if sec == 0:
                    obs[264 + k] = 1.0
                else:
                    obs[270 + k] = 1.0
                    idx = CARD_TO_IDX.get(card)
                    if idx is not None:
                        obs[132 + k * 22 + idx] = 1.0

            for k, (_, _, sec) in enumerate(state.get('cur_history', [])[:6]):
                if sec == 0:
                    obs[276 + k] = 1.0
                else:
                    obs[282 + k] = 1.0

            obs[288:292] = _SPACE_OH[3 - len(state['cur_public'])]
            obs[292:296] = _SPACE_OH[3 - len(state['cur_private'])]
            obs[296:300] = _SPACE_OH[3 - len(state['oppo_public'])]
            obs[300:304] = _SPACE_OH[3 - state['oppo_private_count']]
            turn_idx = int(state.get('cur_turn', -1))
            if 0 <= turn_idx < 12:
                obs[304 + turn_idx] = 1.0
            obs[316 + int(state.get('player_id', 0))] = 1.0
        else:
            # encode empty space
            cur_public_space = one_hot(3 - len(state['cur_public']), 4)
            cur_private_space = one_hot(3 - len(state['cur_private']), 4)
            oppo_public_space = one_hot(3 - len(state['oppo_public']), 4)
            oppo_private_space = one_hot(3 - state['oppo_private_count'], 4)

            # encode score table
            oppo_arr = np.clip(board_public + cur_public + oppo_public, 0.0, 1.0)
            cur_arr = np.clip(oppo_arr + cur_private, 0.0, 1.0)
            cur_table, cur_joker = cur_arr[:20].reshape((5, 4)), cur_arr[20:]
            oppo_table, oppo_joker = oppo_arr[:20].reshape((5, 4)), oppo_arr[20:]

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
            else:
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

        # encode legal actions
        raw_legal_actions = self._get_legal_actions(state)
        legal_actions = {ACTION_2_ID[x]: _ACTION_OH[ACTION_2_ID[x]] for x in raw_legal_actions}

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
        return _ACTION_OH[int(action)].copy()
        