import itertools
import numpy as np
import torch

from rlcard.games.bailongmen.utils import DECK, ID_2_ACTION, ACTION_2_ID

CARD_TO_IDX = {c: i for i, c in enumerate(DECK)}
ACTION_EYE = np.eye(44, dtype=np.float32)

# Precompute score tables for fast vectorized scoring
# A card mask/vector of length 22 -> regular (5, 4), joker (2,)
def _compute_scores_batch(card_vecs):
    """Compute bailongmen table scores for a batch of binary card vectors (N, 22)."""
    regular = card_vecs[:, :20].reshape(-1, 5, 4)
    joker = card_vecs[:, 20:]

    suit_counts = regular.sum(axis=1)  # (N, 4)
    color_counts = regular.sum(axis=2)  # (N, 5)
    joker_counts = joker.sum(axis=1)  # (N,)

    score = np.zeros(card_vecs.shape[0], dtype=np.float32)
    score += ((suit_counts == 5) * 8 + (suit_counts == 4) * 4).sum(axis=1)
    score += ((color_counts[:, :3] == 4) * 3 + (color_counts[:, :3] == 3) * 1).sum(axis=1)
    score += ((color_counts[:, 3:5] == 4) * 6 + (color_counts[:, 3:5] == 3) * 3).sum(axis=1)
    score += (joker_counts == 2) * 5
    return score


def _extract_obs_batch(board_pub, cur_pub, cur_priv, oppo_pub, cur_hand, oppo_priv_count, obs_dim=230, cur_turn=None):
    """Vectorized equivalent of SwyBlmEnv._extract_state for (N, 22) binary vectors."""
    n = board_pub.shape[0]
    obs = np.zeros((n, obs_dim), dtype=np.float32)

    cur_pub_cnt = cur_pub.sum(axis=1).astype(np.int64)
    cur_priv_cnt = cur_priv.sum(axis=1).astype(np.int64)
    oppo_pub_cnt = oppo_pub.sum(axis=1).astype(np.int64)

    if obs_dim == 318:
        obs[:, 0:22] = board_pub
        obs[:, 22:44] = cur_pub
        obs[:, 44:66] = cur_priv
        obs[:, 66:88] = oppo_pub
        obs[:, 88:110] = cur_hand
        seen = np.clip(board_pub + cur_pub + cur_priv + oppo_pub + cur_hand, 0.0, 1.0)
        obs[:, 110:132] = 1.0 - seen
        idx = np.arange(n)
        obs[idx, 288 + (3 - cur_pub_cnt)] = 1.0
        obs[idx, 292 + (3 - cur_priv_cnt)] = 1.0
        obs[idx, 296 + (3 - oppo_pub_cnt)] = 1.0
        obs[idx, 300 + (3 - oppo_priv_count)] = 1.0
        if cur_turn is not None and 0 <= int(cur_turn) < 12:
            obs[:, 304 + int(cur_turn)] = 1.0
            obs[:, 316 + (int(cur_turn) % 2)] = 1.0
        return obs

    obs[:, 0:22] = board_pub
    obs[:, 22:44] = cur_pub
    obs[:, 44:66] = cur_priv
    obs[:, 66:88] = oppo_pub
    obs[:, 88:110] = cur_hand

    idx = np.arange(n)
    obs[idx, 110 + (3 - cur_pub_cnt)] = 1.0
    obs[idx, 114 + (3 - cur_priv_cnt)] = 1.0
    obs[idx, 118 + (3 - oppo_pub_cnt)] = 1.0
    obs[idx, 122 + (3 - oppo_priv_count)] = 1.0

    all_pub = board_pub + cur_pub + oppo_pub
    cur_mine = all_pub + cur_priv

    # cur counts (offset 126..177: 3 + 24 + 25 = 52)
    cur_reg = cur_mine[:, :20].reshape(n, 5, 4)
    cur_jok = cur_mine[:, 20:].sum(axis=1).astype(np.int64)
    obs[idx, 126 + cur_jok] = 1.0
    cur_suits = cur_reg.sum(axis=1).astype(np.int64)  # (N, 4)
    for s in range(4):
        obs[idx, 129 + s * 6 + cur_suits[:, s]] = 1.0
    cur_colors = cur_reg.sum(axis=2).astype(np.int64)  # (N, 5)
    for c in range(5):
        obs[idx, 153 + c * 5 + cur_colors[:, c]] = 1.0

    # oppo counts (offset 178..229: 3 + 24 + 25 = 52)
    oppo_reg = all_pub[:, :20].reshape(n, 5, 4)
    oppo_jok = all_pub[:, 20:].sum(axis=1).astype(np.int64)
    obs[idx, 178 + oppo_jok] = 1.0
    oppo_suits = oppo_reg.sum(axis=1).astype(np.int64)
    for s in range(4):
        obs[idx, 181 + s * 6 + oppo_suits[:, s]] = 1.0
    oppo_colors = oppo_reg.sum(axis=2).astype(np.int64)
    for c in range(5):
        obs[idx, 205 + c * 5 + oppo_colors[:, c]] = 1.0

    if obs_dim == 264:
        seen = np.clip(board_pub + cur_pub + cur_priv + oppo_pub + cur_hand, 0.0, 1.0)
        obs[:, 230:252] = 1.0 - seen
        if cur_turn is not None and 0 <= int(cur_turn) < 12:
            obs[:, 252 + int(cur_turn)] = 1.0

    return obs


@torch.no_grad()
def _batch_best_actions_and_values(dmc_agent, board_pub, cur_pub, cur_priv, oppo_pub, cur_hand, oppo_priv_count, cur_turn=None):
    """For N states, return best action_id (N,) and max Q value (N,)."""
    n = board_pub.shape[0]
    is_equivariant = hasattr(dmc_agent, 'forward_all')
    obs_dim = 318 if is_equivariant else int(dmc_agent.net.fc_layers[0].in_features) - 44
    obs = _extract_obs_batch(
        board_pub, cur_pub, cur_priv, oppo_pub, cur_hand, oppo_priv_count,
        obs_dim=obs_dim, cur_turn=cur_turn,
    )

    can_priv = (cur_priv.sum(axis=1) < 3)
    can_pub = (cur_pub.sum(axis=1) < 3)

    if is_equivariant:
        obs_t = torch.from_numpy(obs).to(dmc_agent.device)
        wdl_logits, diff_pred, _, _ = dmc_agent.forward_all(obs_t)
        all_utils = dmc_agent.net.compute_utility(wdl_logits, diff_pred).cpu().numpy()
        best_actions = np.zeros(n, dtype=np.int64)
        best_values = np.zeros(n, dtype=np.float32)
        for i in range(n):
            cards = np.flatnonzero(cur_hand[i])
            legal = []
            if can_pub[i]:
                legal.extend((cards + 22).tolist())
            if can_priv[i]:
                legal.extend(cards.tolist())
            sub_u = all_utils[i, legal]
            best_idx = int(np.argmax(sub_u))
            best_actions[i] = legal[best_idx]
            best_values[i] = sub_u[best_idx]
        return best_actions, best_values

    obs_list = []
    act_list = []
    act_ids = []
    state_offsets = np.zeros(n + 1, dtype=np.int64)

    for i in range(n):
        cards = np.flatnonzero(cur_hand[i])
        legal = []
        if can_pub[i]:
            legal.extend((cards + 22).tolist())
        if can_priv[i]:
            legal.extend(cards.tolist())
        k = len(legal)
        state_offsets[i + 1] = state_offsets[i] + k
        obs_list.append(np.repeat(obs[i:i + 1], k, axis=0))
        act_list.append(ACTION_EYE[legal])
        act_ids.extend(legal)

    all_obs = torch.from_numpy(np.concatenate(obs_list, axis=0)).to(dmc_agent.device)
    all_acts = torch.from_numpy(np.concatenate(act_list, axis=0)).to(dmc_agent.device)
    q_vals = dmc_agent.net.forward(all_obs, all_acts).cpu().numpy()

    best_actions = np.zeros(n, dtype=np.int64)
    best_values = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s, e = state_offsets[i], state_offsets[i + 1]
        sub_q = q_vals[s:e]
        best_idx = int(np.argmax(sub_q))
        best_actions[i] = act_ids[s + best_idx]
        best_values[i] = sub_q[best_idx]

    return best_actions, best_values


class DMCSearchAgent:
    """Belief-sampling + lookahead search wrapper around trained two-player DMCAgents."""

    def __init__(
        self,
        agents,
        player_id,
        num_samples=64,
        start_turn=10,
        rollout_from_turn=10,
        win_weight=10.0,
        exact_turn11_opponent=True,
        seed=None,
    ):
        self.use_raw = False
        self.agents = agents
        self.player_id = player_id
        self.num_samples = num_samples
        self.start_turn = start_turn
        self.rollout_from_turn = rollout_from_turn
        self.win_weight = win_weight
        self.exact_turn11_opponent = exact_turn11_opponent
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _cards_to_vec(cards):
        vec = np.zeros(22, dtype=np.float32)
        for c in cards:
            vec[CARD_TO_IDX[c]] = 1.0
        return vec

    def _turn11_exact(self, board_pub, cur_pub, cur_priv, oppo_pub, cur_hand, oppo_priv_count, legal_ids):
        """Exact Bayesian evaluation at cur_turn == 11 (only oppo_private is unknown)."""
        seen = board_pub + cur_pub + cur_priv + oppo_pub + cur_hand
        unknown = np.flatnonzero(seen == 0)

        if oppo_priv_count == 0:
            oppo_priv_mats = np.zeros((1, 22), dtype=np.float32)
        else:
            combos = list(itertools.combinations(unknown.tolist(), oppo_priv_count))
            oppo_priv_mats = np.zeros((len(combos), 22), dtype=np.float32)
            for idx, comb in enumerate(combos):
                oppo_priv_mats[idx, list(comb)] = 1.0

        m = oppo_priv_mats.shape[0]
        base_pub = np.repeat((board_pub + cur_pub + oppo_pub)[None, :], m, axis=0)
        base_cur_mine = np.repeat((board_pub + cur_pub + oppo_pub + cur_priv)[None, :], m, axis=0)
        base_oppo_mine = base_pub + oppo_priv_mats

        action_scores = {}
        action_diffs = {}
        for aid in legal_ids:
            card_idx = aid % 22
            is_pub = (aid >= 22)
            cur_mine = base_cur_mine.copy()
            cur_mine[:, card_idx] = 1.0
            if is_pub:
                oppo_mine = base_oppo_mine.copy()
                oppo_mine[:, card_idx] = 1.0
            else:
                oppo_mine = base_oppo_mine

            s_cur = _compute_scores_batch(cur_mine)
            s_oppo = _compute_scores_batch(oppo_mine)
            diff = s_cur - s_oppo
            win_util = (diff > 0).astype(np.float32) + 0.5 * (diff == 0).astype(np.float32)
            action_diffs[aid] = float(diff.mean())
            action_scores[aid] = float((self.win_weight * win_util + diff).mean())

        return action_scores, action_diffs

    def _sample_worlds(self, unknown_idx, oppo_priv_count, k):
        """Sample K determinized worlds: oppo_priv (K,22), oppo_hand (K,22), deck_order (K, rem)."""
        n_unk = len(unknown_idx)
        oppo_priv = np.zeros((k, 22), dtype=np.float32)
        oppo_hand = np.zeros((k, 22), dtype=np.float32)
        deck_rem = np.zeros((k, n_unk - oppo_priv_count - 3), dtype=np.int64)

        for i in range(k):
            perm = self.rng.permutation(unknown_idx)
            if oppo_priv_count > 0:
                oppo_priv[i, perm[:oppo_priv_count]] = 1.0
            oppo_hand[i, perm[oppo_priv_count:oppo_priv_count + 3]] = 1.0
            deck_rem[i] = perm[oppo_priv_count + 3:]

        return oppo_priv, oppo_hand, deck_rem

    def _turn11_oppo_greedy(self, board_pub, oppo_pub, oppo_priv, cur_pub, oppo_hand):
        """Vectorized exact non-peeking greedy move for turn 11 (always 4 cards in hand)."""
        n = board_pub.shape[0]
        all_pub = board_pub + cur_pub + oppo_pub
        can_priv = oppo_priv.sum(axis=1) < 3

        # At turn 11, oppo_hand always has 4 cards -> extract (N, 4) card indices
        _, hand_cols = np.where(oppo_hand > 0)
        hand_cards = hand_cols.reshape(n, 4)

        # Build (N, 4, 22) candidate tables
        oppo_cands = np.repeat((all_pub + oppo_priv)[:, None, :], 4, axis=1)
        pub_cands = np.repeat(all_pub[:, None, :], 4, axis=1)
        rows = np.repeat(np.arange(n), 4)
        cols = np.tile(np.arange(4), n)
        flat_cards = hand_cards.reshape(-1)
        oppo_cands[rows, cols, flat_cards] = 1.0
        pub_cands[rows, cols, flat_cards] = 1.0

        oppo_scores = _compute_scores_batch(oppo_cands.reshape(-1, 22)).reshape(n, 4)
        pub_scores = _compute_scores_batch(pub_cands.reshape(-1, 22)).reshape(n, 4)

        # Private slot doesn't change other player's table; public slot subtracts visible public gain
        utilities = np.where(can_priv[:, None], oppo_scores, oppo_scores - pub_scores)
        best_choice = np.argmax(utilities, axis=1)
        best_cards = hand_cards[np.arange(n), best_choice]
        return np.where(can_priv, best_cards, best_cards + 22)

    def _search(self, state):
        raw = state['raw_obs']
        cur_turn = raw['cur_turn']
        legal_ids = list(state['legal_actions'].keys())

        board_pub = self._cards_to_vec(raw['board_public'])
        cur_pub = self._cards_to_vec(raw['cur_public'])
        cur_priv = self._cards_to_vec(raw['cur_private'])
        oppo_pub = self._cards_to_vec(raw['oppo_public'])
        cur_hand = self._cards_to_vec(raw['cur_hand'])
        oppo_priv_count = int(raw['oppo_private_count'])

        if len(legal_ids) == 1:
            aid = legal_ids[0]
            return {aid: 0.0}, {aid: 0.0}

        if cur_turn == 11:
            return self._turn11_exact(
                board_pub, cur_pub, cur_priv, oppo_pub, cur_hand, oppo_priv_count, legal_ids
            )

        seen = board_pub + cur_pub + cur_priv + oppo_pub + cur_hand
        unknown_idx = np.flatnonzero(seen == 0)
        k = self.num_samples
        num_a = len(legal_ids)
        total = k * num_a

        oppo_priv_w, oppo_hand_w, deck_rem_w = self._sample_worlds(unknown_idx, oppo_priv_count, k)

        # Expand across all (K * num_a) branches
        b_board = np.repeat(board_pub[None, :], total, axis=0)
        b_cur_pub = np.repeat(cur_pub[None, :], total, axis=0)
        b_cur_priv = np.repeat(cur_priv[None, :], total, axis=0)
        b_cur_hand = np.repeat(cur_hand[None, :], total, axis=0)
        b_oppo_pub = np.repeat(oppo_pub[None, :], total, axis=0)
        b_oppo_priv = np.tile(oppo_priv_w, (num_a, 1))
        b_oppo_hand = np.tile(oppo_hand_w, (num_a, 1))
        b_deck = np.tile(deck_rem_w, (num_a, 1))

        # Apply our candidate actions
        for a_idx, aid in enumerate(legal_ids):
            sl = slice(a_idx * k, (a_idx + 1) * k)
            c = aid % 22
            b_cur_hand[sl, c] = 0.0
            if aid >= 22:
                b_cur_pub[sl, c] = 1.0
            else:
                b_cur_priv[sl, c] = 1.0

        deck_ptr = 0
        # Opponent draws 1 card
        b_oppo_hand[np.arange(total), b_deck[:, deck_ptr]] = 1.0
        deck_ptr += 1

        # Simulate opponent move and subsequent turns
        oppo_id = 1 - self.player_id
        my_agent = self.agents[self.player_id]
        oppo_agent = self.agents[oppo_id]

        use_full_rollout = (cur_turn >= self.rollout_from_turn)
        t = cur_turn + 1

        # Step 1: opponent acts at turn t = cur_turn + 1
        if t == 11 and self.exact_turn11_opponent:
            oppo_acts = self._turn11_oppo_greedy(b_board, b_oppo_pub, b_oppo_priv, b_cur_pub, b_oppo_hand)
        else:
            oppo_acts, _ = _batch_best_actions_and_values(
                oppo_agent, b_board, b_oppo_pub, b_oppo_priv, b_cur_pub, b_oppo_hand,
                b_cur_priv.sum(axis=1).astype(np.int64), cur_turn=t,
            )
        idx_arr = np.arange(total)
        oppo_cards = oppo_acts % 22
        b_oppo_hand[idx_arr, oppo_cards] = 0.0
        is_oppo_pub = oppo_acts >= 22
        b_oppo_pub[idx_arr[is_oppo_pub], oppo_cards[is_oppo_pub]] = 1.0
        b_oppo_priv[idx_arr[~is_oppo_pub], oppo_cards[~is_oppo_pub]] = 1.0

        t += 1
        if t < 12:
            # We draw 1 card
            b_cur_hand[idx_arr, b_deck[:, deck_ptr]] = 1.0
            deck_ptr += 1

        if use_full_rollout:
            while t < 12:
                if (t % 2) == self.player_id:
                    if t == 11:
                        acts = self._turn11_oppo_greedy(b_board, b_cur_pub, b_cur_priv, b_oppo_pub, b_cur_hand)
                    else:
                        acts, _ = _batch_best_actions_and_values(
                            my_agent, b_board, b_cur_pub, b_cur_priv, b_oppo_pub, b_cur_hand,
                            b_oppo_priv.sum(axis=1).astype(np.int64), cur_turn=t,
                        )
                    cards = acts % 22
                    b_cur_hand[idx_arr, cards] = 0.0
                    is_pub = acts >= 22
                    b_cur_pub[idx_arr[is_pub], cards[is_pub]] = 1.0
                    b_cur_priv[idx_arr[~is_pub], cards[~is_pub]] = 1.0
                    t += 1
                    if t < 12:
                        b_oppo_hand[idx_arr, b_deck[:, deck_ptr]] = 1.0
                        deck_ptr += 1
                else:
                    if t == 11 and self.exact_turn11_opponent:
                        acts = self._turn11_oppo_greedy(b_board, b_oppo_pub, b_oppo_priv, b_cur_pub, b_oppo_hand)
                    else:
                        acts, _ = _batch_best_actions_and_values(
                            oppo_agent, b_board, b_oppo_pub, b_oppo_priv, b_cur_pub, b_oppo_hand,
                            b_cur_priv.sum(axis=1).astype(np.int64), cur_turn=t,
                        )
                    cards = acts % 22
                    b_oppo_hand[idx_arr, cards] = 0.0
                    is_pub = acts >= 22
                    b_oppo_pub[idx_arr[is_pub], cards[is_pub]] = 1.0
                    b_oppo_priv[idx_arr[~is_pub], cards[~is_pub]] = 1.0
                    t += 1
                    if t < 12:
                        b_cur_hand[idx_arr, b_deck[:, deck_ptr]] = 1.0
                        deck_ptr += 1

        if t >= 12:
            all_pub = b_board + b_cur_pub + b_oppo_pub
            s_cur = _compute_scores_batch(all_pub + b_cur_priv)
            s_oppo = _compute_scores_batch(all_pub + b_oppo_priv)
            diffs = s_cur - s_oppo
            win_utils = (diffs > 0).astype(np.float32) + 0.5 * (diffs == 0).astype(np.float32)
            utils = self.win_weight * win_utils + diffs
        else:
            # Depth-2 Expectimax leaf evaluated by DMC value network
            _, leaf_vals = _batch_best_actions_and_values(
                my_agent, b_board, b_cur_pub, b_cur_priv, b_oppo_pub, b_cur_hand,
                b_oppo_priv.sum(axis=1).astype(np.int64), cur_turn=t,
            )
            diffs = leaf_vals
            utils = leaf_vals

        action_scores = {}
        action_diffs = {}
        for a_idx, aid in enumerate(legal_ids):
            sl = slice(a_idx * k, (a_idx + 1) * k)
            action_scores[aid] = float(utils[sl].mean())
            action_diffs[aid] = float(diffs[sl].mean())

        return action_scores, action_diffs

    def step(self, state):
        if state['raw_obs']['cur_turn'] < self.start_turn:
            return self.agents[self.player_id].step(state)
        action_scores, _ = self._search(state)
        return max(action_scores, key=action_scores.get)

    def eval_step(self, state):
        if state['raw_obs']['cur_turn'] < self.start_turn:
            return self.agents[self.player_id].eval_step(state)
        action_scores, action_diffs = self._search(state)
        best_action = max(action_scores, key=action_scores.get)
        info = {
            'values': {ID_2_ACTION[aid]: action_diffs[aid] for aid in state['legal_actions'].keys()},
            'search_scores': {ID_2_ACTION[aid]: action_scores[aid] for aid in state['legal_actions'].keys()},
        }
        return best_action, info
