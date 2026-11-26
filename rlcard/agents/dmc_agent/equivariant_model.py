import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .model import _normalize_device
from rlcard.envs.bailongmen import OBS_V4_DIM


def _table_stats_and_deltas(mask22):
    """Compute exact bailongmen scores, counts, and per-card marginal score gains."""
    bsz = mask22.shape[0]
    reg = mask22[:, :20].view(bsz, 5, 4)
    jok = mask22[:, 20:]

    suit_cnt = reg.sum(dim=1)      # (B, 4)
    color_cnt = reg.sum(dim=2)     # (B, 5)
    jok_cnt = jok.sum(dim=1)       # (B,)

    score = (
        (suit_cnt == 5).float() * 8.0 + (suit_cnt == 4).float() * 4.0
    ).sum(dim=1)
    score = score + (
        (color_cnt[:, :3] == 4).float() * 3.0 + (color_cnt[:, :3] == 3).float() * 1.0
    ).sum(dim=1)
    score = score + (
        (color_cnt[:, 3:5] == 4).float() * 6.0 + (color_cnt[:, 3:5] == 3).float() * 3.0
    ).sum(dim=1)
    score = score + (jok_cnt == 2).float() * 5.0

    suit_gain = ((suit_cnt == 3) | (suit_cnt == 4)).float() * 4.0
    low_gain = (color_cnt[:, :3] == 2).float() * 1.0 + (color_cnt[:, :3] == 3).float() * 2.0
    high_gain = ((color_cnt[:, 3:5] == 2) | (color_cnt[:, 3:5] == 3)).float() * 3.0
    color_gain = torch.cat((low_gain, high_gain), dim=1)

    reg_delta = (suit_gain.unsqueeze(1) + color_gain.unsqueeze(2)) * (1.0 - reg)
    jok_delta = ((jok_cnt.unsqueeze(1) == 1).float() * 5.0) * (1.0 - jok)
    delta = torch.cat((reg_delta.reshape(bsz, 20), jok_delta), dim=1)

    return score, suit_cnt, color_cnt, jok_cnt, delta


class EquivariantBlock(nn.Module):
    """Suit-, low-color-, high-color-, and joker-equivariant residual interaction block."""

    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.norm_reg = nn.LayerNorm(d_model)
        self.norm_jok = nn.LayerNorm(d_model)
        self.norm_glob = nn.LayerNorm(d_model)

        # Global update: [H_glob, low_mean, low_max, high_mean, high_max, jok_mean, jok_max] = 7 * d
        self.glob_mlp = nn.Sequential(
            nn.Linear(7 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        # Regular card update: [H_reg, row_mean, row_max, low_col_mean, low_col_max, high_col_mean, high_col_max, H_glob] = 8 * d
        self.reg_mlp = nn.Sequential(
            nn.Linear(8 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        # Joker update: [H_jok, jok_mean, jok_max, low_mean, low_max, high_mean, high_max, H_glob] = 8 * d
        self.jok_mlp = nn.Sequential(
            nn.Linear(8 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, h_reg, h_jok, h_glob):
        bsz = h_reg.shape[0]
        r_in = self.norm_reg(h_reg)
        j_in = self.norm_jok(h_jok)
        g_in = self.norm_glob(h_glob)

        # Suit-axis summaries per color row: (B, 5, 1, D)
        row_mean = r_in.mean(dim=2, keepdim=True)
        row_max = r_in.amax(dim=2, keepdim=True)

        # Color-axis summaries per suit column split into low (0..2) and high (3..4) orbits: (B, 1, 4, D)
        low_col_mean = r_in[:, :3].mean(dim=1, keepdim=True)
        low_col_max = r_in[:, :3].amax(dim=1, keepdim=True)
        high_col_mean = r_in[:, 3:].mean(dim=1, keepdim=True)
        high_col_max = r_in[:, 3:].amax(dim=1, keepdim=True)

        # Region-level permutation-invariant summaries: (B, D)
        low_flat = r_in[:, :3].reshape(bsz, 12, self.d_model)
        high_flat = r_in[:, 3:].reshape(bsz, 8, self.d_model)
        low_mean = low_flat.mean(dim=1)
        low_max = low_flat.amax(dim=1)
        high_mean = high_flat.mean(dim=1)
        high_max = high_flat.amax(dim=1)
        jok_mean = j_in.mean(dim=1)
        jok_max = j_in.amax(dim=1)

        g_cat = torch.cat((g_in, low_mean, low_max, high_mean, high_max, jok_mean, jok_max), dim=-1)
        h_glob_next = h_glob + self.glob_mlp(g_cat)

        g_reg = h_glob_next.view(bsz, 1, 1, self.d_model).expand(bsz, 5, 4, self.d_model)
        r_cat = torch.cat(
            (
                r_in,
                row_mean.expand(-1, -1, 4, -1),
                row_max.expand(-1, -1, 4, -1),
                low_col_mean.expand(-1, 5, -1, -1),
                low_col_max.expand(-1, 5, -1, -1),
                high_col_mean.expand(-1, 5, -1, -1),
                high_col_max.expand(-1, 5, -1, -1),
                g_reg,
            ),
            dim=-1,
        )
        h_reg_next = h_reg + self.reg_mlp(r_cat)

        g_jok = h_glob_next.view(bsz, 1, self.d_model).expand(bsz, 2, self.d_model)
        j_cat = torch.cat(
            (
                j_in,
                jok_mean.unsqueeze(1).expand(-1, 2, -1),
                jok_max.unsqueeze(1).expand(-1, 2, -1),
                low_mean.unsqueeze(1).expand(-1, 2, -1),
                low_max.unsqueeze(1).expand(-1, 2, -1),
                high_mean.unsqueeze(1).expand(-1, 2, -1),
                high_max.unsqueeze(1).expand(-1, 2, -1),
                g_jok,
            ),
            dim=-1,
        )
        h_jok_next = h_jok + self.jok_mlp(j_cat)

        return h_reg_next, h_jok_next, h_glob_next


class SwyBlmEquivariantNet(nn.Module):
    """576-fold suit/color/joker equivariant multi-head network for swy-blm v4 obs (318 dims).

    Outputs:
      - wdl_logits: (B, 44, 3) logits for [loss, draw, win] per action
      - diff_pred: (B, 44) predicted score differential per action
      - belief_logits: (B, 22) logits predicting P(card in oppo_private)
      - belief_logits_mid: (B, 22) intermediate belief logits used for conditioning
    """

    def __init__(self, d_model=128, num_blocks=3, margin_weight=0.02):
        super().__init__()
        if num_blocks < 2:
            raise ValueError("num_blocks must be >= 2")
        self.d_model = int(d_model)
        self.num_blocks = int(num_blocks)
        self.margin_weight = float(margin_weight)

        self.reg_proj = nn.Sequential(
            nn.Linear(28, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.jok_proj = nn.Sequential(
            nn.Linear(20, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.glob_proj = nn.Sequential(
            nn.Linear(59, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.stage1_blocks = nn.ModuleList([EquivariantBlock(d_model) for _ in range(num_blocks - 1)])
        self.belief_head_mid = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.belief_reg_inject = nn.Linear(2, d_model)
        self.belief_jok_inject = nn.Linear(2, d_model)

        self.stage2_block = EquivariantBlock(d_model)
        self.belief_head_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        head_in = 2 * d_model + 1
        self.priv_trunk = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.pub_trunk = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.wdl_priv = nn.Linear(d_model, 3)
        self.wdl_pub = nn.Linear(d_model, 3)
        self.diff_priv = nn.Linear(d_model, 1)
        self.diff_pub = nn.Linear(d_model, 1)

        high_mask = torch.zeros(1, 5, 4, 1, dtype=torch.float32)
        high_mask[:, 3:, :, :] = 1.0
        self.register_buffer("high_color_mask", high_mask, persistent=False)

        is_jok = torch.zeros(1, 22, 1, dtype=torch.float32)
        is_jok[:, 20:, :] = 1.0
        self.register_buffer("is_joker_flag", is_jok, persistent=False)

        idx03 = torch.arange(4, dtype=torch.float32)
        self.register_buffer("idx03", idx03, persistent=False)

    def _build_inputs(self, obs):
        bsz = obs.shape[0]
        cards_raw = obs[:, :264].view(bsz, 12, 22).permute(0, 2, 1)  # (B, 22, 12)
        board_pub = cards_raw[:, :, 0]
        cur_pub = cards_raw[:, :, 1]
        cur_priv = cards_raw[:, :, 2]
        oppo_pub = cards_raw[:, :, 3]
        cur_hand = cards_raw[:, :, 4]
        unseen = cards_raw[:, :, 5]

        all_pub = board_pub + cur_pub + oppo_pub
        cur_mine = all_pub + cur_priv

        oppo_priv_rem = (obs[:, 300:304] * self.idx03).sum(dim=1, keepdim=True)
        oppo_priv_cnt = 3.0 - oppo_priv_rem
        unseen_cnt = unseen.sum(dim=1, keepdim=True).clamp(min=1.0)
        prior_oppo_priv = unseen * (oppo_priv_cnt / unseen_cnt)

        cur_score, cur_suit_cnt, cur_color_cnt, cur_jok_cnt, delta_mine = _table_stats_and_deltas(cur_mine)
        oppo_pub_score, oppo_suit_cnt, oppo_color_cnt, oppo_jok_cnt, delta_oppo_pub = _table_stats_and_deltas(all_pub)

        hand_reg = cur_hand[:, :20].view(bsz, 5, 4)
        unseen_reg = unseen[:, :20].view(bsz, 5, 4)
        hand_suit_cnt = hand_reg.sum(dim=1)
        hand_color_cnt = hand_reg.sum(dim=2)
        hand_jok_cnt = cur_hand[:, 20:].sum(dim=1)
        unseen_suit_cnt = unseen_reg.sum(dim=1)
        unseen_color_cnt = unseen_reg.sum(dim=2)
        unseen_jok_cnt = unseen[:, 20:].sum(dim=1)

        avail_suit = cur_suit_cnt + hand_suit_cnt + unseen_suit_cnt
        avail_color = cur_color_cnt + hand_color_cnt + unseen_color_cnt
        can_4_suit = (avail_suit >= 4).float()
        can_5_suit = (avail_suit >= 5).float()
        can_3_color = (avail_color >= 3).float()
        can_4_color = (avail_color >= 4).float()
        can_2_jok = ((cur_jok_cnt + hand_jok_cnt + unseen_jok_cnt) >= 2).float()

        suit_feats = torch.stack(
            (
                cur_suit_cnt / 5.0,
                oppo_suit_cnt / 5.0,
                hand_suit_cnt / 4.0,
                unseen_suit_cnt / 5.0,
                can_4_suit,
                can_5_suit,
            ),
            dim=-1,
        ).unsqueeze(1).expand(bsz, 5, 4, 6)

        color_feats = torch.stack(
            (
                cur_color_cnt / 4.0,
                oppo_color_cnt / 4.0,
                hand_color_cnt / 4.0,
                unseen_color_cnt / 4.0,
                can_3_color,
                can_4_color,
            ),
            dim=-1,
        ).unsqueeze(2).expand(bsz, 5, 4, 6)

        x_reg = torch.cat(
            (
                cards_raw[:, :20].view(bsz, 5, 4, 12),
                prior_oppo_priv[:, :20].view(bsz, 5, 4, 1),
                (delta_mine[:, :20].view(bsz, 5, 4, 1) / 10.0),
                (delta_oppo_pub[:, :20].view(bsz, 5, 4, 1) / 10.0),
                self.high_color_mask.expand(bsz, -1, -1, -1),
                suit_feats,
                color_feats,
            ),
            dim=-1,
        )

        jok_counts = torch.stack(
            (
                cur_jok_cnt / 2.0,
                oppo_jok_cnt / 2.0,
                hand_jok_cnt / 2.0,
                unseen_jok_cnt / 2.0,
                can_2_jok,
            ),
            dim=-1,
        ).unsqueeze(1).expand(bsz, 2, 5)

        x_jok = torch.cat(
            (
                cards_raw[:, 20:],
                prior_oppo_priv[:, 20:].unsqueeze(-1),
                (delta_mine[:, 20:].unsqueeze(-1) / 10.0),
                (delta_oppo_pub[:, 20:].unsqueeze(-1) / 10.0),
                jok_counts,
            ),
            dim=-1,
        )

        x_glob = torch.cat(
            (
                obs[:, 264:],
                (cur_score.unsqueeze(1) / 20.0),
                (oppo_pub_score.unsqueeze(1) / 20.0),
                ((cur_score - oppo_pub_score).unsqueeze(1) / 10.0),
                (oppo_priv_cnt / 3.0),
                (unseen_cnt / 16.0),
            ),
            dim=-1,
        )
        return x_reg, x_jok, x_glob, unseen, oppo_priv_cnt

    def forward_all(self, obs):
        obs = torch.flatten(obs, 1).float()
        bsz = obs.shape[0]
        x_reg, x_jok, x_glob, unseen, oppo_priv_cnt = self._build_inputs(obs)

        h_reg = self.reg_proj(x_reg)
        h_jok = self.jok_proj(x_jok)
        h_glob = self.glob_proj(x_glob)

        for block in self.stage1_blocks:
            h_reg, h_jok, h_glob = block(h_reg, h_jok, h_glob)

        cards_mid = torch.cat((h_reg.reshape(bsz, 20, self.d_model), h_jok), dim=1)
        belief_logits_mid = self.belief_head_mid(cards_mid).squeeze(-1)

        p_raw = torch.sigmoid(belief_logits_mid) * unseen
        p_norm = (p_raw * oppo_priv_cnt / p_raw.sum(dim=1, keepdim=True).clamp(min=1e-4)).clamp(max=1.0)
        belief_feat = torch.stack((p_raw, p_norm), dim=-1)  # (B, 22, 2)

        h_reg = h_reg + self.belief_reg_inject(belief_feat[:, :20].view(bsz, 5, 4, 2))
        h_jok = h_jok + self.belief_jok_inject(belief_feat[:, 20:])

        h_reg, h_jok, h_glob = self.stage2_block(h_reg, h_jok, h_glob)

        cards_out = torch.cat((h_reg.reshape(bsz, 20, self.d_model), h_jok), dim=1)
        belief_logits = self.belief_head_out(cards_out).squeeze(-1)

        g_exp = h_glob.unsqueeze(1).expand(bsz, 22, self.d_model)
        act_in = torch.cat((cards_out, g_exp, self.is_joker_flag.expand(bsz, -1, -1)), dim=-1)

        priv_h = self.priv_trunk(act_in)
        pub_h = self.pub_trunk(act_in)

        wdl_logits = torch.cat((self.wdl_priv(priv_h), self.wdl_pub(pub_h)), dim=1)  # (B, 44, 3)
        diff_pred = torch.cat((self.diff_priv(priv_h), self.diff_pub(pub_h)), dim=1).squeeze(-1)  # (B, 44)

        return wdl_logits, diff_pred, belief_logits, belief_logits_mid

    def compute_utility(self, wdl_logits, diff_pred):
        probs = F.softmax(wdl_logits, dim=-1)
        win_score = probs[..., 2] + 0.5 * probs[..., 1]
        return win_score + self.margin_weight * diff_pred

    def forward(self, obs, actions):
        wdl_logits, diff_pred, _, _ = self.forward_all(obs)
        utility = self.compute_utility(wdl_logits, diff_pred)
        actions = torch.flatten(actions, 1).float()
        return (utility * actions).sum(dim=1)


class EquivariantDMCAgent:
    """DMCAgent-compatible wrapper around SwyBlmEquivariantNet."""

    def __init__(
        self,
        state_shape=(OBS_V4_DIM,),
        action_shape=(44,),
        d_model=128,
        num_blocks=3,
        margin_weight=0.02,
        exp_epsilon=0.01,
        device="0",
    ):
        self.use_raw = False
        self.device = _normalize_device(device)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.exp_epsilon = exp_epsilon
        self.net = SwyBlmEquivariantNet(
            d_model=d_model,
            num_blocks=num_blocks,
            margin_weight=margin_weight,
        ).to(self.device)

    def step(self, state):
        action_keys, values = self.predict(state)
        if self.exp_epsilon > 0 and np.random.rand() < self.exp_epsilon:
            return int(np.random.choice(action_keys))
        return int(action_keys[int(np.argmax(values))])

    def eval_step(self, state):
        action_keys, values, extra = self.predict_detailed(state)
        action_idx = int(np.argmax(values))
        action = int(action_keys[action_idx])
        info = {
            "values": {state["raw_legal_actions"][i]: float(values[i]) for i in range(len(action_keys))},
            "win_probs": {state["raw_legal_actions"][i]: float(extra["win_probs"][i]) for i in range(len(action_keys))},
            "diff_preds": {state["raw_legal_actions"][i]: float(extra["diff_preds"][i]) for i in range(len(action_keys))},
            "belief_probs": extra["belief_probs"],
        }
        return action, info

    @torch.no_grad()
    def predict_detailed(self, state):
        obs = torch.from_numpy(state["obs"].astype(np.float32)).unsqueeze(0).to(self.device)
        action_keys = np.array(list(state["legal_actions"].keys()), dtype=np.int64)
        wdl_logits, diff_pred, belief_logits, _ = self.net.forward_all(obs)
        utility = self.net.compute_utility(wdl_logits, diff_pred)[0]
        probs = F.softmax(wdl_logits[0], dim=-1)
        win_probs = probs[:, 2] + 0.5 * probs[:, 1]
        belief_probs = torch.sigmoid(belief_logits[0]).cpu().numpy()

        key_t = torch.from_numpy(action_keys).to(self.device)
        values = utility[key_t].cpu().numpy()
        extra = {
            "win_probs": win_probs[key_t].cpu().numpy(),
            "diff_preds": diff_pred[0, key_t].cpu().numpy(),
            "belief_probs": belief_probs,
        }
        return action_keys, values, extra

    def predict(self, state):
        action_keys, values, _ = self.predict_detailed(state)
        return action_keys, values

    def forward_all(self, obs):
        return self.net.forward_all(obs)

    def forward(self, obs, actions):
        return self.net.forward(obs, actions)

    def share_memory(self):
        self.net.share_memory()

    def eval(self):
        self.net.eval()

    def parameters(self):
        return self.net.parameters()

    def load_state_dict(self, state_dict):
        return self.net.load_state_dict(state_dict)

    def state_dict(self):
        return self.net.state_dict()

    def set_device(self, device):
        self.device = _normalize_device(device)
        self.net.to(self.device)


class EquivariantDMCModel:
    """Two-player container matching DMCModel interface for EquivariantDMCAgent."""

    def __init__(
        self,
        state_shape,
        action_shape,
        d_model=128,
        num_blocks=3,
        margin_weight=0.02,
        exp_epsilon=0.01,
        device=0,
    ):
        self.agents = []
        for player_id in range(len(state_shape)):
            agent = EquivariantDMCAgent(
                state_shape[player_id],
                action_shape[player_id],
                d_model=d_model,
                num_blocks=num_blocks,
                margin_weight=margin_weight,
                exp_epsilon=exp_epsilon,
                device=device,
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
