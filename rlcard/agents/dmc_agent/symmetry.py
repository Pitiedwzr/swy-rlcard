import itertools
import torch


def build_swy_blm_symmetry_tables(obs_dim=264, include_joker_swap=True):
    """Build all equivalent suit/color(/joker) gather permutation tables for swy-blm.

    In swy-blm:
      - 4 suits (M, L, Z, J) are fully symmetric (4! = 24)
      - 3 low colors (G, B, Y) are fully symmetric (3! = 6)
      - 2 high colors (C, R) are fully symmetric (2! = 2)
      - 2 jokers (XW, DW) are symmetric in scoring (2! = 2)
    Total: 288 symmetries (or 576 with joker swap).

    Returns:
        obs_perms: LongTensor of shape (num_syms, obs_dim)
        act_perms: LongTensor of shape (num_syms, 44)
    """
    if obs_dim not in (230, 264, 318):
        raise ValueError(f"Unsupported swy-blm obs_dim: {obs_dim}")

    suit_perms = list(itertools.permutations(range(4)))       # 24
    low_color_perms = list(itertools.permutations(range(3)))  # 6
    high_color_perms = list(itertools.permutations((3, 4)))   # 2
    joker_perms = [(20, 21), (21, 20)] if include_joker_swap else [(20, 21)]

    obs_rows = []
    act_rows = []

    for sp in suit_perms:
        for lcp in low_color_perms:
            for hcp in high_color_perms:
                cp = lcp + hcp  # length 5
                for jp in joker_perms:
                    card_perm = [cp[c] * 4 + sp[s] for c in range(5) for s in range(4)] + list(jp)

                    # Action space: [private 0..21, public 22..43]
                    act_perm = card_perm + [22 + idx for idx in card_perm]
                    act_rows.append(act_perm)

                    # Observation vector layout (see SwyBlmEnv._extract_state)
                    obs_perm = []
                    if obs_dim == 318:
                        # 12 card regions of 22 dims each:
                        # 6 state masks (0..131) + 6 oppo_pub_by_step masks (132..263)
                        for base in range(0, 264, 22):
                            obs_perm.extend(base + idx for idx in card_perm)
                        # 54 permutation-invariant timing/space/turn/player features (264..317)
                        obs_perm.extend(range(264, 318))
                        obs_rows.append(obs_perm)
                        continue

                    # 5 card regions: board_public, cur_public, cur_private, oppo_public, cur_hand (0..109)
                    for base in (0, 22, 44, 66, 88):
                        obs_perm.extend(base + idx for idx in card_perm)
                    # 4 empty-space one-hots (110..125) + cur_joker_count (126..128)
                    obs_perm.extend(range(110, 129))
                    # cur_suit_count: 4 suits * 6 (129..152)
                    for s in range(4):
                        src_s = sp[s]
                        obs_perm.extend(range(129 + src_s * 6, 129 + (src_s + 1) * 6))
                    # cur_color_count: 5 colors * 5 (153..177)
                    for c in range(5):
                        src_c = cp[c]
                        obs_perm.extend(range(153 + src_c * 5, 153 + (src_c + 1) * 5))
                    # oppo_joker_count (178..180)
                    obs_perm.extend(range(178, 181))
                    # oppo_suit_count: 4 suits * 6 (181..204)
                    for s in range(4):
                        src_s = sp[s]
                        obs_perm.extend(range(181 + src_s * 6, 181 + (src_s + 1) * 6))
                    # oppo_color_count: 5 colors * 5 (205..229)
                    for c in range(5):
                        src_c = cp[c]
                        obs_perm.extend(range(205 + src_c * 5, 205 + (src_c + 1) * 5))

                    if obs_dim == 264:
                        # unseen_cards (230..251)
                        obs_perm.extend(230 + idx for idx in card_perm)
                        # cur_turn one-hot (252..263)
                        obs_perm.extend(range(252, 264))

                    obs_rows.append(obs_perm)

    return torch.tensor(obs_rows, dtype=torch.long), torch.tensor(act_rows, dtype=torch.long)
