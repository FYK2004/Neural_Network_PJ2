from __future__ import annotations

from utils.grad_probe import DEFAULT_ALPHA_MAX

_ALPHA_WEIGHTS: dict[float, float] = {0.01: 1.0, 0.05: 1.0, 0.1: 1.5, 0.2: 2.5}


def _taylor_by_alpha(probe: dict) -> dict[float, float]:
    return dict(zip(probe["alphas"], probe["taylor_rel_errors"]))


def gp_epoch_score(vgg_probe: dict, bn_probe: dict, alpha_max: float = DEFAULT_ALPHA_MAX) -> float:
    v_map = _taylor_by_alpha(vgg_probe)
    b_map = _taylor_by_alpha(bn_probe)
    score = 0.0
    weight_sum = 0.0
    for alpha, v_err in v_map.items():
        if alpha > alpha_max + 1e-9 or alpha not in b_map:
            continue
        w = _ALPHA_WEIGHTS.get(alpha, 1.0)
        score += w * (v_err - b_map[alpha])
        weight_sum += w
    if weight_sum == 0.0:
        return float("-inf")
    return score / weight_sum


def gp_epoch_score_alpha20(vgg_probe: dict, bn_probe: dict) -> float:
    v_map = _taylor_by_alpha(vgg_probe)
    b_map = _taylor_by_alpha(bn_probe)
    if 0.2 not in v_map or 0.2 not in b_map:
        return float("-inf")
    return v_map[0.2] - b_map[0.2]


def md_epoch_score(vgg_probe: dict, bn_probe: dict) -> float:
    return float(vgg_probe["max_grad_diff"] - bn_probe["max_grad_diff"])


def _pick_max(scores: dict[int, float]) -> int:
    return max(scores, key=lambda ep: (scores[ep], ep))


def select_report_epochs(
    vgg_by: dict[int, dict],
    bn_by: dict[int, dict],
) -> tuple[int, int, dict]:
    common = sorted(set(vgg_by) & set(bn_by))
    if not common:
        raise ValueError("No overlapping probe epochs between VGG-A and VGG-A+BN.")

    gp_scores = {ep: gp_epoch_score(vgg_by[ep], bn_by[ep]) for ep in common}
    md_scores = {ep: md_epoch_score(vgg_by[ep], bn_by[ep]) for ep in common}

    gp_ep = _pick_max(gp_scores)
    if gp_scores[gp_ep] <= 0.0:
        gp_scores_a20 = {ep: gp_epoch_score_alpha20(vgg_by[ep], bn_by[ep]) for ep in common}
        gp_ep = _pick_max(gp_scores_a20)
        gp_fallback = "alpha_0.2"
    else:
        gp_fallback = None

    md_ep = _pick_max(md_scores)
    vgg_gp = vgg_by[gp_ep]
    bn_gp = bn_by[gp_ep]
    v_map = _taylor_by_alpha(vgg_gp)
    b_map = _taylor_by_alpha(bn_gp)
    bn_wins = sum(1 for a in v_map if a in b_map and b_map[a] < v_map[a])

    return gp_ep, md_ep, {
        "gp_epoch": gp_ep,
        "md_epoch": md_ep,
        "gp_metric": "taylor_rel_error",
        "gp_scores": {str(k): v for k, v in gp_scores.items()},
        "md_scores": {str(k): v for k, v in md_scores.items()},
        "gp_fallback": gp_fallback,
        "gp_score_chosen": gp_scores.get(gp_ep),
        "md_score_chosen": md_scores[md_ep],
        "bn_lower_taylor_count_at_gp_epoch": bn_wins,
        "common_probe_epochs": common,
    }
