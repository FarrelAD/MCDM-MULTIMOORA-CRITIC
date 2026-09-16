import numpy as np
from scipy.stats import rankdata

from multimoora_critic.models import MultimooraResult
from multimoora_critic.utils import validate_input, critic_weights, vector_normalization


def multimoora_critic(
    matrix: np.ndarray,
    criteria: np.ndarray,
    eps: float = 1e-9,
) -> MultimooraResult:
    """Execute full MULTIMOORA decision analysis with CRITIC criterion weighting.

    Parameters
    ----------
    matrix : np.ndarray
        2D decision matrix of shape (n_alternatives, n_criteria).
    criteria : np.ndarray
        1D array of criteria types (1 for benefit, -1 for cost).
    eps : float, optional
        Small constant added to matrix entries to avoid log(0) in FMF calculation.

    Returns
    -------
    MultimooraResult
        Structured container with weights, intermediate scores, component ranks,
        and final aggregated ranks.
    """
    matrix_arr, criteria_arr = validate_input(matrix, criteria)
    n_alt, n_crit = matrix_arr.shape

    # 1. Weight Calculation (CRITIC)
    weights = critic_weights(matrix_arr)

    # 2. Vector Normalization
    norm_matrix = vector_normalization(matrix_arr)

    benefit_mask = criteria_arr == 1
    cost_mask = criteria_arr == -1

    # --- Phase 1: Ratio System (RS) ---
    # RS Score = Σ(w_j * r_ij)_benefit - Σ(w_j * r_ij)_cost
    # Higher score is better
    benefit_rs = (
        np.sum(norm_matrix[:, benefit_mask] * weights[benefit_mask], axis=1)
        if np.any(benefit_mask)
        else np.zeros(n_alt)
    )
    cost_rs = (
        np.sum(norm_matrix[:, cost_mask] * weights[cost_mask], axis=1)
        if np.any(cost_mask)
        else np.zeros(n_alt)
    )
    rs_scores = benefit_rs - cost_rs
    # Rank 1 is highest score (rankdata on negative scores)
    rs_ranks = rankdata(-rs_scores, method="min").astype(int)

    # --- Phase 2: Reference Point Approach (RPA) ---
    # Reference point r*
    ref_point = np.zeros(n_crit)
    if np.any(benefit_mask):
        ref_point[benefit_mask] = np.max(norm_matrix[:, benefit_mask], axis=0)
    if np.any(cost_mask):
        ref_point[cost_mask] = np.min(norm_matrix[:, cost_mask], axis=0)

    # RPA Score = max_j(w_j * |r*_j - r_ij|)
    # Lower score is better
    rpa_scores = np.max(weights * np.abs(ref_point - norm_matrix), axis=1)
    # Rank 1 is lowest score
    rpa_ranks = rankdata(rpa_scores, method="min").astype(int)

    # --- Phase 3: Full Multiplicative Form (FMF) ---
    # FMF Score using log scale: Σ(w_j * log(x_ij))_benefit - Σ(w_j * log(x_ij))_cost
    # Higher score is better
    log_matrix = np.log(matrix_arr + eps)
    benefit_fmf = (
        np.sum(weights[benefit_mask] * log_matrix[:, benefit_mask], axis=1)
        if np.any(benefit_mask)
        else np.zeros(n_alt)
    )
    cost_fmf = (
        np.sum(weights[cost_mask] * log_matrix[:, cost_mask], axis=1)
        if np.any(cost_mask)
        else np.zeros(n_alt)
    )
    fmf_scores = benefit_fmf - cost_fmf
    # Rank 1 is highest score
    fmf_ranks = rankdata(-fmf_scores, method="min").astype(int)

    # --- Phase 4: Final Aggregation (Dominance Theory via Multiplicative Rank Product) ---
    final_scores = rs_ranks * rpa_ranks * fmf_ranks
    # Rank 1 is lowest rank product score
    final_ranks = rankdata(final_scores, method="min").astype(int)

    return MultimooraResult(
        weights=weights,
        rs_scores=rs_scores,
        rs_ranks=rs_ranks,
        rpa_scores=rpa_scores,
        rpa_ranks=rpa_ranks,
        fmf_scores=fmf_scores,
        fmf_ranks=fmf_ranks,
        final_scores=final_scores,
        final_ranks=final_ranks,
    )
