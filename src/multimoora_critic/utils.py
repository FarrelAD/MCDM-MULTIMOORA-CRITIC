from typing import Tuple
import numpy as np


def validate_input(matrix: np.ndarray, criteria: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and sanitize input decision matrix and criteria array.

    Parameters
    ----------
    matrix : np.ndarray
        2D decision matrix of shape (n_alternatives, n_criteria).
    criteria : np.ndarray
        1D array of criteria types (1 for benefit, -1 for cost) of length n_criteria.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Sanitized (matrix, criteria) as float 2D array and int 1D array.
    """
    matrix_arr = np.asarray(matrix, dtype=float)
    if matrix_arr.ndim != 2:
        raise ValueError(f"Decision matrix must be a 2D array, got {matrix_arr.ndim}D array.")

    n_alt, n_crit = matrix_arr.shape
    if n_alt == 0 or n_crit == 0:
        raise ValueError(f"Decision matrix cannot be empty, got shape {matrix_arr.shape}.")

    if np.isnan(matrix_arr).any() or np.isinf(matrix_arr).any():
        raise ValueError("Decision matrix contains NaN or Inf values.")

    if np.any(matrix_arr < 0):
        raise ValueError("Decision matrix must contain non-negative values for MULTIMOORA calculation.")

    criteria_arr = np.asarray(criteria)
    if criteria_arr.ndim != 1:
        raise ValueError(f"Criteria array must be 1D, got {criteria_arr.ndim}D array.")

    if len(criteria_arr) != n_crit:
        raise ValueError(
            f"Criteria length ({len(criteria_arr)}) does not match matrix columns ({n_crit})."
        )

    # Validate criteria values
    unique_vals = set(criteria_arr.tolist())
    if not unique_vals.issubset({1, -1}):
        raise ValueError(
            f"Criteria values must be 1 (benefit) or -1 (cost), found: {unique_vals}"
        )

    return matrix_arr, criteria_arr.astype(int)


def vector_normalization(matrix: np.ndarray) -> np.ndarray:
    """Perform vector normalization on matrix columns."""
    try:
        from pymcdm.normalizations import vector_normalization as pymcdm_vec_norm
        return pymcdm_vec_norm(matrix)
    except Exception:
        norms = np.sqrt(np.sum(matrix ** 2, axis=0))
        norms[norms == 0] = 1.0
        return matrix / norms


def critic_weights(matrix: np.ndarray) -> np.ndarray:
    """Calculate criterion weights using CRITIC method."""
    if matrix.shape[1] == 1:
        return np.array([1.0])
    try:
        from pymcdm.weights import critic_weights as pymcdm_critic
        res = pymcdm_critic(matrix)
        if np.isnan(res).any():
            raise ValueError("NaN in CRITIC weights")
        return res
    except Exception:
        # Fallback pure NumPy/SciPy implementation of CRITIC
        n, m = matrix.shape
        min_vals = np.min(matrix, axis=0)
        max_vals = np.max(matrix, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0

        # Min-Max Normalization
        norm_x = (matrix - min_vals) / range_vals

        # Standard deviation for each criterion
        std_dev = np.std(norm_x, axis=0, ddof=0)

        # Correlation matrix
        if n > 1:
            corr_matrix = np.corrcoef(norm_x, rowvar=False)
            if np.ndim(corr_matrix) == 0:
                corr_matrix = np.array([[1.0]])
            corr_matrix = np.nan_to_num(corr_matrix, nan=1.0)
        else:
            corr_matrix = np.ones((m, m))

        # Amount of information C_j = std_j * sum(1 - r_jk)
        info_amount = std_dev * np.sum(1.0 - corr_matrix, axis=1)

        total_info = np.sum(info_amount)
        if total_info == 0:
            return np.ones(m) / m

        return info_amount / total_info
