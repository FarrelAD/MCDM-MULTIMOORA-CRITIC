import numpy as np
from pymcdm.normalizations import vector_normalization
from pymcdm.weights import critic_weights
from tabulate import tabulate

def multimoora_critic(matrix, criteria):
    """
    Full implementation of MULTIMOORA with CRITIC weighting.
    """
    # 1. Weight Calculation (CRITIC)
    # CRITIC doesn't strictly need criteria type, but usually based on normalized matrix
    weights = critic_weights(matrix)
    
    # 2. Normalization
    norm_matrix = vector_normalization(matrix)
    
    # --- Phase 1: Ratio System (RS) ---
    benefit_mask = criteria == 1
    cost_mask = criteria == -1
    
    # RS Score = Σ(w_j * r_ij)_benefit - Σ(w_j * r_ij)_cost
    # Higher is better
    rs_scores = np.sum(norm_matrix[:, benefit_mask] * weights[benefit_mask], axis=1) - \
                np.sum(norm_matrix[:, cost_mask] * weights[cost_mask], axis=1)
    rs_ranks = np.argsort(np.argsort(-rs_scores)) + 1 # Rank 1 is highest score
    
    # --- Phase 2: Reference Point Approach (RPA) ---
    # Reference point r*
    ref_point = np.zeros(matrix.shape[1])
    ref_point[benefit_mask] = np.max(norm_matrix[:, benefit_mask], axis=0)
    ref_point[cost_mask] = np.min(norm_matrix[:, cost_mask], axis=0)
    
    # RPA Score = max_j(w_j * |r*_j - r_ij|)
    # Lower is better
    rpa_scores = np.max(weights * np.abs(ref_point - norm_matrix), axis=1)
    rpa_ranks = np.argsort(np.argsort(rpa_scores)) + 1 # Rank 1 is lowest score
    
    # --- Phase 3: Full Multiplicative Form (FMF) ---
    # FMF Score = Π(x_ij^w_j)_benefit / Π(x_ij^w_j)_cost
    # To avoid overflow/underflow, we use log scale: Σ(w_j * log(x_ij))_benefit - Σ(w_j * log(x_ij))_cost
    # Higher is better
    # Small epsilon to avoid log(0)
    eps = 1e-9
    log_matrix = np.log(matrix + eps)
    fmf_scores = np.sum(weights[benefit_mask] * log_matrix[:, benefit_mask], axis=1) - \
                 np.sum(weights[cost_mask] * log_matrix[:, cost_mask], axis=1)
    fmf_ranks = np.argsort(np.argsort(-fmf_scores)) + 1 # Rank 1 is highest score
    
    # --- Final Aggregation (Dominance Theory via Multiplicative Rank) ---
    # Many versions use the product of ranks for final ranking
    final_scores = rs_ranks * rpa_ranks * fmf_ranks
    final_ranks = np.argsort(np.argsort(final_scores)) + 1 # Rank 1 is lowest product
    
    return weights, rs_ranks, rpa_ranks, fmf_ranks, final_ranks

# Sample Data
matrix = np.array([
    [250, 16, 12],
    [200, 20, 8],
    [300, 18, 11],
    [270, 17, 10],
])

# 1 for benefit, -1 for cost
criteria = np.array([1, 1, -1])

if __name__ == "__main__":
    print("-" * 50)
    print("MCDM: MULTIMOORA with CRITIC Weighting")
    print("-" * 50)
    
    weights, rs, rpa, fmf, final = multimoora_critic(matrix, criteria)
    
    print("\nCalculated CRITIC Weights:")
    for i, w in enumerate(weights):
        print(f"Criterion C{i+1}: {w:.4f}")
        
    results = []
    for i in range(len(matrix)):
        results.append([
            f"Alternative A{i+1}",
            rs[i],
            rpa[i],
            fmf[i],
            final[i]
        ])
        
    print("\nRanking Results Summary:")
    print(tabulate(results, headers=["Alternative", "RS Rank", "RPA Rank", "FMF Rank", "MULTIMOORA Rank"], tablefmt="fancy_grid"))
    
    best_idx = np.argmin(final)
    print(f"\nConclusion: The best alternative is Alternative A{best_idx+1}")
