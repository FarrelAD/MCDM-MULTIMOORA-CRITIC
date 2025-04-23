import numpy as np
from pymcdm.normalizations import vector_normalization
from pymcdm.methods import MOORA

matrix = np.array([
    [250, 16, 12],
    [200, 20, 8],
    [300, 18, 11],
])

weights = np.array([0.5, 0.3, 0.2])

# 1 for benefit, -1 for cost
criteria = np.array([1, 1, -1])

option = input("Manual ? (y/n) ")

if option.lower() == "y":
    # Manual MOORA
    norm_matrix = vector_normalization(matrix)
    weighted_matrix = norm_matrix * weights

    # Convert criteria to boolean masks
    benefit_mask = criteria == 1
    cost_mask = criteria == -1

    scores = np.sum(weighted_matrix[:, benefit_mask], axis=1) - np.sum(weighted_matrix[:, cost_mask], axis=1)
    ranked_indices = np.argsort(scores)[::-1]

    for idx in ranked_indices:
        print(f"Alternative A{idx+1} → Score: {scores[idx]:.4f}")
else:
    # Using pymcdm's MOORA method
    model = MOORA()
    scores = model(matrix, weights, criteria)
    ranking = np.argsort(scores)[::-1]

    for i, alt_index in enumerate(ranking):
        print(f"Rank {i+1}: Alternative A{alt_index+1} → Score: {scores[alt_index]:.4f}")
