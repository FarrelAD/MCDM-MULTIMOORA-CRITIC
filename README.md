# MCDM-MULTIMOORA-CRITIC

A standard, fully-typed Python library and CLI tool for **Multi-Criteria Decision Making (MCDM)** combining **MULTIMOORA** (*Multi-Objective Optimization by Ratio Analysis plus the Full Multiplicative Form*) and **CRITIC** (*CRiteria Importance Through Intercriteria Correlation*) criterion weighting.

---

## Features

- **CRITIC Objective Weighting**: Automatically computes criterion weights based on standard deviations and inter-criterion correlations.
- **Three-Subsystem MULTIMOORA**:
  1. **Ratio System (RS)**: Linear evaluation of weighted normalized values.
  2. **Reference Point Approach (RPA)**: Chebyshev min-max distance from maximal/minimal reference points.
  3. **Full Multiplicative Form (FMF)**: Logarithmic utility product for benefit and cost criteria.
- **Dominance Theory Aggregation**: Combines component ranks via multiplicative rank product to derive final robust rankings.
- **Typed Result Dataclass**: `MultimooraResult` container supporting export to `pandas.DataFrame` and Unicode/ASCII tables (`tabulate`).
- **Command Line Interface (CLI)**: Executable command `multimoora-critic` with built-in `--demo`, CSV file inputs/outputs, and customizable table formatting.

---

## Installation

### Standard User Installation

From the repository root directory:

```bash
pip install .
```

### Development Installation

To install in editable mode with development dependencies (`pytest`, `ruff`, `mypy`):

```bash
pip install -e .[dev]
```

---

## Python API Usage

```python
import numpy as np
from multimoora_critic import multimoora_critic, MultimooraResult

# 1. Define Decision Matrix (Rows: Alternatives, Columns: Criteria)
matrix = np.array([
    [250.0, 16.0, 12.0],
    [200.0, 20.0,  8.0],
    [300.0, 18.0, 11.0],
    [270.0, 17.0, 10.0],
])

# 2. Define Criteria Types: 1 for Benefit (maximized), -1 for Cost (minimized)
criteria = np.array([1, 1, -1])

# 3. Compute MULTIMOORA + CRITIC Evaluation
result: MultimooraResult = multimoora_critic(matrix, criteria)

# Access CRITIC weights and rankings
print("CRITIC Weights:", result.weights)
print("MULTIMOORA Ranks:", result.final_ranks)
print("Best Alternative Index:", result.best_alternative_index)

# 4. Export as Pandas DataFrame
df = result.to_dataframe(alternative_names=["Alt A", "Alt B", "Alt C", "Alt D"])
print(df)

# 5. Print Formatted ASCII Table
print(result.to_table(tablefmt="fancy_grid"))
```

---

## Command Line Interface (CLI)

The package provides the `multimoora-critic` CLI binary (or `python -m multimoora_critic.cli`).

### Built-in Demo

Run the demo evaluation dataset:

```bash
multimoora-critic --demo
```

Output:
```text
--------------------------------------------------
MCDM: MULTIMOORA with CRITIC Weighting
--------------------------------------------------

Calculated CRITIC Weights:
Criterion C1: 0.2454
Criterion C2: 0.4582
Criterion C3: 0.2964

Ranking Results Summary:
╒════════════════╤════════════╤═══════════╤═════════════╤════════════╤═════════════╤════════════╤═══════════════════╕
│ Alternative    │   RS Score │   RS Rank │   RPA Score │   RPA Rank │   FMF Score │   FMF Rank │   MULTIMOORA Rank │
╞════════════════╪════════════╪═══════════╪═════════════╪════════════╪═════════════╪════════════╪═══════════════════╡
│ Alternative A1 │   0.126018 │         3 │  0.0237431  │          3 │     1.88882 │          4 │                 4 │
├────────────────┼────────────┼───────────┼─────────────┼────────────┼─────────────┼────────────┼───────────────────┤
│ Alternative A2 │   0.108115 │         4 │  0.0474863  │          4 │     2.05648 │          1 │                 3 │
├────────────────┼────────────┼───────────┼─────────────┼────────────┼─────────────┼────────────┼───────────────────┤
│ Alternative A3 │   0.152107 │         1 │  0.00177306 │          1 │     2.01332 │          2 │                 1 │
├────────────────┼────────────┼───────────┼─────────────┼────────────┼─────────────┼────────────┼───────────────────┤
│ Alternative A4 │   0.137549 │         2 │  0.0142459  │          2 │     1.98953 │          3 │                 2 │
╘════════════════╧════════════╧═══════════╧═════════════╧════════════╧═════════════╧════════════╧═══════════════════╛

Conclusion: The best alternative is Alternative A3
```

### Process CSV Decision Matrix

Evaluate custom matrix data from a CSV file and save results:

```bash
multimoora-critic --input data.csv --criteria 1 1 -1 --output results.csv --format github
```

---

## Mathematical Background

### 1. CRITIC Criterion Weighting

1. **Min-Max Normalization**:

$$
x'_{ij} = \frac{x_{ij} - \min_{k} x_{kj}}{\max_{k} x_{kj} - \min_{k} x_{kj}}
$$

2. **Information Measure**:

$$
C_{j} = \sigma_{j} \sum_{k=1}^{m} (1 - r_{jk})
$$

where $\sigma_{j}$ is the standard deviation of normalized criterion $j$, and $r_{jk}$ is the Pearson correlation coefficient between criteria $j$ and $k$.

3. **Weights**:

$$
w_{j} = \frac{C_{j}}{\sum_{k=1}^{m} C_{k}}
$$

### 2. Vector Normalization

$$
r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^{n} x_{kj}^{2}}}
$$

### 3. Ratio System (RS)

$$
y_{i} = \sum_{j \in B} w_{j} r_{ij} - \sum_{j \in C} w_{j} r_{ij}
$$

where $B$ is the set of benefit criteria and $C$ is the set of cost criteria. Higher score implies higher rank ($1 = \max y_{i}$).

### 4. Reference Point Approach (RPA)

Reference point $r^{*}_{j}$:

$$
r^{*}_{j} = \begin{cases} \max_{i} r_{ij}, & j \in B \\ \min_{i} r_{ij}, & j \in C \end{cases}
$$

RPA Score (Chebyshev Distance):

$$
z_{i} = \max_{j} \left( w_{j} \cdot \left| r^{*}_{j} - r_{ij} \right| \right)
$$

Lower score implies higher rank ($1 = \min z_{i}$).

### 5. Full Multiplicative Form (FMF)

FMF utility in log-scale:

$$
u_{i} = \sum_{j \in B} w_{j} \ln(x_{ij} + \epsilon) - \sum_{j \in C} w_{j} \ln(x_{ij} + \epsilon)
$$

Higher score implies higher rank ($1 = \max u_{i}$).

### 6. Dominance Theory (Final Aggregation)

Multiplicative rank product:

$$
P_{i} = R_{\text{RS}, i} \cdot R_{\text{RPA}, i} \cdot R_{\text{FMF}, i}
$$

Lower rank product implies overall highest MULTIMOORA rank ($1 = \min P_{i}$).

---

## Testing

Run the full unit test suite with `pytest`:

```bash
pytest -v
```

---

## License

Distributed under the MIT License.
