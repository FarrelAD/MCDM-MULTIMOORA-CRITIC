import numpy as np
import pytest

from multimoora_critic import multimoora_critic, MultimooraResult


@pytest.fixture
def sample_data():
    matrix = np.array([
        [250.0, 16.0, 12.0],
        [200.0, 20.0, 8.0],
        [300.0, 18.0, 11.0],
        [270.0, 17.0, 10.0],
    ])
    criteria = np.array([1, 1, -1])
    return matrix, criteria


def test_multimoora_critic_sample_data(sample_data):
    matrix, criteria = sample_data
    result = multimoora_critic(matrix, criteria)

    assert isinstance(result, MultimooraResult)
    assert len(result.weights) == 3
    assert np.isclose(np.sum(result.weights), 1.0)

    n_alt = len(matrix)
    expected_ranks = set(range(1, n_alt + 1))

    assert set(result.rs_ranks) == expected_ranks
    assert set(result.rpa_ranks) == expected_ranks
    assert set(result.fmf_ranks) == expected_ranks
    assert set(result.final_ranks) == expected_ranks

    assert isinstance(result.best_alternative_index, int)
    assert 0 <= result.best_alternative_index < n_alt


def test_result_export(sample_data):
    matrix, criteria = sample_data
    result = multimoora_critic(matrix, criteria)

    # Test DataFrame export
    df = result.to_dataframe()
    expected_columns = [
        "Alternative",
        "RS Score",
        "RS Rank",
        "RPA Score",
        "RPA Rank",
        "FMF Score",
        "FMF Rank",
        "MULTIMOORA Rank",
    ]
    assert list(df.columns) == expected_columns
    assert len(df) == 4
    assert df["Alternative"].iloc[0] == "Alternative A1"

    # Test custom names
    custom_names = ["Alt_Alpha", "Alt_Beta", "Alt_Gamma", "Alt_Delta"]
    df_custom = result.to_dataframe(alternative_names=custom_names)
    assert list(df_custom["Alternative"]) == custom_names

    # Test invalid custom names length
    with pytest.raises(ValueError, match="Expected 4 alternative names"):
        result.to_dataframe(alternative_names=["A1", "A2"])

    # Test Table export
    table_str = result.to_table()
    assert isinstance(table_str, str)
    assert "Alternative A1" in table_str
    assert "MULTIMOORA Rank" in table_str


def test_invalid_criteria_values(sample_data):
    matrix, _ = sample_data
    invalid_criteria = np.array([1, 0, 2])
    with pytest.raises(ValueError, match="Criteria values must be 1 .* or -1"):
        multimoora_critic(matrix, invalid_criteria)


def test_criteria_dimension_mismatch(sample_data):
    matrix, _ = sample_data
    mismatched_criteria = np.array([1, 1])
    with pytest.raises(ValueError, match="Criteria length .* does not match matrix columns"):
        multimoora_critic(matrix, mismatched_criteria)


def test_invalid_matrix_ndim():
    with pytest.raises(ValueError, match="must be a 2D array"):
        multimoora_critic(np.array([1, 2, 3]), np.array([1, 1, 1]))


def test_empty_matrix():
    with pytest.raises(ValueError, match="cannot be empty"):
        multimoora_critic(np.empty((0, 3)), np.array([1, 1, -1]))


def test_negative_matrix_values():
    matrix = np.array([
        [-10.0, 16.0],
        [200.0, 20.0],
    ])
    criteria = np.array([1, -1])
    with pytest.raises(ValueError, match="non-negative values"):
        multimoora_critic(matrix, criteria)


def test_single_criterion_matrix():
    matrix = np.array([
        [100.0],
        [200.0],
        [150.0],
    ])
    criteria = np.array([1])
    result = multimoora_critic(matrix, criteria)
    assert len(result.weights) == 1
    assert result.weights[0] == 1.0
    assert len(result.final_ranks) == 3
