import os
import pandas as pd
import pytest

from multimoora_critic.cli import main


def test_cli_demo(capsys):
    exit_code = main(["--demo"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "MCDM: MULTIMOORA with CRITIC Weighting" in captured.out
    assert "Calculated CRITIC Weights:" in captured.out
    assert "Ranking Results Summary:" in captured.out
    assert "Conclusion: The best alternative is Alternative A3" in captured.out


def test_cli_csv_input_output(tmp_path, capsys):
    input_csv = tmp_path / "matrix.csv"
    output_csv = tmp_path / "results.csv"

    # Write test input matrix CSV
    df_in = pd.DataFrame([
        [250.0, 16.0, 12.0],
        [200.0, 20.0, 8.0],
        [300.0, 18.0, 11.0],
    ])
    df_in.to_csv(input_csv, index=False)

    exit_code = main([
        "--input", str(input_csv),
        "--criteria", "1", "1", "-1",
        "--output", str(output_csv),
        "--format", "plain"
    ])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Ranking Results Summary:" in captured.out
    assert os.path.exists(output_csv)

    df_out = pd.read_csv(output_csv)
    assert "Alternative" in df_out.columns
    assert "MULTIMOORA Rank" in df_out.columns
    assert len(df_out) == 3


def test_cli_input_with_alternative_names(tmp_path, capsys):
    input_csv = tmp_path / "named_matrix.csv"

    df_in = pd.DataFrame({
        "Name": ["Supplier X", "Supplier Y", "Supplier Z"],
        "C1": [250.0, 200.0, 300.0],
        "C2": [16.0, 20.0, 18.0],
    })
    df_in.to_csv(input_csv, index=False)

    exit_code = main([
        "--input", str(input_csv),
        "--criteria", "1", "-1"
    ])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Supplier X" in captured.out
    assert "Conclusion: The best alternative is" in captured.out


def test_cli_no_args():
    exit_code = main([])
    assert exit_code == 1


def test_cli_missing_input_file():
    exit_code = main(["--input", "nonexistent_file_123.csv", "--criteria", "1"])
    assert exit_code == 1


def test_cli_missing_criteria():
    exit_code = main(["--input", "some_file.csv"])
    assert exit_code == 1
