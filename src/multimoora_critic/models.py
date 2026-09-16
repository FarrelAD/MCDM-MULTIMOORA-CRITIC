from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from tabulate import tabulate


@dataclass
class MultimooraResult:
    """Dataclass holding the results of MULTIMOORA with CRITIC weighting."""

    weights: np.ndarray
    rs_scores: np.ndarray
    rs_ranks: np.ndarray
    rpa_scores: np.ndarray
    rpa_ranks: np.ndarray
    fmf_scores: np.ndarray
    fmf_ranks: np.ndarray
    final_scores: np.ndarray
    final_ranks: np.ndarray

    @property
    def best_alternative_index(self) -> int:
        """Return index of the best alternative (lowest rank number)."""
        return int(np.argmin(self.final_ranks))

    def _default_alternative_names(self) -> List[str]:
        return [f"Alternative A{i + 1}" for i in range(len(self.final_ranks))]

    def to_dataframe(self, alternative_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert result scores and ranks to a structured pandas DataFrame."""
        if alternative_names is None:
            names = self._default_alternative_names()
        else:
            if len(alternative_names) != len(self.final_ranks):
                raise ValueError(
                    f"Expected {len(self.final_ranks)} alternative names, got {len(alternative_names)}"
                )
            names = list(alternative_names)

        df = pd.DataFrame({
            "Alternative": names,
            "RS Score": self.rs_scores,
            "RS Rank": self.rs_ranks,
            "RPA Score": self.rpa_scores,
            "RPA Rank": self.rpa_ranks,
            "FMF Score": self.fmf_scores,
            "FMF Rank": self.fmf_ranks,
            "MULTIMOORA Rank": self.final_ranks,
        })
        return df

    def to_table(self, alternative_names: Optional[List[str]] = None, tablefmt: str = "fancy_grid") -> str:
        """Format results as a tabulated string."""
        df = self.to_dataframe(alternative_names=alternative_names)
        return tabulate(df, headers="keys", tablefmt=tablefmt, showindex=False)
