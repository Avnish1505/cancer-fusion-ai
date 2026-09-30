"""
Metadata encoding for the v2 fusion model.

Why this replaces the v1 encoding:

v1 one-hot encoded sex/localization with "unknown" as just another category
and imputed missing age with the training median. In HAM10000 the rows with
sex == "unknown" are 47 nv + 10 bkl and zero malignant; rows with missing age
are 45 nv + 10 bkl + 2 mel. So "unknown" is not neutral in v1, it is a weak
"benign" signal learned from ~57 odd images. Any caller that leaves a field
blank (or a frontend that fills in a default) is steering the prediction.

v2 encodes each field with an explicit missing flag and, during training,
randomly blanks fields (metadata dropout). "Missing" then means "not
provided" across the whole label distribution, and the model is trained to
work with anything from full metadata to none.

The fitted encoder is a plain dict (JSON-serialisable) saved inside the
checkpoint, so serving never has to recompute the training split to
rebuild it.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "unknown", "nan", "none", "null"}


def _clean_category(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip().lower()
    return None if text in MISSING_TOKENS else text


def _clean_age(value: Any) -> float | None:
    if value is None:
        return None
    try:
        age = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(age) or age < 0 or age > 120:
        return None
    return age


class MetadataEncoder:
    """age (standardised) + age_missing, sex one-hot + sex_missing,
    localization one-hot + loc_missing."""

    def __init__(self, age_mean: float, age_std: float, sexes: list[str], sites: list[str]):
        self.age_mean = float(age_mean)
        self.age_std = float(age_std) if age_std > 0 else 1.0
        self.sexes = list(sexes)
        self.sites = list(sites)

    # ---- construction / persistence ----
    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "MetadataEncoder":
        ages = [a for a in (_clean_age(v) for v in train_df["age"]) if a is not None]
        sexes = sorted({s for s in (_clean_category(v) for v in train_df["sex"]) if s})
        sites = sorted({s for s in (_clean_category(v) for v in train_df["localization"]) if s})
        return cls(float(np.mean(ages)), float(np.std(ages)), sexes, sites)

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "sexes": self.sexes,
            "sites": self.sites,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetadataEncoder":
        if d.get("version") != 2:
            raise ValueError(f"Unsupported metadata encoder version: {d.get('version')}")
        return cls(d["age_mean"], d["age_std"], d["sexes"], d["sites"])

    # ---- encoding ----
    @property
    def columns(self) -> list[str]:
        return (
            ["age_scaled", "age_missing"]
            + [f"sex_{s}" for s in self.sexes] + ["sex_missing"]
            + [f"loc_{s}" for s in self.sites] + ["loc_missing"]
        )

    @property
    def dim(self) -> int:
        return len(self.columns)

    def validate(self, sex: Any, localization: Any) -> list[str]:
        """Returns human-readable problems with categorical inputs (empty if fine).
        Missing is always allowed; an unrecognised value is not, because silently
        zeroing it would look like a real answer to the model."""
        problems = []
        s = _clean_category(sex)
        if s is not None and s not in self.sexes:
            problems.append(f"sex '{sex}' not recognised; expected one of {self.sexes} or empty")
        loc = _clean_category(localization)
        if loc is not None and loc not in self.sites:
            problems.append(f"localization '{localization}' not recognised; expected one of {self.sites} or empty")
        return problems

    def encode(self, age: Any = None, sex: Any = None, localization: Any = None,
               drop: tuple[bool, bool, bool] = (False, False, False)) -> np.ndarray:
        """One row. `drop` forces (age, sex, localization) to missing — used for
        metadata dropout in training and for the no-metadata evaluation."""
        vec = np.zeros(self.dim, dtype=np.float32)
        i = 0
        a = None if drop[0] else _clean_age(age)
        if a is None:
            vec[i + 1] = 1.0
        else:
            vec[i] = (a - self.age_mean) / self.age_std
        i += 2

        s = None if drop[1] else _clean_category(sex)
        if s in self.sexes:
            vec[i + self.sexes.index(s)] = 1.0
        else:
            vec[i + len(self.sexes)] = 1.0
        i += len(self.sexes) + 1

        loc = None if drop[2] else _clean_category(localization)
        if loc in self.sites:
            vec[i + self.sites.index(loc)] = 1.0
        else:
            vec[i + len(self.sites)] = 1.0
        return vec

    def encode_row(self, row: pd.Series, drop=(False, False, False)) -> np.ndarray:
        return self.encode(row.get("age"), row.get("sex"), row.get("localization"), drop)
