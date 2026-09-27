"""
Preprocessing and feature engineering.

Two things happen here, and keeping them straight matters:

  add_derived_features  works on the DataFrame and encodes DOMAIN knowledge —
                        ratios and counts a credit analyst would compute by hand.
  build_preprocessor    returns an unfitted sklearn transformer that is part of
                        the model Pipeline, so scaling and encoding are FITTED ON
                        TRAINING DATA ONLY and travel with the model.

TODO: Complete add_derived_features and build_preprocessor.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.config import (
    BILL_FEATURES,
    CATEGORICAL_FEATURES,
    PAY_AMT_FEATURES,
    PAY_FEATURES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TODO 1: Implement add_derived_features
# =============================================================================
# Create the six columns listed in config.DERIVED_FEATURES. These encode what a
# credit analyst would compute by hand, and they matter more than the model
# choice: a linear model on good features beats a boosted tree on raw columns
# more often than anyone expects.
#
#   utilisation_ratio   mean(BILL_AMT1..6) / LIMIT_BAL, clipped to [0, 5]
#                       How much of the credit line is in use. The single most
#                       informative ratio in consumer credit risk.
#   payment_ratio       PAY_AMT1 / BILL_AMT1, clipped to [0, 5]
#                       What share of the statement the customer actually pays.
#   max_delay           max(PAY_0, PAY_2..PAY_6)
#   n_months_delayed    count of PAY_* columns strictly greater than 0
#   avg_bill_amt        mean(BILL_AMT1..6)
#   avg_pay_amt         mean(PAY_AMT1..6)
#
# Two requirements that are easy to miss:
#
#   1. RETURN A COPY. Mutating the caller's frame makes this stage depend on how
#      many times it has been called, which is the kind of bug that only shows
#      up once the pipeline is on a schedule.
#
#   2. Divide by LIMIT_BAL.replace(0, np.nan), not by LIMIT_BAL. A zero limit
#      would give inf, which silently poisons the scaler downstream. NaN is an
#      honest "not applicable" and the imputer in build_preprocessor handles it.
#

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the six engineered features listed in config.DERIVED_FEATURES."""
    out = df.copy()

    avg_bill = out[BILL_FEATURES].mean(axis=1)
    out["utilisation_ratio"] = (avg_bill / out["LIMIT_BAL"].replace(0, np.nan)).clip(0, 5)

    out["payment_ratio"] = (out["PAY_AMT1"] / out["BILL_AMT1"].replace(0, np.nan)).clip(0, 5)

    out["max_delay"] = out[PAY_FEATURES].max(axis=1)

    out["n_months_delayed"] = (out[PAY_FEATURES] > 0).sum(axis=1)

    out["avg_bill_amt"] = avg_bill

    out["avg_pay_amt"] = out[PAY_AMT_FEATURES].mean(axis=1)

    return out


# =============================================================================
# TODO 2: Implement build_preprocessor
# =============================================================================
# Return an UNFITTED ColumnTransformer that:
#   - one-hot encodes the columns in CATEGORICAL_FEATURES that are present,
#     with handle_unknown="ignore" and sparse_output=False
#   - for every other column: median imputation, then StandardScaler
#   - drops anything else (remainder="drop")
#
# handle_unknown="ignore" is not defensive clutter. An EDUCATION code the model
# has never seen will appear in production eventually — a new product, a data
# migration, a typo upstream. Without it the transformer raises and the whole
# request fails; with it the unknown category becomes all-zeros and the model
# still answers.
#
# Return it UNFITTED. build_pipeline in training.py puts it inside the sklearn
# Pipeline, so it gets fitted on the training fold only and travels with the
# model. Fit it here and you have leaked test statistics into training.
#
def build_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    """Unfitted transformer: one-hot the categoricals, impute and scale the rest."""
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in feature_columns]
    numeric_cols = [c for c in feature_columns if c not in categorical_cols]

    transformers = []

    if categorical_cols:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ))

    if numeric_cols:
        transformers.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric_cols,
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """The full feature step: derive, then drop nothing and let the model decide."""
    return add_derived_features(df)
