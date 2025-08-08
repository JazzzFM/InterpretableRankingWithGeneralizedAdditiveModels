from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pygam import LogisticGAM, s, f
from pygam.terms import TermList

@dataclass
class FeatureSpec:
    numeric: List[str]
    categorical: List[str]
    target: str

class GAMTrainer:
    """Trains a Logistic GAM with numeric splines and categorical factors.
    Stores encoders and imputers to package as a PyFunc model for serving.
    """
    def __init__(self, spec: FeatureSpec, test_size: float = 0.2, random_state: int = 42):
        self.spec = spec
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.term_index: Dict[str, int] = {}
        self.gam: LogisticGAM | None = None
        self.numeric_median: Dict[str, float] = {}
        self.categorical_mode: Dict[str, int] = {}
        self.cols: List[str] = []

    # ---------- preprocessing ----------
    def _fit_encoders_and_imputers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # imputers (medians and modes) learned from training data
        for col in self.spec.numeric:
            self.numeric_median[col] = float(df[col].median())
            df[col] = df[col].fillna(self.numeric_median[col])
        for col in self.spec.categorical:
            # coerce to string for consistent label space
            series = df[col].astype(str)
            # learn mode BEFORE encoding (most frequent token)
            self.categorical_mode[col] = int(series.mode(dropna=True).index[0]) if series.mode(dropna=True).size>0 else 0
            le = LabelEncoder()
            df[col] = le.fit_transform(series.fillna("NA_SENTINEL"))
            self.label_encoders[col] = le
        return df

    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.spec.numeric:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = df[col].astype(float).fillna(self.numeric_median[col])
        for col in self.spec.categorical:
            if col not in df.columns:
                df[col] = "NA_SENTINEL"
            series = df[col].astype(str).fillna("NA_SENTINEL")
            le = self.label_encoders[col]
            # unseen labels → map to 'NA_SENTINEL' if present; otherwise to 0
            known = set(le.classes_)
            if "NA_SENTINEL" not in known:
                # expand encoder to include sentinel if missing
                le.classes_ = np.append(le.classes_, "NA_SENTINEL")
            mapped = np.array([x if x in set(le.classes_) else "NA_SENTINEL" for x in series])
            df[col] = le.transform(mapped)
        return df

    # ---------- design matrix / terms ----------
    def build_design_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        cols = self.spec.numeric + self.spec.categorical
        self.cols = cols
        X = df[cols].to_numpy()
        self.term_index = {c: i for i, c in enumerate(cols)}
        return X, cols

    def build_terms(self):
        """Build GAM terms with splines for numeric and factors for categorical variables"""
        terms = []
        # Splines (smooth terms) for numeric variables
        for col in self.spec.numeric:
            terms.append(s(self.term_index[col]))
        # Factor terms for categorical variables  
        for col in self.spec.categorical:
            terms.append(f(self.term_index[col]))
        return TermList(*terms)

    # ---------- training ----------
    def fit(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        y = df[self.spec.target].to_numpy()
        df = self._fit_encoders_and_imputers(df)

        X, _ = self.build_design_matrix(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        base = LogisticGAM(self.build_terms())
        lam_grid = np.logspace(-2, 4, 7)
        self.gam = base.gridsearch(X_train, y_train, lam=lam_grid)

        train_df = pd.DataFrame({ "y": y_train, "p": self.gam.predict_proba(X_train) })
        test_df  = pd.DataFrame({ "y": y_test,  "p": self.gam.predict_proba(X_test)  })
        train_df["rank"] = (-train_df["p"]).rank(method="first")
        test_df["rank"]  = (-test_df["p"]).rank(method="first")
        return train_df, test_df

    # ---------- interpretability ----------
    def partial_effect(self, feature: str, grid: int = 100):
        assert self.gam is not None, "Model not fitted."
        idx = self.term_index[feature]
        XX = self.gam.generate_X_grid(term=idx, n=grid)
        pdp = self.gam.partial_dependence(term=idx, X=XX)
        lb, ub = self.gam.partial_dependence(term=idx, X=XX, width=0.95, mc_samples=200, width_type="confidence")
        return XX[:, idx], pdp, (lb, ub)

    # ---------- inference ----------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        assert self.gam is not None, "Model not fitted."
        proc = self._apply_preprocessing(df)
        X, _ = self.build_design_matrix(proc)
        return self.gam.predict_proba(X)

# ---------- MLflow PyFunc wrapper (includes preprocessing) ----------
import mlflow.pyfunc
class GAMPyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, gam: LogisticGAM, numeric: List[str], categorical: List[str],
                 label_encoders: Dict[str, LabelEncoder],
                 numeric_median: Dict[str, float]):
        self.gam = gam
        self.numeric = numeric
        self.categorical = categorical
        self.label_encoders = label_encoders
        self.numeric_median = numeric_median

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.numeric:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = out[col].astype(float).fillna(self.numeric_median[col])
        for col in self.categorical:
            if col not in out.columns:
                out[col] = "NA_SENTINEL"
            series = out[col].astype(str).fillna("NA_SENTINEL")
            le = self.label_encoders[col]
            if "NA_SENTINEL" not in set(le.classes_):
                le.classes_ = np.append(le.classes_, "NA_SENTINEL")
            mapped = np.array([x if x in set(le.classes_) else "NA_SENTINEL" for x in series])
            out[col] = le.transform(mapped)
        # design matrix order
        cols = self.numeric + self.categorical
        return out[cols]

    def predict(self, context, model_input):
        proc = self._apply(pd.DataFrame(model_input))
        X = proc.to_numpy()
        return self.gam.predict_proba(X)
