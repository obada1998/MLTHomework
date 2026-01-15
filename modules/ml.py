import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as _tt
from config import TEST_SIZE, RANDOM_STATE

def prepare_ml_data(df: pd.DataFrame, test_size=TEST_SIZE, random_state=RANDOM_STATE) -> Dict[str, Any]:
    df = df.copy()
    if df.empty:
        empty_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if hasattr(OneHotEncoder, "sparse_output") else OneHotEncoder(handle_unknown="ignore", sparse=False)
        return {
            "train_df": pd.DataFrame(), "test_df": pd.DataFrame(),
            "X_train": np.zeros((0,0)), "X_test": np.zeros((0,0)),
            "y_next_train": pd.Series([], dtype=str), "y_next_test": pd.Series([], dtype=str),
            "y_out_train": pd.Series([], dtype=int), "y_out_test": pd.Series([], dtype=int),
            "ohe": empty_ohe, "feature_names": [], "cat_cols": ["Country","solution","last_action"], "num_cols": ["num_past_touches","days_since_prev_touch","is_lead"]
        }

    df["opportunity_id"] = df.get("opportunity_id", pd.Series([pd.NA]*len(df))).replace({np.nan: pd.NA, "": pd.NA})
    df["journey_id"] = df["opportunity_id"].where(df["opportunity_id"].notna() & df["opportunity_id"].astype(str).str.strip().ne(""), df["account_id"])
    df = df.sort_values(["journey_id", "activity_date"]).reset_index(drop=True)
    df["next_action"] = df.groupby("journey_id")["types"].shift(-1)
    df["last_action"] = df.groupby("journey_id")["types"].shift(1).fillna("NONE")

    df_rows = df.dropna(subset=["next_action"]).copy()
    df_rows["candidate_action"] = df_rows["next_action"].astype(str).fillna("NONE")
    cat_cols = ["Country", "solution", "last_action", "candidate_action"]
    num_cols = ["num_past_touches", "days_since_prev_touch", "is_lead"]


    if df_rows.empty:
        empty_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if hasattr(OneHotEncoder, "sparse_output") else OneHotEncoder(handle_unknown="ignore", sparse=False)
        return {
            "train_df": pd.DataFrame(), "test_df": pd.DataFrame(),
            "X_train": np.zeros((0,0)), "X_test": np.zeros((0,0)),
            "y_next_train": pd.Series([], dtype=str), "y_next_test": pd.Series([], dtype=str),
            "y_out_train": pd.Series([], dtype=int), "y_out_test": pd.Series([], dtype=int),
            "ohe": empty_ohe, "feature_names": [], "cat_cols": ["Country","solution","last_action"], "num_cols": ["num_past_touches","days_since_prev_touch","is_lead"]
        }

    ever_won_j = df.groupby("journey_id")["opportunity_stage"].apply(lambda s: (s.astype(str).str.lower() == "won").any()).rename("ever_won").reset_index()
    df_rows = df_rows.merge(ever_won_j, on="journey_id", how="left")

    if "num_past_touches" not in df_rows.columns:
        df_rows["num_past_touches"] = df_rows.groupby("journey_id").cumcount()
    if "days_since_prev_touch" not in df_rows.columns:
        df_rows["days_since_prev_touch"] = df_rows.groupby("journey_id")["activity_date"].diff().dt.total_seconds().div(86400).fillna(0).clip(lower=0)

    df_rows[["Country","solution","last_action"]] = df_rows[["Country","solution","last_action"]].fillna({"Country":"UNKNOWN","solution":"UNKNOWN","last_action":"NONE"})

    journey_meta = df_rows.groupby("journey_id")["ever_won"].max().reset_index()
    journey_ids = journey_meta["journey_id"].values
    stratify_vals = journey_meta["ever_won"].astype(int).values if len(journey_meta) > 0 else None

    if stratify_vals is not None and len(np.unique(stratify_vals)) > 1:
        train_j, test_j = _tt(journey_ids, test_size=test_size, random_state=random_state, stratify=stratify_vals)
    else:
        train_j, test_j = _tt(journey_ids, test_size=test_size, random_state=random_state)

    train = df_rows[df_rows["journey_id"].isin(train_j)].reset_index(drop=True)
    test = df_rows[df_rows["journey_id"].isin(test_j)].reset_index(drop=True)

    cat_cols = ["Country","solution","last_action"]
    num_cols = ["num_past_touches","days_since_prev_touch","is_lead"]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    if not train.empty:
        ohe.fit(train[cat_cols])
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols)) if (not train.empty) else []
        feature_names = cat_feature_names + num_cols
        try:
            ohe._training_df = train.copy()
        except Exception:
            ohe._training_df = train 

    def build_feature_matrix(df_in):
        try:
            cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            cat_feature_names = []
        n_cat = len(cat_feature_names)
        n_num = len(num_cols)
        if df_in.empty:
            return np.zeros((0, n_cat + n_num))
        try:
            cat = ohe.transform(df_in[cat_cols])
        except Exception:
            cat = np.zeros((len(df_in), n_cat))
        num = df_in[num_cols].astype(float).to_numpy()
        X = np.hstack([cat, num])
        return X

    X_train = build_feature_matrix(train)
    X_test = build_feature_matrix(test)

    y_next_train = train["next_action"].astype(str).reset_index(drop=True)
    y_next_test = test["next_action"].astype(str).reset_index(drop=True)
    y_out_train = train["ever_won"].astype(int).reset_index(drop=True)
    y_out_test = test["ever_won"].astype(int).reset_index(drop=True)

    try:
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols)) if (not train.empty) else []
    except Exception:
        cat_feature_names = []
    feature_names = cat_feature_names + num_cols

    train_df = train.copy()

    return {
        "train_df": train_df, "test_df": test, "X_train": X_train, "X_test": X_test,
        "y_next_train": y_next_train, "y_next_test": y_next_test, "y_out_train": y_out_train, "y_out_test": y_out_test,
        "ohe": ohe, "feature_names": feature_names, "cat_cols": cat_cols, "num_cols": num_cols
    }


def train_decision_trees(X_train: np.ndarray, y_next_train: pd.Series, y_out_train: pd.Series, random_state=RANDOM_STATE):
    dt_next = None
    dt_outcome = None
    try:
        if X_train.shape[0] > 0 and len(y_next_train) > 0:
            dt_next = DecisionTreeClassifier(max_depth=6, random_state=random_state)
            dt_next.fit(X_train, y_next_train)
    except Exception:
        dt_next = None

    try:
        if X_train.shape[0] > 0 and len(y_out_train) > 0:
            unique_out = np.unique(np.asarray(y_out_train))
            if len(unique_out) > 1:
                dt_outcome = DecisionTreeClassifier(max_depth=8, random_state=random_state, class_weight="balanced")
                dt_outcome.fit(X_train, y_out_train)
            else:
                print(f"DIAG: skipping dt_outcome training because y_out_train has single class: {unique_out}")
                dt_outcome = None
        else:
            dt_outcome = None
    except Exception as ex:
        print("DIAG: dt_outcome training failed:", ex)
        dt_outcome = None

    return dt_next, dt_outcome

def extract_feature_importances(dt, feature_names: List[str]):

    if dt is None:
        return []
    importances = getattr(dt, "feature_importances_", None)
    if importances is None:
        return []
    fnames = list(feature_names or [])
    if len(fnames) < len(importances):
        fnames = fnames + [f"f_{i}" for i in range(len(fnames), len(importances))]
    elif len(fnames) > len(importances):
        fnames = fnames[:len(importances)]
    pairs = list(zip(fnames, importances))
    return sorted(pairs, key=lambda x: -float(x[1]))
