import os
import re
import pandas as pd
from config import ACTION_CANONICAL_MAPPING, NO_OPP_KEYWORDS, CLEANED_CSV
from typing import Any

def canonicalize_action(a: Any) -> str:
    if pd.isna(a):
        return "UNKNOWN"
    s = str(a).strip()
    if s == "":
        return "UNKNOWN"
    s_norm = re.sub(r"\s+", " ", s)
    s_key = re.sub(r"[^\w\s\-]", "", s_norm).upper()
    s_key = s_key.replace("_", " ").replace("-", " ").strip()
    if s_key in ACTION_CANONICAL_MAPPING:
        return ACTION_CANONICAL_MAPPING[s_key]
    if s.isupper() and len(s) <= 4:
        return s
    return s_norm.title()

def _normalize_opportunity_stage(s: Any) -> str:
    if pd.isna(s):
        return "no_opp"
    st = str(s).strip()
    if st == "":
        return "no_opp"
    sl = st.lower()

    try:
        no_opp_lower = _normalize_opportunity_stage._no_opp_lower
    except AttributeError:
        no_opp_lower = [str(k).strip().lower() for k in NO_OPP_KEYWORDS if str(k).strip() != ""]
        _normalize_opportunity_stage._no_opp_lower = no_opp_lower

    if "won" in sl or "win" in sl:
        return "Won"
    if "lost" in sl or "lose" in sl:
        return "Lost"

    for kw in no_opp_lower:
        if not kw:
            continue
        try:
            if re.search(r"\b" + re.escape(kw) + r"\b", sl):
                return "no_opp"
        except re.error:
            if kw in sl:
                return "no_opp"
    return st

def to_int_flag(x):
    try:
        return int(float(x))
    except Exception:
        s = str(x).strip().lower()
        return 1 if s in {"1", "true", "yes"} else 0

def drop_rows_without_dates(df, src_col="activity_date"):
    df = df[~df[src_col].isna()].copy()
    return df

def load_and_clean(path: str, drop_missing_source: bool = False) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found in {os.getcwd()}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])
    required_cols = {"account_id", "SourceSystem", "activity_date", "types", "Country", "solution", "opportunity_stage", "is_lead"}
    for c in required_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce", dayfirst=False)

    if drop_missing_source:
        df = df[(df["types"].astype(str).str.strip() != "") & (df["SourceSystem"].astype(str).str.strip() != "")].copy()
    else:
        df = df[df["types"].astype(str).str.strip() != ""].copy()
        df["SourceSystem"] = df["SourceSystem"].replace({"": "UNKNOWN", "[NULL]": "UNKNOWN"}).fillna("UNKNOWN")

    df["is_lead"] = df["is_lead"].apply(to_int_flag)
    df["Country"] = df["Country"].replace({"": "UNKNOWN", "[NULL]": "UNKNOWN"}).fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["solution"] = df["solution"].replace({"": "UNKNOWN", "[NULL]": "UNKNOWN"}).fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["types"] = df["types"].apply(canonicalize_action)
    df["opportunity_stage"] = df["opportunity_stage"].apply(_normalize_opportunity_stage)

    dup_subset = ["account_id", "activity_date", "types", "who_id", "opportunity_id"]
    dup_subset = [c for c in dup_subset if c in df.columns]
    df = df.drop_duplicates(subset=dup_subset)

    df["account_id"] = df["account_id"].fillna("UNKNOWN_ACCOUNT").astype(str)

    df = df.sort_values(["account_id", "activity_date"], na_position="last").reset_index(drop=True)
    last_touch_idx = df.groupby("account_id")["activity_date"].transform("max") == df["activity_date"]
    df["is_last_touch_in_data"] = last_touch_idx.astype(int)
    df["touch_index"] = df.groupby("account_id").cumcount() + 1
    df["days_since_prev_touch"] = df.groupby("account_id")["activity_date"].diff().dt.total_seconds().div(86400).fillna(0).clip(lower=0)
    df["num_past_touches"] = df["touch_index"] - 1

    df = drop_rows_without_dates(df, src_col="activity_date")

    try:
        df.to_csv(CLEANED_CSV, index=False)
    except Exception:
        pass

    return df
