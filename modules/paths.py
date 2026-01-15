from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from config import DEFAULT_BASE_WEIGHT

def normalize_path(path: List[str], dedupe_all: bool = False) -> Tuple[str, ...]:
    if dedupe_all:
        seen = set()
        out = []
        for a in path:
            if a not in seen:
                out.append(a)
                seen.add(a)
        return tuple(out)
    out = []
    for a in path:
        if not out or out[-1] != a:
            out.append(a)
    return tuple(out)

def _path_until_close(sub_df: pd.DataFrame) -> List[str]:
    sub = sub_df.sort_values("activity_date").reset_index(drop=True)
    if sub.empty:
        return []
    is_close = sub["opportunity_stage"].astype(str).str.lower().isin({"won", "lost"})
    if is_close.any():
        first_pos = int(is_close.idxmax() - sub.index[0])  
        return sub.iloc[: first_pos + 1]["types"].tolist()
    else:
        return sub["types"].tolist()

def extract_account_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return pd.DataFrame(columns=["journey_id", "raw_path", "path", "path_len", "path_str",
                                     "account_id", "Country", "solution", "first_touch_date",
                                     "is_lead", "ever_won", "time_to_win_days"])

    df["opportunity_id"] = df.get("opportunity_id", pd.Series([pd.NA] * len(df))).replace({np.nan: pd.NA, "": pd.NA})
    df["journey_id"] = df["opportunity_id"].where(df["opportunity_id"].notna() & df["opportunity_id"].astype(str).str.strip().ne(""), df["account_id"])

    ever_won_j = (df.groupby("journey_id")["opportunity_stage"]
                   .apply(lambda s: (s.astype(str).str.lower() == "won").any())
                   .rename("ever_won")
                   .reset_index())

    first_touch = df.groupby("journey_id")["activity_date"].min().rename("first_touch_date")
    won_mask = df["opportunity_stage"].astype(str).str.lower() == "won"
    if won_mask.any():
        first_won = df.loc[won_mask].groupby("journey_id")["activity_date"].min().rename("first_won_date")
    else:
        first_won = pd.Series(dtype="datetime64[ns]")

    time_to_win = pd.concat([first_touch, first_won], axis=1)
    time_to_win = time_to_win.reset_index()
    time_to_win["time_to_win_days"] = (time_to_win["first_won_date"] - time_to_win["first_touch_date"]).dt.days

    grouped = df.sort_values(["journey_id", "activity_date"]).groupby("journey_id", sort=False)
    records = []
    for jid, g in grouped:
        raw = _path_until_close(g[["activity_date", "opportunity_stage", "types"]])
        records.append({"journey_id": jid, "raw_path": raw})
    paths = pd.DataFrame.from_records(records)

    meta = (df.sort_values(["journey_id", "activity_date"])
              .groupby("journey_id", sort=False)
              .agg(account_id=("account_id", "last"),
                   Country=("Country", "last"),
                   solution=("solution", "last"),
                   first_touch_date=("activity_date", "min"),
                   is_lead=("is_lead", "last"))
              .reset_index())


    acc = (paths.merge(meta, on="journey_id", how="left")
               .merge(ever_won_j, on="journey_id", how="left")
               .merge(time_to_win[["journey_id", "time_to_win_days"]], on="journey_id", how="left"))


    acc["path"] = acc["raw_path"].apply(lambda p: normalize_path(p, dedupe_all=False))
    acc["path_len"] = acc["path"].apply(lambda p: len(p))
    acc["ever_won"] = acc["ever_won"].fillna(False).astype(bool)
    acc["path_str"] = acc["path"].apply(lambda p: " > ".join(p) if p else "")

    return acc

def top_k_paths_by_group(acc_paths: pd.DataFrame, group_cols: List[str], k: int = 5) -> pd.DataFrame:
    for c in group_cols:
        if c not in acc_paths.columns:
            acc_paths[c] = "UNKNOWN"
    agg = (acc_paths.groupby(group_cols + ["path_str"])
           .agg(count=("account_id", "size"),
                wins=("ever_won", "sum"),
                median_len=("path_len", "median"),
                avg_time_to_win=("time_to_win_days", "median"))
           .reset_index())
    agg["conversion_rate"] = agg["wins"].astype(float) / agg["count"].replace(0, np.nan)
    agg = agg.sort_values(group_cols + ["conversion_rate", "median_len", "count"], ascending=[True]*len(group_cols) + [False, True, False])
    topk = agg.groupby(group_cols, sort=False).head(k).reset_index(drop=True)
    return topk
