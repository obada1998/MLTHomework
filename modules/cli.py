import os
from typing import Dict, Any
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from config import CSV_PATH, CLEANED_CSV, LAST_TOUCH_DECAY_BY_ACTION, DEFAULT_BASE_WEIGHT, WEIGHTS_STORE_CSV, MAX_JOURNEY_STEPS, GLOBAL_LAST_TOUCH_DECAY, TOP_PATHS
from data_cleaning import load_and_clean, canonicalize_action
from paths import extract_account_paths, top_k_paths_by_group
from weights import DynamicWeightSystem, compute_action_base_weights
from ml import prepare_ml_data, train_decision_trees, extract_feature_importances 
from engine import JourneyEngine

def build_system(csv_path: str = CSV_PATH) -> Dict[str, Any]:
    print("Loading and cleaning data...")
    raw_df = load_and_clean(csv_path)

    print("Extracting account paths...")
    acc_paths = extract_account_paths(raw_df)

    print("Computing top-5 paths...")
    top5_country_solution = top_k_paths_by_group(acc_paths, ["Country", "solution"], k=5)
    top5_country_solution.to_csv(TOP_PATHS, index=False)

    print("Preparing ML datasets...")
    ml = prepare_ml_data(raw_df)
    dt_next, dt_outcome = train_decision_trees(ml["X_train"], ml["y_next_train"], ml["y_out_train"])

    print("Evaluating models...")
    try:
        ya_pred = dt_next.predict(ml["X_test"]) if dt_next is not None and ml["X_test"].shape[0] > 0 else []
        next_acc = accuracy_score(ml["y_next_test"], ya_pred) if len(ml["y_next_test"]) > 0 else None
    except Exception:
        next_acc = None
    try:
        yo_pred = dt_outcome.predict(ml["X_test"]) if dt_outcome is not None and ml["X_test"].shape[0] > 0 else []
        out_acc = accuracy_score(ml["y_out_test"], yo_pred) if len(ml["y_out_test"]) > 0 else None
    except Exception:
        out_acc = None

    print(f"Next-action acc: {next_acc}, Outcome acc: {out_acc}")

    print("Computing base weights...")
    base_weights = compute_action_base_weights(raw_df, min_support=3, fallback=DEFAULT_BASE_WEIGHT)

    print("Initializing DynamicWeightSystem and loading persisted state...")
    ws = DynamicWeightSystem(base_weights=base_weights, per_action_decay=LAST_TOUCH_DECAY_BY_ACTION, global_decay=GLOBAL_LAST_TOUCH_DECAY)

    if not os.path.exists(WEIGHTS_STORE_CSV):
        grp = raw_df.sort_values(["account_id", "activity_date"]).groupby("account_id")["types"].apply(list)
        seqs = grp.to_dict()
        ws.process_many_histories(seqs, persist=True)
    else:
        ws._load_weights_store()

    print("Initializing JourneyEngine...")
    engine = JourneyEngine(
        dt_next=dt_next,
        dt_outcome=dt_outcome,
        ohe=ml["ohe"],
        cat_cols=ml["cat_cols"],
        num_cols=ml["num_cols"],
        base_weights=base_weights,
        ws=ws,
        feature_names=ml["feature_names"]
    )
    engine.engine_train_df = ml.get("train_df")
    try:
        engine.ohe._training_df = ml.get("train_df")
    except Exception:
        pass
      
    engine.selected_feature_indices = []
    engine.selected_feature_names = []
    try:
        if dt_next is not None and ml.get("feature_names"):
            feat_imps = extract_feature_importances(dt_next, ml["feature_names"])
            TOP_FEATURES_N = 5
            top_feats = [f for f, _ in feat_imps[:TOP_FEATURES_N]]
            feat_indices = [ml["feature_names"].index(f) for f in top_feats if f in ml["feature_names"]]
            if engine.selected_feature_indices:
                max_dim = len(ml["feature_names"]) - 1
                engine.selected_feature_indices = [i for i in engine.selected_feature_indices if 0 <= i <= max_dim]
                if not engine.selected_feature_indices:
                    print("Warning: selected_feature_indices all out-of-range; ignoring mask.")

            engine.selected_feature_indices = feat_indices
            engine.selected_feature_names = top_feats
        else:
            engine.selected_feature_indices = []
            engine.selected_feature_names = []
    except Exception as ex:
        engine.selected_feature_indices = []
        engine.selected_feature_names = []


    print("Precomputing global top lists...")
    account_sample_df = raw_df.sort_values(["account_id", "activity_date"]).groupby("account_id").last().reset_index()
    for c in ["Country", "solution", "is_lead"]:
        if c not in account_sample_df.columns:
            account_sample_df[c] = 1 if c == "is_lead" else "UNKNOWN"


    top_by_country = engine.compute_global_top_actions(account_sample_df, ["Country"], n=4)
    top_by_solution = engine.compute_global_top_actions(account_sample_df, ["solution"], n=4)
    top_by_country_solution = engine.compute_global_top_actions(account_sample_df, ["Country", "solution"], n=4)

    top_by_country_next = engine.compute_global_top_actions(account_sample_df, ["Country"], n=4, metric="next")
    top_by_country_outcome = engine.compute_global_top_actions(account_sample_df, ["Country"], n=4, metric="outcome")


    feature_importances_outcome = extract_feature_importances(dt_outcome, ml["feature_names"])
    feature_importances_next = extract_feature_importances(dt_next, ml["feature_names"])
    if dt_outcome is not None:
        engine.feature_importances_outcome = feature_importances_outcome
        engine.selected_feature_names = [f for f,_ in feature_importances_outcome[:4]]
    else:
        engine.feature_importances_outcome = []
        engine.selected_feature_names = []

    return {
        "raw_df": raw_df,
        "acc_paths": acc_paths,
        "top5_country_solution": top5_country_solution,
        "ml": ml,
        "dt_next": dt_next,
        "dt_outcome": dt_outcome,
        "base_weights": base_weights,
        "ws": ws,
        "engine": engine,
        "top_by_country": top_by_country,
        "top_by_solution": top_by_solution,
        "top_by_country_solution": top_by_country_solution,
        "feature_importances_outcome": feature_importances_outcome,
        "feature_importances_next": feature_importances_next,
        "model_eval": {"next_acc": next_acc, "out_acc": out_acc},
        "account_sample_df": account_sample_df,
        "top_by_country_next":top_by_country_next,
        "top_by_country_outcome":top_by_country_outcome
    }

def _ensure_activity_date_datetime(df: pd.DataFrame, col: str = "activity_date") -> pd.DataFrame:
    if col not in df.columns:
        df[col] = pd.NaT
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def _build_new_row_df(raw_df: pd.DataFrame, sample: Dict[str,Any]) -> pd.DataFrame:
    cols = list(raw_df.columns)
    row = {c: sample.get(c, np.nan) for c in cols}
    return pd.DataFrame([row])

def _recompute_account_fields(raw_df: pd.DataFrame, account_id: str) -> pd.DataFrame:
    acc_df = raw_df.loc[raw_df["account_id"] == account_id].copy()
    if acc_df.empty:
        return acc_df
    acc_df = _ensure_activity_date_datetime(acc_df).sort_values("activity_date").reset_index(drop=True)
    acc_df["touch_index"] = acc_df.index + 1
    acc_df["num_past_touches"] = acc_df["touch_index"] - 1
    acc_df["days_since_prev_touch"] = acc_df["activity_date"].diff().dt.total_seconds().div(86400).fillna(0).clip(lower=0)
    acc_df["is_last_touch_in_data"] = 0
    acc_df.at[acc_df.index[-1], "is_last_touch_in_data"] = 1
    return acc_df

def pretty_print_top_list(lst, title: str = "Top actions"):
    print(f"\n======== {title} ========")
    if not lst:
        print(" (no entries)")
        return
    for i, row in enumerate(lst, 1):
        if isinstance(row, dict):
            action = row.get("action", "<unknown>")
            avg_p_next = row.get("avg_p_next", None)
            avg_p_win = row.get("avg_p_win", None)
            base_w = row.get("base_weight", None)
            score = row.get("score", None)
            print(f"{i}. {action} — score={score}, p_next={avg_p_next}, p_win={avg_p_win}, base_weight={base_w}")
        else:
            try:
                print(f"{i}. {row[0]} — details: {row[1:5]}")
            except Exception:
                print(f"{i}. {row}")


def run_cli(system: Dict[str, Any]):
    raw_df: pd.DataFrame = system["raw_df"]
    top5 = system["top5_country_solution"]
    engine: JourneyEngine = system["engine"]
    ws: DynamicWeightSystem = system["ws"]

    raw_df = _ensure_activity_date_datetime(raw_df)

    print("="*100)
    print("Type 'exit' to quit.\n")
    print("="*100)

    while True:
        account_id = input("\nAccount ID: ").strip()
        if account_id.lower() in {"exit","quit"}:
            break
        country = input("Country [leave blank = UNKNOWN]: ").strip() or "UNKNOWN"
        solution = input("Solution: [leave blank = UNKNOWN]: ").strip() or "UNKNOWN"
        is_lead_input = input("Is Lead? (1 = Yes, 0 = No) [default=1]: ").strip()
        is_lead = int(is_lead_input) if is_lead_input in {"0","1"} else 1

        raw_df = _ensure_activity_date_datetime(raw_df)
        acct_rows = raw_df.loc[raw_df["account_id"] == account_id].sort_values("activity_date", ascending=False)
        last_in_raw = None
        if not acct_rows.empty:
            valid = acct_rows.dropna(subset=["activity_date"])
            if not valid.empty:
                last_in_raw = valid.iloc[0]["types"]
            else:
                last_in_raw = acct_rows.iloc[0]["types"]
        last_in_ws = None
        if account_id in ws.state and ws.state[account_id].get("actions"):
            last_in_ws = ws.state[account_id]["actions"][-1]
        if last_in_raw and last_in_ws and str(last_in_raw) != str(last_in_ws):
            hist = raw_df.loc[raw_df["account_id"] == account_id].sort_values("activity_date", na_position="first")["types"].tolist()
            ws.initialize_account(account_id, action_history=hist, reset=True, persist=True)
            print(f"(Re-synced ws from raw_df - last='{hist[-1] if hist else None}')")
        elif last_in_ws:
            print(f"(Using existing account state - last='{last_in_ws}')")
        else:
            if last_in_raw:
                hist = raw_df.loc[raw_df["account_id"] == account_id].sort_values("activity_date", na_position="first")["types"].tolist()
                ws.initialize_account(account_id, action_history=hist, reset=True, persist=True)
                print(f"(Preloaded historical actions for account {account_id}, last='{hist[-1]}')")
            else:
                print(f"(No historical actions found for account {account_id}; first add_action will seed state.)")

        country_q = country.strip().upper()
        solution_q = solution.strip().upper()
        print("="*75)
        print("Top 5 historical paths for this Country + Solution:")
        print("="*75)
        try:
            subset = top5.query("Country == @country_q and solution == @solution_q")
        except Exception:
            subset = pd.DataFrame()
        if subset.empty:
            print(" No historical paths found")
        else:
            cols_needed = ["count", "wins", "conversion_rate"]
            if "path" in subset.columns:
                display_cols = ["path"] + cols_needed
            elif "path_str" in subset.columns:
                display_cols = ["path_str"] + cols_needed
            else:
                display_cols = cols_needed
            print(subset[display_cols].to_string(index=False))


        top_country = system["top_by_country"].get((country_q,), [])
        pretty_print_top_list(top_country, f"Top-4 actions for Country = {country_q} (global ranking)")
        top_solution = system["top_by_solution"].get((solution_q,), [])
        pretty_print_top_list(top_solution, f"Top-4 actions for Solution = {solution_q} (global ranking)")
        top_cs = system["top_by_country_solution"].get((country_q, solution_q), [])
        pretty_print_top_list(top_cs, f"Top-4 actions for Country + Solution= ({country_q},{solution_q}) (global ranking)")

        print("\nTop 4 actions for this account (weights applied):")
        try:
            acc_top4 = engine.top_n_actions_for_account(account_id, country_q, solution_q, is_lead=is_lead, n=4)
            for i,(a,p,w,s) in enumerate(acc_top4,1):
                print(f" {i}. {a} — predicted_prob={p:.4f}, weight={w:.4f}, score={s:.4f}")
        except Exception as e:
            print("Error computing per-account top-4:", e)

        new_action = input("\nAdd an action to this account to update weights (press Enter to skip): ").strip()
        if new_action:
            new_action_canon = canonicalize_action(new_action)
            now_ts = pd.Timestamp.now()
            sample_row = {
                "account_id": account_id, "SourceSystem": "MANUAL", "activity_date": now_ts,
                "who_id": "", "opportunity_id": "", "opportunity_stage": "", "is_lead": is_lead,
                "types": new_action_canon, "Country": country_q, "solution": solution_q,
                "touch_index": None, "days_since_prev_touch": None, "num_past_touches": None, "is_last_touch_in_data": 1
            }
            cols = raw_df.columns.tolist()
            new_df_row = _build_new_row_df(raw_df, sample_row).reindex(columns=cols)
            try:
                new_df_row["activity_date"] = pd.to_datetime(new_df_row["activity_date"], errors="coerce")
            except Exception:
                new_df_row["activity_date"] = pd.to_datetime(now_ts)

            try:
                if "activity_date" in raw_df.columns and not raw_df["activity_date"].dropna().empty:
                    max_existing = raw_df["activity_date"].max()
                    if pd.isna(new_df_row.loc[0, "activity_date"]) or pd.to_datetime(new_df_row.loc[0, "activity_date"]) <= pd.to_datetime(max_existing):
                        new_df_row.loc[0, "activity_date"] = pd.to_datetime(max_existing) + pd.Timedelta(seconds=1)
            except Exception:
                new_df_row.loc[0, "activity_date"] = pd.to_datetime(now_ts)

            for c in cols:
                if c in raw_df.columns:
                    dtype = raw_df[c].dtype
                    if np.issubdtype(dtype, np.number):
                        try:
                            new_df_row[c] = pd.to_numeric(new_df_row[c], errors="coerce")
                        except Exception:
                            pass
                    elif np.issubdtype(dtype, np.datetime64):
                        try:
                            new_df_row[c] = pd.to_datetime(new_df_row[c], errors="coerce")
                        except Exception:
                            pass
                    else:
                        new_df_row[c] = new_df_row[c].astype(object)

            raw_df.loc[raw_df["account_id"] == account_id, "is_last_touch_in_data"] = 0

            raw_df = pd.concat([raw_df, new_df_row], ignore_index=True, sort=False)

            raw_df = _ensure_activity_date_datetime(raw_df)
            acc_df = _recompute_account_fields(raw_df, account_id)
            other_df = raw_df.loc[raw_df["account_id"] != account_id].reset_index(drop=True)
            raw_df = pd.concat([other_df, acc_df], ignore_index=True, sort=False)
            raw_df = _ensure_activity_date_datetime(raw_df)
            system["raw_df"] = raw_df
            try:
                raw_df.to_csv(CLEANED_CSV, index=False)
            except Exception:
                pass

            hist = raw_df.loc[raw_df["account_id"] == account_id].sort_values("activity_date")["types"].tolist()
            ws.initialize_account(account_id, action_history=hist, reset=True, persist=True)

            system["account_sample_df"] = raw_df.sort_values(["account_id","activity_date"]).groupby("account_id").last().reset_index()
            system["acc_paths"] = extract_account_paths(raw_df)
            system["top5_country_solution"] = top_k_paths_by_group(system["acc_paths"], ["Country","solution"], k=5)
            system["top_by_country"] = engine.compute_global_top_actions(system["account_sample_df"], ["Country"], n=4)
            system["top_by_solution"] = engine.compute_global_top_actions(system["account_sample_df"], ["solution"], n=4)
            system["top_by_country_solution"] = engine.compute_global_top_actions(system["account_sample_df"], ["Country","solution"], n=4)
            top5 = system["top5_country_solution"]

            print("Action added and weights updated. New account state:")
            print(ws.dump_account_state(account_id))
            print("New last action should be this one:", ws.state.get(account_id, {}).get("actions", [])[-1] if ws.state.get(account_id, {}).get("actions") else "(none)")

        print("\nSuggested journey to WIN:")
        win_journey = engine.build_journey(account_id, country_q, solution_q, target="win", steps=MAX_JOURNEY_STEPS, is_lead=is_lead)
        if not win_journey:
            print(" (no suggested steps)")
        else:
            for st in win_journey:
                print(st)

        print("\nSuggested journey to LOSS:")
        loss_journey = engine.build_journey(account_id, country_q, solution_q, target="loss", steps=MAX_JOURNEY_STEPS, is_lead=is_lead)
        if not loss_journey:
            print(" (no suggested steps)")
        else:
            for st in loss_journey:
                print(st)

        print("\nFeature importances (outcome):")
        for feat, imp in system.get("feature_importances_outcome", [])[:10]:
            print(f" {feat}: {imp:.6f}")
        print("\nFeature importances (next):")
        for feat, imp in system.get("feature_importances_next", [])[:10]:
            print(f" {feat}: {imp:.6f}")
        
        cont = input("\nContinue? (Y/n) [Default=Y]: ").strip().lower()
        if cont == "n":
            break

    try:
        ws.save_state()
    except Exception:
        pass

if __name__ == "__main__":
    try:
        system = build_system(CSV_PATH)
        run_cli(system)
    except Exception as e:
        print("Fatal error building system:", e)
        raise