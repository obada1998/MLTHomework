import os
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any
from collections import defaultdict
from config import DEFAULT_BASE_WEIGHT, WS_STATE_FILE, GLOBAL_LAST_TOUCH_DECAY, MIN_ADJUSTED_WEIGHT, WEIGHTS_STORE_CSV, WEIGHTS_HISTORY_CSV

class DynamicWeightSystem:
    def __init__(self, base_weights: Dict[str, float], per_action_decay: Dict[str,float]=None, global_decay: float = GLOBAL_LAST_TOUCH_DECAY,
                 min_adjusted: float = MIN_ADJUSTED_WEIGHT,
                 weights_store: str = WEIGHTS_STORE_CSV, history_store: str = WEIGHTS_HISTORY_CSV, json_state: str = WS_STATE_FILE,
                 use_per_action_decay: bool = True):

        self.base_weights = defaultdict(lambda: float(DEFAULT_BASE_WEIGHT))
        if base_weights:
            for k,v in base_weights.items():
                try:
                    self.base_weights[k] = float(v)
                except Exception:
                    self.base_weights[k] = float(DEFAULT_BASE_WEIGHT)
        self.per_action_decay = {k: float(v) for k,v in (per_action_decay or {}).items()}
        self.global_decay = float(global_decay)
        self.min_adjusted = float(min_adjusted)
        self.state = defaultdict(lambda: {"actions": [], "adjusted": defaultdict(float)})
        self.weights_store = weights_store
        self.history_store = history_store
        self.json_state = json_state
        self.use_per_action_decay = bool(use_per_action_decay)
        self._load_weights_store()
        self._load_json_state()

    def _append_history_row(self, account_id: str, action: str, adjusted_weight: float, last_touch_weight: Any = ""):
        row = {
            "account_id": account_id,
            "action": action,
            "adjusted_weight": float(adjusted_weight),
            "base_weight": float(self.base_weights.get(action, DEFAULT_BASE_WEIGHT)),
            "last_touch_weight": last_touch_weight if last_touch_weight != "" else "",
            "ts": datetime.now(timezone.utc).isoformat()
        }
        header = not os.path.exists(self.history_store)
        try:
            with open(self.history_store, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            pass

    def _persist_weights_store(self):
        rows = []
        for aid, st in self.state.items():
            for act,w in st["adjusted"].items():
                rows.append({"account_id": aid, "action": act, "base_weight": float(self.base_weights.get(act, DEFAULT_BASE_WEIGHT)), "adjusted_weight": float(w), "last_updated": datetime.now(timezone.utc).isoformat()})
        try:
            if rows:
                pd.DataFrame(rows).to_csv(self.weights_store, index=False)
            else:
                if os.path.exists(self.weights_store):
                    os.remove(self.weights_store)
        except Exception:
            pass

    def _load_weights_store(self):
        if not os.path.exists(self.weights_store):
            return
        try:
            df = pd.read_csv(self.weights_store)
            for _, r in df.iterrows():
                aid = r.get("account_id")
                act = r.get("action")
                try:
                    w = float(r.get("adjusted_weight", np.nan))
                except Exception:
                    w = float(self.base_weights.get(act, DEFAULT_BASE_WEIGHT))
                st = self.state[aid]
                st["adjusted"][act] = w
        except Exception:
            pass

    def save_state(self, filepath: str = None):
        fp = filepath or self.json_state
        serial = {}
        for acc, st in self.state.items():
            serial[acc] = {"actions": list(st["actions"]), "adjusted": {k: float(v) for k,v in st["adjusted"].items()}}
        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(serial, f, indent=2)
        except Exception:
            pass

    def _load_json_state(self, filepath: str = None):
        fp = filepath or self.json_state
        if not os.path.exists(fp):
            return
        try:
            with open(fp, "r", encoding="utf-8") as f:
                serial = json.load(f)
            for acc, st in serial.items():
                self.state[acc]["actions"] = list(st.get("actions", []))
                adjusted = st.get("adjusted", {})
                self.state[acc]["adjusted"] = defaultdict(float, {k: float(v) for k,v in adjusted.items()})
        except Exception:
            pass

    def _get_decay_for_action(self, action: str) -> float:
        try:
            if not bool(self.use_per_action_decay):
                d = float(self.global_decay)
            else:
                if not action:
                    d = float(self.global_decay)
                else:
                    d = float(self.per_action_decay.get(action, self.global_decay))
        except Exception:
            d = float(self.global_decay)
        return min(max(d, 0.0), 1.0)

    def initialize_account(self, account_id: str, action_history: List[str] = None, reset: bool = True, persist: bool = True):
        if not account_id:
            return
        st = self.state[account_id]
        if reset:
            st["actions"] = []
            st["adjusted"] = defaultdict(float)
        if not action_history:
            return
        batch_rows = []
        for idx, a in enumerate(action_history):
            a = str(a).strip()
            if idx == 0:
                st["adjusted"][a] = float(self.base_weights.get(a, DEFAULT_BASE_WEIGHT))
                if persist:
                    batch_rows.append({
                        "account_id": account_id,
                        "action": a,
                        "adjusted_weight": float(st["adjusted"][a]),
                        "base_weight": float(self.base_weights.get(a, DEFAULT_BASE_WEIGHT)),
                        "last_touch_weight": "",
                        "ts": datetime.now(timezone.utc).isoformat()
                    })
                st["actions"].append(a)
                continue

            last = st["actions"][-1]
            base_last = float(self.base_weights.get(last, DEFAULT_BASE_WEIGHT)) or float(DEFAULT_BASE_WEIGHT)

            last_touch_weight = self._get_decay_for_action(last)

            new_w = float(max(base_last * (1.0 - last_touch_weight), self.min_adjusted))
            st["adjusted"][last] = new_w

            if persist:
                batch_rows.append({
                    "account_id": account_id,
                    "action": last,
                    "adjusted_weight": float(new_w),
                    "base_weight": float(base_last),
                    "last_touch_weight": float(last_touch_weight),
                    "ts": datetime.now(timezone.utc).isoformat()
                })

            if a not in st["adjusted"] or float(st["adjusted"].get(a, 0.0)) == 0.0:
                st["adjusted"][a] = float(self.base_weights.get(a, DEFAULT_BASE_WEIGHT))
                if persist:
                    batch_rows.append({
                        "account_id": account_id,
                        "action": a,
                        "adjusted_weight": float(st["adjusted"][a]),
                        "base_weight": float(self.base_weights.get(a, DEFAULT_BASE_WEIGHT)),
                        "last_touch_weight": "",
                        "ts": datetime.now(timezone.utc).isoformat()
                    })

            if not st["actions"] or st["actions"][-1] != a:
                st["actions"].append(a)

        if persist and batch_rows:
            try:
                header = not os.path.exists(self.history_store)
                with open(self.history_store, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(batch_rows[0].keys()))
                    if header:
                        writer.writeheader()
                    writer.writerows(batch_rows)
            except Exception as e:
                try:
                    print("Failed writing batch_rows in initialize_account:", e)
                except Exception:
                    pass

        if persist:
            try:
                self._persist_weights_store()
                self.save_state()
            except Exception as e:
                try:
                    print("Failed persisting weights/state after initialize_account:", e)
                except Exception:
                    pass

    def add_action(self, account_id: str, action: str, persist: bool = True):
        if not account_id or not action:
            return
        st = self.state[account_id]
        action = str(action).strip()

        if st["adjusted"].get(action, 0.0) == 0.0:
            st["adjusted"][action] = float(self.base_weights.get(action, DEFAULT_BASE_WEIGHT))
            if persist:
                try:
                    self._append_history_row(account_id, action, st["adjusted"][action], last_touch_weight="")
                except Exception:
                    pass

        if st["actions"]:
            last = st["actions"][-1]
            base_last = float(self.base_weights.get(last, DEFAULT_BASE_WEIGHT)) or float(DEFAULT_BASE_WEIGHT)

            last_touch_weight = self._get_decay_for_action(last)

            new_w = base_last * (1.0 - last_touch_weight)
            st["adjusted"][last] = new_w

            if persist:
                try:
                    self._append_history_row(account_id, last, new_w, last_touch_weight=last_touch_weight)
                except Exception:
                    pass

        if not st["actions"] or st["actions"][-1] != action:
            st["actions"].append(action)


        if persist:
            try:
                self._append_history_row(account_id, action, st["adjusted"].get(action, self.base_weights.get(action, DEFAULT_BASE_WEIGHT)), last_touch_weight="")
                self._persist_weights_store()
                self.save_state()
            except Exception:
                pass

    def get_weight(self, account_id: str, action: str) -> float:
        st = self.state.get(account_id, {"actions": [], "adjusted": {}})
        w = st.get("adjusted", {}).get(action)
        if w is not None and float(w) > 0:
            return float(w)
        return float(self.base_weights.get(action, float(DEFAULT_BASE_WEIGHT)))

    def reset_account(self, account_id: str):
        self.state.pop(account_id, None)
        self._persist_weights_store()
        self.save_state()

    def dump_account_state(self, account_id: str) -> Dict[str, Any]:
        st = self.state.get(account_id, {"actions": [], "adjusted": {}})
        return {"actions": list(st["actions"]), "adjusted": {k: float(v) for k,v in dict(st["adjusted"]).items()}}

    def process_many_histories(self, sequences_dict: Dict[str, List[str]], persist: bool = True):
        self.state = defaultdict(lambda: {"actions": [], "adjusted": defaultdict(float)})
        history_rows = []
        for aid, seq in sequences_dict.items():
            if not seq:
                continue
            st = self.state[aid]
            for idx, a in enumerate(seq):
                a = str(a).strip()
                if idx == 0:
                    st["adjusted"][a] = float(self.base_weights.get(a, DEFAULT_BASE_WEIGHT))
                    st["actions"].append(a)
                    if persist:
                        history_rows.append({
                            "account_id": aid,
                            "action": a,
                            "adjusted_weight": float(st["adjusted"][a]),
                            "base_weight": float(self.base_weights.get(a, DEFAULT_BASE_WEIGHT)),
                            "last_touch_weight": "",
                            "ts": datetime.now(timezone.utc).isoformat()
                        })
                    continue

                last = st["actions"][-1]
                base_last = float(self.base_weights.get(last, DEFAULT_BASE_WEIGHT)) or float(DEFAULT_BASE_WEIGHT)

                last_touch_weight = self._get_decay_for_action(last)

                new_w = float(max(base_last * (1.0 - last_touch_weight), self.min_adjusted))
                st["adjusted"][last] = new_w

                if persist:
                    history_rows.append({
                        "account_id": aid,
                        "action": last,
                        "adjusted_weight": float(new_w),
                        "base_weight": float(base_last),
                        "last_touch_weight": float(last_touch_weight),
                        "ts": datetime.now(timezone.utc).isoformat()
                    })

                if a not in st["adjusted"] or float(st["adjusted"].get(a, 0.0)) == 0.0:
                    st["adjusted"][a] = float(self.base_weights.get(a, DEFAULT_BASE_WEIGHT))
                    if persist:
                        history_rows.append({
                            "account_id": aid,
                            "action": a,
                            "adjusted_weight": float(st["adjusted"][a]),
                            "base_weight": float(self.base_weights.get(a, DEFAULT_BASE_WEIGHT)),
                            "last_touch_weight": "",
                            "ts": datetime.now(timezone.utc).isoformat()
                        })

                if not st["actions"] or st["actions"][-1] != a:
                    st["actions"].append(a)

        if persist and history_rows:
            try:
                header = not os.path.exists(self.history_store)
                with open(self.history_store, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
                    if header:
                        writer.writeheader()
                    writer.writerows(history_rows)
            except Exception as e:
                try:
                    print("Failed writing history_rows in process_many_histories:", e)
                except Exception:
                    pass

        if persist:
            try:
                self._persist_weights_store()
                self.save_state()
            except Exception as e:
                try:
                    print("Failed persisting weights/state after process_many_histories:", e)
                except Exception:
                    pass
        print("Loaded state accounts:", len(self.state))
        distinct = set()
        for st in self.state.values():
            distinct.update(st["adjusted"].keys())
        print("Distinct actions with adjusted weights:", len(distinct))

def compute_action_base_weights(df: pd.DataFrame, min_support: int = 10, fallback=DEFAULT_BASE_WEIGHT) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    df = df.copy()
    df["opportunity_id"] = df.get("opportunity_id", pd.Series([pd.NA]*len(df))).replace({np.nan: pd.NA, "": pd.NA})
    df["journey_id"] = df["opportunity_id"].where(df["opportunity_id"].notna() & df["opportunity_id"].astype(str).str.strip().ne(""), df["account_id"])

    ever_won = (df.groupby("journey_id")["opportunity_stage"]
                  .apply(lambda s: (s.astype(str).str.lower() == "won").any())
                  .rename("ever_won")
                  .reset_index())
    journey_types = df[["journey_id", "types"]].dropna().drop_duplicates(subset=["journey_id", "types"])
    journey_types = journey_types.merge(ever_won, on="journey_id", how="left").fillna({"ever_won": False})

    agg = (journey_types.groupby("types")
                     .agg(count_journeys=("journey_id", "nunique"),
                          wins=("ever_won", "sum"))
                     .reset_index())
    total_counts = agg["count_journeys"].sum()
    global_win = float(agg["wins"].sum() / total_counts) if total_counts > 0 else float(fallback or 0.2)

    def compute_row_weight(row):
        cnt = int(row["count_journeys"]) if not pd.isna(row["count_journeys"]) else 0
        wins = int(row["wins"]) if not pd.isna(row["wins"]) else 0
        if cnt >= int(min_support):
            return float(wins) / float(cnt) if wins > 0 else float(max(global_win, float(fallback)))
        else:
            return float(max(global_win, float(fallback)))
    agg["base_weight"] = agg.apply(compute_row_weight, axis=1)
    return dict(zip(agg["types"], agg["base_weight"]))
