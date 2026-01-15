from typing import List
import pandas as pd
import numpy as np
from collections import defaultdict
from config import DEFAULT_BASE_WEIGHT, LOSS_THRESHOLD, WIN_THRESHOLD, MAX_JOURNEY_STEPS
from weights import  DynamicWeightSystem


class JourneyEngine:
    def __init__(self, dt_next, dt_outcome, ohe, cat_cols, num_cols, base_weights, ws: DynamicWeightSystem, feature_names):
        self.dt_next = dt_next
        self.dt_outcome = dt_outcome
        self.ohe = ohe
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        try:
            actions_set = set(dt_next.classes_.tolist()) if dt_next is not None else set()
        except Exception:
            actions_set = set()
        self.actions = sorted(list(actions_set.union(set(base_weights.keys())).union({"NONE"})))
        self.ws = ws
        self.base_weights = defaultdict(lambda: float(DEFAULT_BASE_WEIGHT), {k: float(v) for k,v in (base_weights or {}).items()})
        self.feature_names = feature_names
        self._p_cache = {}
    
    def compute_global_top_actions(self, df_accounts: pd.DataFrame, group_cols: List[str], n: int = 4, metric: str = "outcome", alpha: float = 0.7):
        results = {}
        grouped = df_accounts.groupby(group_cols)
        for group_values, sub in grouped:
            key = group_values if isinstance(group_values, tuple) else (group_values,)
            action_scores = []
            if sub.empty:
                results[key] = []
                continue
            acc_sample = list(sub["account_id"].unique())
            for action in [a for a in self.actions if a not in {"NONE", "UNKNOWN"}]:
                p_list = []
                w_list = []
                for acc in acc_sample:
                    r = sub[sub["account_id"] == acc]
                    if r.empty:
                        continue
                    row = r.iloc[0]
                    country = row.get("Country", "UNKNOWN")
                    solution = row.get("solution","UNKNOWN")
                    is_lead = int(row.get("is_lead",1)) if "is_lead" in row else 1
                    st = self.ws.state.get(acc, {})
                    if st and st.get("actions"):
                        last_act = st["actions"][-1]
                        num_past = len(st.get("actions", []))
                    else:
                        last_act = row.get("types", "NONE") if "types" in row else "NONE"
                        num_past = int(row.get("num_past_touches", 0)) if "num_past_touches" in row else 0

                    p_next = self.predict_next_action_probs(country, solution, last_action=last_act, num_past_touches=num_past, days_since_prev_touch=0.0, is_lead=is_lead).get(action, 0.0)

                    p_win = self.predict_outcome_prob_given_action(country, solution, action, num_past_touches=num_past, days_since_prev_touch=0.0, is_lead=is_lead, last_action=last_act)

                    w = self.ws.get_weight(acc, action)
                    p_list.append((p_next, p_win))
                    w_list.append(w)
                if not p_list:
                    avg_score = 0.0
                    avg_p_next = 0.0
                    avg_p_win = 0.0
                else:
                    fallback_win = None
                    try:
                        fallback_win = float(self._empirical_win_rates.get("__global__", (0.2,0))[0])
                    except Exception:
                        fallback_win = 0.2

                    p_nexts = np.array([0.0 if (pp[0] is None or (isinstance(pp[0], float) and not np.isfinite(pp[0]))) else float(pp[0]) for pp in p_list], dtype=float)
                    p_wins = np.array([fallback_win if (pp[1] is None or (isinstance(pp[1], float) and not np.isfinite(pp[1]))) else float(pp[1]) for pp in p_list], dtype=float)

                    ws_arr = np.array(w_list, dtype=float)
                    if ws_arr.sum() > 0:
                        avg_p_next = float(np.dot(p_nexts, ws_arr) / ws_arr.sum())
                        avg_p_win = float(np.dot(p_wins, ws_arr) / ws_arr.sum())
                    else:
                        avg_p_next = float(p_nexts.mean() if p_nexts.size>0 else 0.0)
                        avg_p_win = float(p_wins.mean() if p_wins.size>0 else fallback_win)

                    if metric == "next":
                        avg_score = avg_p_next * float(self.base_weights.get(action, DEFAULT_BASE_WEIGHT))
                    elif metric == "hybrid":
                        mix = (alpha * avg_p_next) + ((1.0 - alpha) * avg_p_win)
                        avg_score = mix * float(self.base_weights.get(action, DEFAULT_BASE_WEIGHT))
                    else:  
                        avg_score = avg_p_win * float(self.base_weights.get(action, DEFAULT_BASE_WEIGHT))

                action_scores.append((action, round(avg_p_next,6), round(avg_p_win,6), round(float(self.base_weights.get(action, DEFAULT_BASE_WEIGHT)),6), round(avg_score,6)))
            action_scores.sort(key=lambda x: x[4], reverse=True)
            results[key] = action_scores[:n]
        return results

    def _build_context_matrix(self, country, solution, last_action, candidate_action=None, num_past_touches=0, days_since_prev_touch=0.0, is_lead=1):
        import re
        import difflib

        def safe_str(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
            return str(x).strip()

        c0 = safe_str(country).upper()
        s0 = safe_str(solution).upper()
        la0 = safe_str(last_action)
        ca0 = safe_str(candidate_action) if candidate_action is not None else safe_str(last_action)

        cat_vals = {"Country": c0, "solution": s0, "last_action": la0, "candidate_action": ca0}

        categories = getattr(self.ohe, "categories_", None)
        if categories is None:
            df = pd.DataFrame([{
                "Country": c0, "solution": s0, "last_action": la0, "candidate_action": ca0,
                "num_past_touches": float(num_past_touches),
                "days_since_prev_touch": float(days_since_prev_touch),
                "is_lead": int(is_lead)
            }])
            try:
                cat = self.ohe.transform(df[self.cat_cols])
            except Exception:
                cat = np.zeros((1, 0))
            num = df[self.num_cols].to_numpy().astype(float)
            X = np.hstack([cat, num]) if cat.size else num
            return X

        def coerce_value_to_category(col_idx, raw_val):
            if categories is None:
                return None
            known = [str(v) for v in categories[col_idx]]
            rv = safe_str(raw_val)
            candidates = [
                rv,
                rv.strip(),
                rv.title(),
                rv.upper(),
                rv.lower(),
                re.sub(r"[_\-]+", " ", rv).strip(),
                re.sub(r"[^\w\s]", "", rv).strip()
            ]
            for cand in candidates:
                if cand in known:
                    return cand
            try:
                if known:
                    match = difflib.get_close_matches(rv, known, n=1, cutoff=0.66)
                    if match:
                        return match[0]
            except Exception:
                pass
            try:
                from data_cleaning import canonicalize_action
                cand = canonicalize_action(rv)
                if cand in known:
                    return cand
            except Exception:
                pass
            return None

        coerced = {}
        unknown_warns = []
        for idx, col in enumerate(self.cat_cols):
            raw = cat_vals.get(col, "")
            coer = coerce_value_to_category(idx, raw)
            if coer is None:
                unknown_warns.append((col, raw, idx))
                coerced[col] = "__UNKNOWN__"
            else:
                coerced[col] = coer

        df = pd.DataFrame([{
            "Country": coerced.get("Country", "__UNKNOWN__"),
            "solution": coerced.get("solution", "__UNKNOWN__"),
            "last_action": coerced.get("last_action", "__UNKNOWN__"),
            "candidate_action": coerced.get("candidate_action", "__UNKNOWN__"),
            "num_past_touches": float(num_past_touches),
            "days_since_prev_touch": float(days_since_prev_touch),
            "is_lead": int(is_lead)
        }])

        try:
            cat = self.ohe.transform(df[self.cat_cols])
        except Exception as ex:
            cat = np.zeros((1, sum(len(c) for c in categories)))

        try:
            expected_cat_cols = len(self.ohe.get_feature_names_out(self.cat_cols))
        except Exception:
            expected_cat_cols = cat.shape[1] if cat is not None else 0

        if cat.shape[1] < expected_cat_cols:
            pad = np.zeros((cat.shape[0], expected_cat_cols - cat.shape[1]))
            cat = np.hstack([cat, pad])
        elif cat.shape[1] > expected_cat_cols:
            cat = cat[:, :expected_cat_cols]

        num = df[self.num_cols].to_numpy().astype(float)
        X = np.hstack([cat, num])


        if float(np.sum(np.abs(X))) == 0.0:
            try:
                cats_lens = [len(c) for c in categories]
            except Exception:
                cats_lens = None
        return X


    def predict_next_action_probs(self, country, solution, last_action, num_past_touches=0, days_since_prev_touch=0.0, is_lead=1):
        if self.dt_next is None:
            return {a: 0.0 for a in self.actions}
        X = self._build_context_matrix(country, solution, last_action, candidate_action=last_action, num_past_touches=num_past_touches, days_since_prev_touch=days_since_prev_touch, is_lead=is_lead)

        probs = self.dt_next.predict_proba(X)[0]
        classes = list(self.dt_next.classes_)
        prob_map = {str(c): float(p) for c,p in zip(classes, probs)}
        return {a: float(prob_map.get(str(a), 0.0)) for a in self.actions}

    def predict_outcome_prob_given_action(self, country, solution, action, num_past_touches=0, days_since_prev_touch=0.0, is_lead=1, last_action=None):
        key = (str(country).upper() if country is not None else "UNKNOWN",
               str(solution).upper() if solution is not None else "UNKNOWN",
               str(action).strip(),
               int(num_past_touches),
               int(is_lead),
               str(last_action).strip() if last_action is not None else "NONE")
        if key in self._p_cache:
            return self._p_cache[key]
        if getattr(self, "_empirical_built", False):
            pass
        else:
            try:
                train_df = None
                if hasattr(self, "dt_outcome_train_df") and isinstance(getattr(self, "dt_outcome_train_df"), pd.DataFrame):
                    train_df = getattr(self, "dt_outcome_train_df")
                elif hasattr(self, "engine_train_df") and isinstance(getattr(self, "engine_train_df"), pd.DataFrame):
                    train_df = getattr(self, "engine_train_df")
                else:
                    tdf = getattr(self.ohe, "_training_df", None)
                    if isinstance(tdf, pd.DataFrame):
                        train_df = tdf


                if train_df is None or train_df.empty:
                    self._empirical_win_rates = {}
                    self._empirical_country_action = {}
                    self._empirical_solution_action = {}
                    self._empirical_global = None
                else:
                    self._empirical_win_rates = {}
                    self._empirical_country_action = {}
                    self._empirical_solution_action = {}
                    grp = train_df.groupby(["Country", "solution", "last_action"]) ["ever_won"].agg(["mean","count"]).reset_index()
                    for _, r in grp.iterrows():
                        k = (str(r["Country"]).upper(), str(r["solution"]).upper(), str(r["last_action"]))
                        self._empirical_win_rates[k] = (float(r["mean"]), int(r["count"]))
                    grp_ca = train_df.groupby(["Country", "last_action"])["ever_won"].agg(["mean","count"]).reset_index()
                    for _, r in grp_ca.iterrows():
                        k = (str(r["Country"]).upper(), str(r["last_action"]))
                        self._empirical_country_action[k] = (float(r["mean"]), int(r["count"]))
                    grp_sa = train_df.groupby(["solution", "last_action"])["ever_won"].agg(["mean","count"]).reset_index()
                    for _, r in grp_sa.iterrows():
                        k = (str(r["solution"]).upper(), str(r["last_action"]))
                        self._empirical_solution_action[k] = (float(r["mean"]), int(r["count"]))
                    try:
                        global_win = float(train_df["ever_won"].mean())
                    except Exception:
                        global_win = None
                    self._empirical_global = float(global_win) if (global_win is not None and not np.isnan(global_win)) else None

                self._empirical_built = True

            except Exception as ex:
                self._empirical_win_rates = {}
                self._empirical_country_action = {}
                self._empirical_solution_action = {}
                self._empirical_global = None
                self._empirical_built = True

        def lookup_emp(c,s,a):
            k = (str(c).upper(), str(s).upper(), str(a))
            if k in self._empirical_win_rates:
                return float(self._empirical_win_rates[k][0])
            k2 = (str(c).upper(), str(a))
            if k2 in self._empirical_country_action:
                return float(self._empirical_country_action[k2][0])
            k3 = (str(s).upper(), str(a))
            if k3 in self._empirical_solution_action:
                return float(self._empirical_solution_action[k3][0])
            if self._empirical_global is not None:
                return float(self._empirical_global)
            return None

        if self.dt_outcome is None:
            val = lookup_emp(country, solution, action)
            if val is None:
                self._p_cache[key] = None
                return None
            if self._empirical_global is not None and val == self._empirical_global:
                self._p_cache[key] = None
                return None
            self._p_cache[key] = float(val)
            return self._p_cache[key]


        try:
            last_act = last_action if last_action is not None else "NONE"
            X = self._build_context_matrix(country, solution, last_act, candidate_action=action, num_past_touches=num_past_touches + 1, days_since_prev_touch=0.0, is_lead=is_lead)
        except Exception as ex:
            X = None

        if isinstance(X, np.ndarray):
            if float(np.sum(np.abs(X))) == 0.0:
                val = lookup_emp(country, solution, action)
                self._p_cache[key] = float(val) if val is not None else 0.0
                return self._p_cache[key]
        else:
            val = lookup_emp(country, solution, action)
            self._p_cache[key] = float(val) if val is not None else 0.0
            return self._p_cache[key]

        try:
            proba = self.dt_outcome.predict_proba(X)[0]
            classes = list(self.dt_outcome.classes_)
            pos_idx = None
            for i,cls in enumerate(classes):
                try:
                    scls = str(cls).strip().lower()
                    if scls in {"1","true","won","yes","y","t"}:
                        pos_idx = i
                        break
                    if isinstance(cls, (int, np.integer)) and int(cls) == 1:
                        pos_idx = i
                        break
                except Exception:
                    continue
            if pos_idx is None and len(proba) == 2:
                pos_idx = 1
            if pos_idx is None:
                pos_idx = min(1, len(proba)-1)
            pos_prob = float(proba[pos_idx])
            if not np.isfinite(pos_prob):
                raise ValueError("non-finite prob")
            self._p_cache[key] = max(0.0, min(1.0, pos_prob))
            return self._p_cache[key]
        except Exception:
            val = lookup_emp(country, solution, action)
            self._p_cache[key] = float(val) if val is not None else 0.0
            return self._p_cache[key]

    def _lookup_empirical_win_rate(self, country, solution, action):
        rates = getattr(self, "_empirical_win_rates", {})
        c = str(country).upper()
        s = str(solution).upper()
        a_raw = str(action).strip()
        variants = [a_raw, a_raw.title(), a_raw.upper()]
        for av in variants:
            k = (c, s, av)
            if k in rates:
                return float(rates[k][0])
        k2 = (c, s, "NONE")
        if k2 in rates:
            return float(rates[k2][0])
        return float(rates.get("__global__", (0.2, 0))[0])

    def top_n_actions_for_account(self, account_id, country, solution, is_lead=1, n=4):

        st = self.ws.state.get(account_id, {})
        last_action = None
        if st and st.get("actions"):
            last_action = st["actions"][-1]
        last_action = last_action or "NONE"

        try:
            probs = self.predict_next_action_probs(country, solution, last_action,
                                                num_past_touches=len(st.get("actions", [])),
                                                days_since_prev_touch=0.0, is_lead=is_lead)
        except Exception:
            probs = {a: 0.0 for a in self.actions}

        scored = []
        for action in [a for a in self.actions if a not in {"NONE", "UNKNOWN"}]:
            p_next = float(probs.get(action, 0.0))
            w = float(self.ws.get_weight(account_id, action))
            score = float(p_next * w)
            scored.append((action, round(p_next, 6), round(w, 6), round(score, 6)))

        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:n]

    def build_journey(self, account_id, country, solution, target="win", steps=MAX_JOURNEY_STEPS, is_lead=1):
        if target not in {"win","loss"}:
            raise ValueError("target must be 'win' or 'loss'")
        journey = []
        from copy import deepcopy
        ws_snapshot = deepcopy(self.ws)

        prev_action = None
        for step in range(steps):
            best_candidate = None
            current_num = len(ws_snapshot.state.get(account_id, {}).get("actions", []))

            try:
                p_next_map = self.predict_next_action_probs(country, solution, prev_action or "NONE",
                                                           num_past_touches=current_num,
                                                           days_since_prev_touch=0.0,
                                                           is_lead=is_lead)
            except Exception:
                p_next_map = {a: 0.0 for a in self.actions}

            for action in [a for a in self.actions if a not in {"NONE", "UNKNOWN"}]:
                if prev_action is not None and action == prev_action:
                    continue

                p_win = self.predict_outcome_prob_given_action(country, solution, action,
                                                               num_past_touches=current_num,
                                                               days_since_prev_touch=0.0,
                                                               is_lead=is_lead,
                                                               last_action=prev_action or "NONE")
                if p_win is None or (isinstance(p_win, float) and (not np.isfinite(p_win))):
                    try:
                        p_win_score = float(p_next_map.get(action, 0.0))
                    except Exception:
                        p_win_score = 0.0
                else:
                    p_win_score = float(p_win)

                pnext = float(p_next_map.get(action, 0.0))
                w = ws_snapshot.get_weight(account_id, action)

                if p_win is None or (isinstance(p_win, float) and (not np.isfinite(p_win))):
                    base = pnext
                else:
                    base = p_win_score

                if target == "win":
                    score = ((0.75 * base) + (0.25 * pnext)) * float(w)
                else:
                    inv = (1.0 - base)
                    score = ((0.85 * (inv ** 1.1)) + (0.15 * ((1.0 - pnext) ** 1.05))) * float(w)

                candidate = {"action": action, "p_win": (None if p_win is None else float(p_win)), "p_win_score": p_win_score, "weight": float(w), "score": float(score)}
                if best_candidate is None or candidate["score"] > best_candidate["score"]:
                    best_candidate = candidate

            if best_candidate is None:
                break

            action = best_candidate["action"]
            ws_snapshot.add_action(account_id, action, persist=False)
            journey.append({"step": len(journey)+1, "action": action, "weight_used": round(float(best_candidate["weight"]), 6), "score": round(float(best_candidate["score"]), 6)})
            prev_action = action

            if best_candidate.get("p_win") is not None:
                if target == "win" and best_candidate["p_win"] >= WIN_THRESHOLD:
                    break
                if target == "loss" and (1.0 - best_candidate["p_win"]) >= LOSS_THRESHOLD:
                    break
        return journey