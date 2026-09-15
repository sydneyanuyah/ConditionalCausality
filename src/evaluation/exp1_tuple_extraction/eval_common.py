"""
eval_common.py
Shared parsing and metric utilities for Exp1a and Exp1b conditional causal tuple evaluation.

Prediction CSVs are expected to contain an id column and an output column.
The output column can be one of:
  - {"output": {"predicted_tuples": [...]}, "raw_generation": "..."}
  - {"predicted_tuples": [...]}
  - a raw list of tuple dictionaries

Gold CSVs can use dataset-specific column names. The wrapper scripts define presets.
"""

from __future__ import annotations

import ast
import itertools
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

FIELDS = ["e1", "relation", "e2", "condition"]
FILLER_WORDS = {"in", "the", "a", "an", "at", "of", "on", "with", "by", "is", "are", "to", "and"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    id_col: str
    text_col: str
    gold_col: str


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "synthetic": DatasetSpec("synthetic", "Paragraph_ID", "Paragraph", "Gold Annotation"),
    "pubmed": DatasetSpec("pubmed", "Paragraph_ID", "Paragraph", "Gold Annotation"),
    "reddit": DatasetSpec("reddit", "ID", "Paragraph", "Tuples"),
}


def clean_value(val: Any) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(val).strip().lower())


def tokenize(text: Any, remove_stopwords: bool = False) -> List[str]:
    if not isinstance(text, str):
        return []
    tokens = re.findall(r"[\w']+|[.,!?;]", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in FILLER_WORDS]
    return tokens


def safe_literal_or_json(raw_value: Any) -> Any:
    if isinstance(raw_value, (list, dict)):
        return raw_value
    if raw_value is None:
        return []
    try:
        if pd.isna(raw_value):
            return []
    except Exception:
        pass

    raw_text = str(raw_value).strip()
    if not raw_text:
        return []

    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE).strip()
    raw_text = re.sub(r"\s*```$", "", raw_text).strip()

    try:
        return json.loads(raw_text)
    except Exception:
        pass

    try:
        return ast.literal_eval(raw_text)
    except Exception:
        return []


def extract_tuple_list(parsed_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed_obj, list):
        return parsed_obj
    if not isinstance(parsed_obj, dict):
        return []

    if "predicted_tuples" in parsed_obj:
        return parsed_obj.get("predicted_tuples") or []

    if "output" in parsed_obj:
        inner = parsed_obj.get("output")
        if isinstance(inner, str):
            inner = safe_literal_or_json(inner)
        if isinstance(inner, dict) and "predicted_tuples" in inner:
            return inner.get("predicted_tuples") or []
        if isinstance(inner, list):
            return inner
        if isinstance(inner, dict) and all(k in inner for k in ["e1", "e2"]):
            return [inner]
        return []

    if all(k in parsed_obj for k in ["e1", "e2"]):
        return [parsed_obj]

    return []


def split_on_conjunctions(tuple_dict: Dict[str, Any], split_fields: Iterable[str]) -> List[Dict[str, Any]]:
    split_fields = set(split_fields)
    splits: Dict[str, List[Any]] = {}
    for field in FIELDS:
        val = tuple_dict.get(field, "")
        if field in split_fields and isinstance(val, str):
            parts = re.split(r"\s+(?:and|or)\s+", val, flags=re.IGNORECASE)
            splits[field] = [p.strip() for p in parts if p.strip()]
        else:
            splits[field] = [val]
        if not splits[field]:
            splits[field] = [""]

    return [dict(zip(FIELDS, combo)) for combo in itertools.product(*[splits[f] for f in FIELDS])]


def expand_tuples(raw_tuples: Any, split_fields: Iterable[str]) -> List[Dict[str, Any]]:
    if isinstance(raw_tuples, dict):
        raw_tuples = [raw_tuples]
    if not isinstance(raw_tuples, list):
        return []

    expanded: List[Dict[str, Any]] = []
    for t in raw_tuples:
        if isinstance(t, dict):
            expanded.extend(split_on_conjunctions(t, split_fields=split_fields))
    return expanded


def normalize_tuple(t: Dict[str, Any]) -> Dict[str, str]:
    return {field: clean_value(t.get(field, "")) for field in FIELDS}


def parse_gold_cell(x: Any, split_fields: Iterable[str]) -> List[Dict[str, str]]:
    raw = safe_literal_or_json(x)
    return [normalize_tuple(t) for t in expand_tuples(raw, split_fields=split_fields)]


def parse_prediction_cell(x: Any, split_fields: Iterable[str]) -> Tuple[List[Dict[str, str]], str]:
    try:
        raw = safe_literal_or_json(x)
        tuples = extract_tuple_list(raw)
        return [normalize_tuple(t) for t in expand_tuples(tuples, split_fields=split_fields)], "Success"
    except Exception as e:
        return [], f"Error: {e}"


def token_set(text: Any) -> set:
    return set(tokenize(clean_value(text), remove_stopwords=True))


def token_jaccard(a: Any, b: Any) -> float:
    a_set, b_set = token_set(a), token_set(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def token_f1_score(a: Any, b: Any) -> float:
    a_set, b_set = token_set(a), token_set(b)
    if not a_set or not b_set:
        return 0.0
    overlap = len(a_set & b_set)
    p = overlap / len(a_set) if a_set else 0.0
    r = overlap / len(b_set) if b_set else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def char_similarity(a: Any, b: Any) -> float:
    a_clean, b_clean = clean_value(a), clean_value(b)
    if not a_clean or not b_clean:
        return 0.0
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def containment_score(a: Any, b: Any) -> float:
    a_clean, b_clean = clean_value(a), clean_value(b)
    if not a_clean or not b_clean:
        return 0.0
    if a_clean in b_clean or b_clean in a_clean:
        return min(len(a_clean), len(b_clean)) / max(len(a_clean), len(b_clean))
    return 0.0


def relaxed_field_score(a: Any, b: Any) -> float:
    return max(token_jaccard(a, b), token_f1_score(a, b), char_similarity(a, b), containment_score(a, b))


def get_all_occurrences(text_tokens: List[str], substring: Any) -> List[set]:
    sub_tokens = tokenize(clean_value(substring), remove_stopwords=True)
    m = len(sub_tokens)
    if m == 0:
        return []
    return [set(range(i, i + m)) for i in range(len(text_tokens) - m + 1) if text_tokens[i:i + m] == sub_tokens]


def best_span_iou(pred_text: Any, gold_text: Any, text_tokens: List[str]) -> float:
    pred_spans = get_all_occurrences(text_tokens, pred_text)
    gold_spans = get_all_occurrences(text_tokens, gold_text)
    if not pred_spans or not gold_spans:
        return 0.0
    best = 0.0
    for p_span in pred_spans:
        for g_span in gold_spans:
            union = len(p_span | g_span)
            best = max(best, len(p_span & g_span) / union if union else 0.0)
    return best


def soft_text_match(pred_text: Any, gold_text: Any, text_tokens: List[str], threshold: float) -> bool:
    p, g = clean_value(pred_text), clean_value(gold_text)
    if p == g and p:
        return True
    if not p or not g:
        return False
    return best_span_iou(p, g, text_tokens) >= threshold or token_jaccard(p, g) >= threshold


def prf(tp: int, n_pred: int, n_gold: int) -> Tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def maximum_bipartite_matches(pred: List[Dict[str, str]], gold: List[Dict[str, str]], match_fn: Callable[[Dict[str, str], Dict[str, str]], bool]) -> int:
    if not pred or not gold:
        return 0
    adjacency = [[g_idx for g_idx, g in enumerate(gold) if match_fn(p, g)] for p in pred]
    matched_pred_for_gold: Dict[int, int] = {}

    def dfs(p_idx: int, seen: set) -> bool:
        for g_idx in adjacency[p_idx]:
            if g_idx in seen:
                continue
            seen.add(g_idx)
            if g_idx not in matched_pred_for_gold or dfs(matched_pred_for_gold[g_idx], seen):
                matched_pred_for_gold[g_idx] = p_idx
                return True
        return False

    count = 0
    for p_idx in range(len(pred)):
        if dfs(p_idx, set()):
            count += 1
    return count


def evaluate_strict_rows(gold_df: pd.DataFrame, pred_df: pd.DataFrame, spec: DatasetSpec, pred_id_col: str, pred_output_col: str, threshold: float, split_fields: Iterable[str]):
    gold_by_id: Dict[str, List[Dict[str, str]]] = {}
    tokens_by_id: Dict[str, List[str]] = {}

    for _, row in gold_df.iterrows():
        row_id = str(row[spec.id_col])
        gold_by_id[row_id] = parse_gold_cell(row[spec.gold_col], split_fields=split_fields)
        tokens_by_id[row_id] = tokenize(row.get(spec.text_col, ""), remove_stopwords=True)

    pred_by_id: Dict[str, List[Dict[str, str]]] = {}
    status_by_id: Dict[str, str] = {}
    for _, row in pred_df.iterrows():
        row_id = str(row[pred_id_col])
        pred_by_id[row_id], status_by_id[row_id] = parse_prediction_cell(row[pred_output_col], split_fields=split_fields)

    rows = []
    totals = {"n_gold": 0, "n_pred": 0, "row_exact": 0}
    metric_names = ["e1", "relation", "e2", "condition", "re_triplet", "full_tuple"]
    for m in metric_names:
        totals[f"{m}_tp"] = 0

    for row_id, gold in gold_by_id.items():
        pred = pred_by_id.get(row_id, [])
        text_tokens = tokens_by_id.get(row_id, [])
        n_gold, n_pred = len(gold), len(pred)

        def e1_match(p, g): return bool(g["e1"]) and p["e1"] == g["e1"]
        def e2_match(p, g): return bool(g["e2"]) and p["e2"] == g["e2"]
        def relation_match(p, g): return soft_text_match(p["relation"], g["relation"], text_tokens, threshold)
        def condition_match(p, g): return soft_text_match(p["condition"], g["condition"], text_tokens, threshold)
        def re_triplet_match(p, g): return e1_match(p, g) and relation_match(p, g) and e2_match(p, g)
        def full_tuple_match(p, g): return re_triplet_match(p, g) and condition_match(p, g)

        matchers = {
            "e1": e1_match, "relation": relation_match, "e2": e2_match,
            "condition": condition_match, "re_triplet": re_triplet_match, "full_tuple": full_tuple_match,
        }
        row = {"ID": row_id, "dataset": spec.name, "n_gold": n_gold, "n_pred": n_pred, "parse_status": status_by_id.get(row_id, "Missing prediction row")}
        metric_tps = {}
        for metric in metric_names:
            tp = maximum_bipartite_matches(pred, gold, matchers[metric])
            metric_tps[metric] = tp
            p, r, f = prf(tp, n_pred, n_gold)
            row[f"{metric}_tp"] = tp
            row[f"{metric}_precision"] = round(p, 4)
            row[f"{metric}_recall"] = round(r, 4)
            row[f"{metric}_f1"] = round(f, 4)
            totals[f"{metric}_tp"] += tp

        row_exact = int(metric_tps["full_tuple"] == n_gold == n_pred)
        row["row_exact_match"] = row_exact
        row["RE_given_C"] = round(metric_tps["full_tuple"] / metric_tps["condition"], 4) if metric_tps["condition"] else 0.0
        row["C_given_RE"] = round(metric_tps["full_tuple"] / metric_tps["re_triplet"], 4) if metric_tps["re_triplet"] else 0.0
        rows.append(row)

        totals["n_gold"] += n_gold
        totals["n_pred"] += n_pred
        totals["row_exact"] += row_exact

    detail_df = pd.DataFrame(rows)
    summary_rows = []
    for metric in metric_names:
        tp = totals[f"{metric}_tp"]
        p, r, f = prf(tp, totals["n_pred"], totals["n_gold"])
        summary_rows.append({"metric": metric, "tp": tp, "n_pred": totals["n_pred"], "n_gold": totals["n_gold"], "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)})
    full_tp = totals["full_tuple_tp"]
    cond_tp = totals["condition_tp"]
    re_tp = totals["re_triplet_tp"]
    summary_rows.extend([
        {"metric": "row_exact_match", "tp": totals["row_exact"], "n_pred": len(rows), "n_gold": len(rows), "precision": "", "recall": "", "f1": round(totals["row_exact"] / len(rows), 4) if rows else 0.0},
        {"metric": "RE_given_C", "tp": full_tp, "n_pred": cond_tp, "n_gold": cond_tp, "precision": "", "recall": "", "f1": round(full_tp / cond_tp, 4) if cond_tp else 0.0},
        {"metric": "C_given_RE", "tp": full_tp, "n_pred": re_tp, "n_gold": re_tp, "precision": "", "recall": "", "f1": round(full_tp / re_tp, 4) if re_tp else 0.0},
    ])
    return detail_df, pd.DataFrame(summary_rows)


def evaluate_recall_rows(gold_df: pd.DataFrame, pred_df: pd.DataFrame, spec: DatasetSpec, pred_id_col: str, pred_output_col: str, threshold: float, split_fields: Iterable[str]):
    gold_by_id: Dict[str, Dict[str, Any]] = {}
    for _, row in gold_df.iterrows():
        row_id = str(row[spec.id_col])
        gold_by_id[row_id] = {"gold": parse_gold_cell(row[spec.gold_col], split_fields=split_fields), "paragraph": str(row.get(spec.text_col, ""))}

    pred_by_id: Dict[str, List[Dict[str, str]]] = {}
    status_by_id: Dict[str, str] = {}
    for _, row in pred_df.iterrows():
        row_id = str(row[pred_id_col])
        pred_by_id[row_id], status_by_id[row_id] = parse_prediction_cell(row[pred_output_col], split_fields=split_fields)

    detail_rows = []
    for row_id, item in gold_by_id.items():
        gold = item["gold"]
        pred = pred_by_id.get(row_id, [])
        n_gold, n_pred = len(gold), len(pred)
        matched_gold = set()
        hits = {"e1": 0, "relation": 0, "e2": 0, "condition": 0, "re": 0, "tuple": 0}

        for p in pred:
            best_idx = -1
            best_score = -1
            best_match = None
            for g_idx, g in enumerate(gold):
                if g_idx in matched_gold:
                    continue
                e1_m = relaxed_field_score(p["e1"], g["e1"]) >= threshold
                e2_m = relaxed_field_score(p["e2"], g["e2"]) >= threshold
                rel_m = relaxed_field_score(p["relation"], g["relation"]) >= threshold
                cond_m = relaxed_field_score(p["condition"], g["condition"]) >= threshold
                re_m = e1_m and e2_m and rel_m
                tuple_m = re_m and cond_m
                score = sum([e1_m, e2_m, rel_m, cond_m])
                if score > best_score:
                    best_idx = g_idx
                    best_score = score
                    best_match = (e1_m, rel_m, e2_m, cond_m, re_m, tuple_m)
            if best_idx != -1 and best_match is not None:
                matched_gold.add(best_idx)
                e1_m, rel_m, e2_m, cond_m, re_m, tuple_m = best_match
                hits["e1"] += int(e1_m)
                hits["relation"] += int(rel_m)
                hits["e2"] += int(e2_m)
                hits["condition"] += int(cond_m)
                hits["re"] += int(re_m)
                hits["tuple"] += int(tuple_m)

        p_tuple, r_tuple, f_tuple = prf(hits["tuple"], n_pred, n_gold)
        detail_rows.append({
            "ID": row_id,
            "dataset": spec.name,
            "parse_status": status_by_id.get(row_id, "Missing prediction row"),
            "n_gold": n_gold,
            "n_pred": n_pred,
            "tuple_hits": hits["tuple"],
            "e1_hits": hits["e1"],
            "relation_hits": hits["relation"],
            "e2_hits": hits["e2"],
            "condition_hits": hits["condition"],
            "re_hits": hits["re"],
            "P_tuple": p_tuple,
            "R_tuple": r_tuple,
            "F1_tuple": f_tuple,
            "R_E1": hits["e1"] / n_gold if n_gold else 0.0,
            "R_R": hits["relation"] / n_gold if n_gold else 0.0,
            "R_E2": hits["e2"] / n_gold if n_gold else 0.0,
            "R_C": hits["condition"] / n_gold if n_gold else 0.0,
            "R_RE": hits["re"] / n_gold if n_gold else 0.0,
            "Delta_condition": (hits["re"] / n_gold if n_gold else 0.0) - r_tuple,
            "R_RE_given_C": hits["tuple"] / hits["condition"] if hits["condition"] else 0.0,
            "R_C_given_RE": hits["tuple"] / hits["re"] if hits["re"] else 0.0,
            "PGC_row": int(hits["tuple"] == n_gold) if n_gold else 0,
        })

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    total_gold = int(detail_df["n_gold"].sum())
    total_pred = int(detail_df["n_pred"].sum())
    tuple_hits = int(detail_df["tuple_hits"].sum())
    e1_hits = int(detail_df["e1_hits"].sum())
    rel_hits = int(detail_df["relation_hits"].sum())
    e2_hits = int(detail_df["e2_hits"].sum())
    cond_hits = int(detail_df["condition_hits"].sum())
    re_hits = int(detail_df["re_hits"].sum())
    p_tuple, r_tuple, f_tuple = prf(tuple_hits, total_pred, total_gold)

    summary = pd.DataFrame([{
        "Dataset": spec.name,
        "R_tuple": r_tuple,
        "F1_tuple": f_tuple,
        "R_E1": e1_hits / total_gold if total_gold else 0.0,
        "R_R": rel_hits / total_gold if total_gold else 0.0,
        "R_E2": e2_hits / total_gold if total_gold else 0.0,
        "R_C": cond_hits / total_gold if total_gold else 0.0,
        "R_RE": re_hits / total_gold if total_gold else 0.0,
        "Delta_condition": (re_hits / total_gold if total_gold else 0.0) - r_tuple,
        "R_RE_given_C": tuple_hits / cond_hits if cond_hits else 0.0,
        "R_C_given_RE": tuple_hits / re_hits if re_hits else 0.0,
        "PGC": float(detail_df["PGC_row"].mean()),
        "P_tuple": p_tuple,
        "total_gold_tuples": total_gold,
        "total_pred_tuples": total_pred,
        "rows": len(detail_df),
        "parse_errors": int((detail_df["parse_status"] != "Success").sum()),
        "iou_threshold": threshold,
    }])
    return detail_df, summary
