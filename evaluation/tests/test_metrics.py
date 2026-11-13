"""Metric unit tests T1–T9 + oracle plumbing (plan_v2/03 §4)."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from evaluation.io_utils import load_qrels, load_run, write_trec_run
from evaluation.metrics import (
    aggregate_metrics,
    evaluate_run,
    per_query_metrics,
    rank_of_target,
)

ROOT = Path(__file__).resolve().parents[2]
DEV3_QRELS = ROOT / "data" / "dev3-qrels.txt"


def _qrels_n(n: int = 5) -> dict[str, str]:
    return {str(i): f"D{i}" for i in range(1, n + 1)}


def _rankings_at_rank(qrels: dict[str, str], rank: int, list_len: int = 20) -> dict[str, list[str]]:
    """Place each target at 1-based `rank`; fill other slots with distractors."""
    out = {}
    for qid, target in qrels.items():
        docs = [f"X{qid}_{j}" for j in range(list_len)]
        docs[rank - 1] = target
        out[qid] = docs
    return out


def test_t1_perfect():
    qrels = _qrels_n(10)
    rankings = _rankings_at_rank(qrels, rank=1)
    per_q, missing = per_query_metrics(rankings, qrels)
    agg = aggregate_metrics(per_q, n_missing_from_run=missing)
    assert missing == 0
    assert agg.ndcg10 == pytest.approx(1.0)
    assert agg.mrr == pytest.approx(1.0)
    assert agg.r10 == pytest.approx(1.0)
    assert agg.r100 == pytest.approx(1.0)
    assert agg.r1000 == pytest.approx(1.0)


def test_t2_rank2():
    qrels = _qrels_n(8)
    rankings = _rankings_at_rank(qrels, rank=2)
    per_q, missing = per_query_metrics(rankings, qrels)
    agg = aggregate_metrics(per_q, n_missing_from_run=missing)
    assert agg.mrr == pytest.approx(0.5)
    assert agg.ndcg10 == pytest.approx(1.0 / math.log2(3))
    assert agg.r10 == pytest.approx(1.0)


def test_t3_rank11():
    qrels = _qrels_n(6)
    rankings = _rankings_at_rank(qrels, rank=11, list_len=50)
    per_q, missing = per_query_metrics(rankings, qrels)
    agg = aggregate_metrics(per_q, n_missing_from_run=missing)
    assert agg.ndcg10 == pytest.approx(0.0)
    assert agg.r10 == pytest.approx(0.0)
    assert agg.r100 == pytest.approx(1.0)
    assert agg.mrr == pytest.approx(1.0 / 11)


def test_t4_absent():
    qrels = _qrels_n(4)
    rankings = {qid: [f"Z{i}" for i in range(20)] for qid in qrels}
    per_q, missing = per_query_metrics(rankings, qrels)
    agg = aggregate_metrics(per_q, n_missing_from_run=missing)
    assert agg.ndcg10 == 0.0
    assert agg.mrr == 0.0
    assert agg.r10 == 0.0
    assert agg.r100 == 0.0
    assert agg.r1000 == 0.0
    assert agg.not_found_rate == 1.0


def test_t5_missing_query():
    qrels = _qrels_n(3)
    rankings = _rankings_at_rank({"1": "D1", "2": "D2"}, rank=1)  # qid 3 missing
    per_q, missing = per_query_metrics(rankings, qrels)
    agg = aggregate_metrics(per_q, n_missing_from_run=missing)
    assert missing == 1
    # two perfect + one zero → means 2/3
    assert agg.mrr == pytest.approx(2.0 / 3)
    assert agg.r10 == pytest.approx(2.0 / 3)


def test_t6_mixed_hand_computed():
    # ranks: 1, 2, 5, 11, absent  → n=5
    qrels = {"a": "Ta", "b": "Tb", "c": "Tc", "d": "Td", "e": "Te"}
    rankings = {
        "a": ["Ta", "x", "y"],
        "b": ["x", "Tb", "y"],
        "c": ["x1", "x2", "x3", "x4", "Tc"] + [f"z{i}" for i in range(10)],
        "d": [f"w{i}" for i in range(10)] + ["Td"] + [f"w{i}" for i in range(11, 20)],
        "e": [f"u{i}" for i in range(20)],
    }
    per_q, missing = per_query_metrics(rankings, qrels)
    agg = aggregate_metrics(per_q, n_missing_from_run=missing)
    expected_mrr = (1.0 + 0.5 + 1 / 5 + 1 / 11 + 0.0) / 5
    expected_ndcg = (1.0 + 1 / math.log2(3) + 1 / math.log2(6) + 0.0 + 0.0) / 5
    expected_r10 = (1 + 1 + 1 + 0 + 0) / 5
    expected_r100 = (1 + 1 + 1 + 1 + 0) / 5
    assert agg.mrr == pytest.approx(expected_mrr)
    assert agg.ndcg10 == pytest.approx(expected_ndcg)
    assert agg.r10 == pytest.approx(expected_r10)
    assert agg.r100 == pytest.approx(expected_r100)
    assert missing == 0


@pytest.mark.skipif(not DEV3_QRELS.exists(), reason="Dev3 qrels not downloaded")
def test_t7_random_run_near_zero(tmp_path: Path):
    qrels = load_qrels(DEV3_QRELS)
    rng = random.Random(0)
    # Universe of distractors unlikely to hit targets
    distractors = [f"RAND{i}" for i in range(5000)]
    rankings = {}
    for qid in qrels:
        docs = rng.sample(distractors, 1000)
        rankings[qid] = docs
    run_path = tmp_path / "TEST_random_dev3.run"
    write_trec_run(run_path, rankings, run_name="TEST_random")
    per_q, agg = evaluate_run(run_path, DEV3_QRELS)
    assert agg.n_queries == 536
    assert agg.r10 < 0.01
    assert agg.r100 < 0.05
    assert agg.ndcg10 < 0.01
    assert agg.mrr < 0.01


def test_t8_ranx_agreement():
    ranx = pytest.importorskip("ranx")
    from evaluation.ranx_check import agree_with_ranx

    qrels = _qrels_n(20)
    rankings = {}
    for i, (qid, target) in enumerate(qrels.items(), start=1):
        rank = (i % 15) + 1
        rankings.update(_rankings_at_rank({qid: target}, rank=rank, list_len=50))
    # merge properly
    rankings = {}
    for i, (qid, target) in enumerate(qrels.items(), start=1):
        rank = (i % 15) + 1
        rankings[qid] = _rankings_at_rank({qid: target}, rank=rank, list_len=50)[qid]

    ok, detail = agree_with_ranx(rankings, qrels, tol=1e-6)
    assert ok, detail


def test_t9_duplicate_docids(tmp_path: Path):
    run_path = tmp_path / "dup.run"
    # Same doc appears twice; higher score first occurrence should win position
    run_path.write_text(
        "1 Q0 D1 1 10.0 runx\n"
        "1 Q0 D1 2 9.0 runx\n"
        "1 Q0 D2 3 8.0 runx\n",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="duplicate"):
        rankings = load_run(run_path)
    assert rankings["1"] == ["D1", "D2"]
    qrels = {"1": "D1"}
    per_q, _ = per_query_metrics(rankings, qrels)
    assert per_q[0].rank_of_target == 1


@pytest.mark.skipif(not DEV3_QRELS.exists(), reason="Dev3 qrels not downloaded")
def test_oracle_plumbing(tmp_path: Path):
    qrels = load_qrels(DEV3_QRELS)
    rankings = {qid: [docid] + [f"Z{qid}_{i}" for i in range(999)] for qid, docid in qrels.items()}
    run_path = tmp_path / "TEST_oracle_dev3.run"
    write_trec_run(run_path, rankings, run_name="TEST_oracle")
    per_q, agg = evaluate_run(run_path, DEV3_QRELS)
    assert agg.n_queries == 536
    assert agg.ndcg10 == pytest.approx(1.0)
    assert agg.mrr == pytest.approx(1.0)
    assert agg.r1000 == pytest.approx(1.0)
    # Delete oracle artifact (definition of done)
    run_path.unlink()
    assert not run_path.exists()


def test_rank_helper():
    assert rank_of_target(["a", "b", "c"], "b") == 2
    assert rank_of_target(["a", "b"], "z") is None


def test_success_equals_recall():
    qrels = _qrels_n(5)
    rankings = _rankings_at_rank(qrels, rank=3)
    per_q, _ = per_query_metrics(rankings, qrels)
    for pq in per_q:
        assert pq.r10 == 1.0  # Success@10 == Recall@10
