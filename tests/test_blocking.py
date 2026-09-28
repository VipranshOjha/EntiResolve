"""
tests/test_blocking.py — Comprehensive unit tests for Phase 3 blocking engine.

Tests:
01  exact_normalized_name retrieval
02  legal suffix normalization matching
03  compact name retrieval (whitespace/punct removal)
04  website-style name retrieval (compact_no_tld)
05  name token retrieval (informative token match)
06  name token stopword filtering (inc, corp, ltd ignored)
07  name token max_df pruning
08  character 4-gram retrieval (typo tolerance)
09  address token retrieval (distinctive locality/street)
10  address stopword filtering (street, road, ave ignored)
11  address structural: house number + postal PIN code
12  address structural: house number + first name token
13  non-Latin fallback retrieval via address tokens
14  corrupted-name fallback retrieval via structural signals
15  empty / missing address query
16  empty / missing name query
17  composite blocker: union and deduplication
18  composite blocker: candidate volume capping
19  composite blocker: source preservation (S2 vs S3)
20  composite blocker: attribution tracking
21  blocking evaluator: candidate recall calculation
22  blocking evaluator: reduction ratio calculation
23  deterministic candidate ordering
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blocking.address_structural import AddressStructuralBlocker
from src.blocking.address_token import AddressTokenBlocker
from src.blocking.compact_name import CompactNameBlocker
from src.blocking.composite import CompositeBlocker
from src.blocking.evaluator import BlockingEvaluator
from src.blocking.exact_name import ExactNameBlocker
from src.blocking.name_token import NameTokenBlocker
from src.blocking.ngram_blocker import CharNgramBlocker
from src.normalization.record import normalize_record


# ---------------------------------------------------------------------------
# Test Harness
# ---------------------------------------------------------------------------

_PASSED: list[str] = []
_FAILED: list[str] = []


def _run(name: str, fn) -> None:
    try:
        fn()
        _PASSED.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:
        _FAILED.append(name)
        print(f"  FAIL  {name}  -- {exc}")
        import traceback
        traceback.print_exc()


def _eq(a, b, msg=""):
    assert a == b, f"{msg}: expected {b!r}, got {a!r}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_exact_normalized_name():
    blocker = ExactNameBlocker()
    t1 = normalize_record("S2-1", "Urology Partners Inc", "123 Main St", "US")
    t2 = normalize_record("S3-2", "Global Logistics LLC", "456 Oak Ave", "US")
    blocker.index_record(0, t1)
    blocker.index_record(1, t2)
    blocker.finalize()

    q = normalize_record("S1-100", "urology partners inc", "123 Main St", "US")
    cands = blocker.query(q)
    _eq(cands, [0])

    q_miss = normalize_record("S1-200", "Different Name Inc", "123 Main St", "US")
    _eq(blocker.query(q_miss), [])


def test_legal_suffix_normalization_matching():
    blocker = ExactNameBlocker()
    # Target has 'LLC', query has 'L.L.C.' or no suffix
    t = normalize_record("S2-1", "Acme Services LLC", "123 Main St", "US")
    blocker.index_record(0, t)
    blocker.finalize()

    q = normalize_record("S1-1", "Acme Services L.L.C.", "123 Main St", "US")
    cands = blocker.query(q)
    _eq(cands, [0])


def test_compact_name_retrieval():
    blocker = CompactNameBlocker()
    t = normalize_record("S2-1", "Jeewan Jyoti Appartment", "Rewari, Haryana", "India")
    blocker.index_record(0, t)

    # Query with slightly different punctuation/spacing
    q = normalize_record("S1-1", "Jeewan-Jyoti Appartment", "Rewari, Haryana", "India")
    cands = blocker.query(q)
    _eq(cands, [0])


def test_website_style_name_retrieval():
    blocker = CompactNameBlocker()
    # Target has website name
    t = normalize_record("S3-1", "jeewanjyotiappartment.com", "Rewari, Haryana", "India")
    blocker.index_record(0, t)

    # Query has normal business name
    q = normalize_record("S1-1", "Jeewan Jyoti Appartment", "Rewari, Haryana", "India")
    cands = blocker.query(q)
    _eq(cands, [0])


def test_name_token_retrieval():
    blocker = NameTokenBlocker()
    t1 = normalize_record("S2-1", "Dahlia Power Scientific LLC", "Kansas City, MO", "US")
    t2 = normalize_record("S3-2", "Scientific Research Labs", "Chicago, IL", "US")
    blocker.index_record(0, t1)
    blocker.index_record(1, t2)
    blocker.finalize()

    # Query shares 'Dahlia' and 'Power'
    q = normalize_record("S1-1", "Dahlia Energy Solutions", "Kansas City, MO", "US")
    cands = blocker.query(q)
    _eq(cands, [0])


def test_name_token_stopword_filtering():
    blocker = NameTokenBlocker()
    t = normalize_record("S2-1", "Alpha Corporation", "New York, NY", "US")
    blocker.index_record(0, t)
    blocker.finalize()

    # Query shares ONLY 'Corporation' (stopword)
    q = normalize_record("S1-1", "Beta Corporation", "Boston, MA", "US")
    cands = blocker.query(q)
    _eq(cands, [], "Common business stopwords must not retrieve candidates")


def test_name_token_max_df_pruning():
    blocker = NameTokenBlocker(max_df=2)
    t1 = normalize_record("S2-1", "FrequentToken Alpha", "Address", "US")
    t2 = normalize_record("S2-2", "FrequentToken Beta", "Address", "US")
    t3 = normalize_record("S2-3", "FrequentToken Gamma", "Address", "US")
    blocker.index_record(0, t1)
    blocker.index_record(1, t2)
    blocker.index_record(2, t3)
    blocker.finalize()

    # 'FrequentToken' appeared 3 times (> max_df=2), should be pruned
    assert blocker._pruned_count >= 1
    assert "frequenttoken" not in blocker._index


def test_char_ngram_typo_retrieval():
    blocker = CharNgramBlocker()
    # Real data typo: 'Wilblims' vs 'Williams'
    t = normalize_record("S2-1", "Maure Wilblims Colombier", "Wayne Ave", "US")
    blocker.index_record(0, t)
    blocker.finalize()

    q = normalize_record("S1-1", "Maure Williams Colombier", "Wayne Ave", "US")
    cands = blocker.query(q)
    _eq(cands, [0])


def test_address_token_retrieval():
    blocker = AddressTokenBlocker()
    t = normalize_record("S2-1", "CompletelyDifferentName", "85 Wayne Avenue, Ticonderoga, NY", "US")
    blocker.index_record(0, t)
    blocker.finalize()

    q = normalize_record("S1-1", "Maure Williams", "85 Wayne Avenue, Ticonderoga, NY", "US")
    cands = blocker.query(q)
    _eq(cands, [0], "Distinctive locality token 'ticonderoga' should retrieve target")


def test_address_stopword_filtering():
    blocker = AddressTokenBlocker()
    t = normalize_record("S2-1", "Company One", "123 Main Street Suite 400", "US")
    blocker.index_record(0, t)
    blocker.finalize()

    # Only shares 'street' and 'suite'
    q = normalize_record("S1-1", "Company Two", "555 Oak Street Suite 200", "US")
    cands = blocker.query(q)
    _eq(cands, [], "Address stopwords must not retrieve candidates")


def test_address_structural_house_postal():
    blocker = AddressStructuralBlocker()
    t = normalize_record("S2-1", "Indian Plot Target", "KH NO. -570/13, 110001, Delhi", "India")
    blocker.index_record(0, t)
    blocker.finalize()

    q = normalize_record("S1-1", "Different Name Query", "570/13, 110001, New Delhi", "India")
    cands = blocker.query(q)
    _eq(cands, [0])


def test_address_structural_house_name():
    blocker = AddressStructuralBlocker()
    # Real data example: Payne Enterprises 3315 Fremont St
    t = normalize_record("S2-1", "Payne Enterpires", "3315 FREMONT SAINT", "US")
    blocker.index_record(0, t)
    blocker.finalize()

    q = normalize_record("S1-1", "Payne Enterprises", "3315 Fremont Street", "US")
    cands = blocker.query(q)
    _eq(cands, [0])


def test_non_latin_fallback():
    # S1 is English Latin, S2 is Tamil script
    comp = CompositeBlocker()
    t_tamil = normalize_record(
        "S2-249013014",
        "\u0bb0\u0bbe\u0b9c\u0bcd \u0b87\u0ba9\u0bcd\u0bb5\u0bc6\u0bb8\u0bcd\u0b9f\u0bcd\u0bae\u0bc6\u0ba3\u0bcd\u0b9f\u0bcd\u0bb8\u0bcd",  # Raj Investments in Tamil
        "6(29), C.I.T. Colony, Mylapore, Chennai, Tamil Nadu",
        "India",
    )
    comp.index_target_records([t_tamil])

    q_latin = normalize_record(
        "S1-55344266",
        "Raj Investments LLP",
        "6(29), C.I.T. Colony, Mylapore, Chennai, Tamil Nadu",
        "India",
    )
    cands = comp.query(q_latin)
    assert "S2-249013014" in cands, f"Non-Latin record must be retrieved via address signals! Got: {cands}"


def test_corrupted_name_fallback():
    comp = CompositeBlocker()
    t_corrupt = normalize_record(
        "S3-775321672",
        "Dr\ufffdxkor",
        "85 Wayne Avenue, Ticonderoga, New York",
        "US",
    )
    comp.index_target_records([t_corrupt])

    q = normalize_record(
        "S1-965667",
        "Maure Williams Colombier Inc",
        "85 Wayne Avenue, Ticonderoga, NY",
        "US",
    )
    cands = comp.query(q)
    assert "S3-775321672" in cands, f"Corrupted name record must be retrieved via address signals! Got: {cands}"


def test_empty_address_query():
    comp = CompositeBlocker()
    t = normalize_record("S2-1", "Apex Global Trading", "100 Broadway", "US")
    comp.index_target_records([t])

    # Query with empty address should still match via name
    q = normalize_record("S1-1", "Apex Global Trading", "", "US")
    cands = comp.query(q)
    _eq(cands, ["S2-1"])


def test_empty_name_query():
    comp = CompositeBlocker()
    t = normalize_record("S2-1", "Apex Global Trading", "100 Broadway, New York", "US")
    comp.index_target_records([t])

    # Query with empty name should still match via address
    q = normalize_record("S1-1", "", "100 Broadway, New York", "US")
    cands = comp.query(q)
    _eq(cands, ["S2-1"])


def test_composite_union_and_deduplication():
    comp = CompositeBlocker()
    t1 = normalize_record("S2-1", "Star Logistics", "45 Industrial Way", "US")
    t2 = normalize_record("S3-2", "Star Logistics Inc", "45 Industrial Way", "US")
    comp.index_target_records([t1, t2])

    q = normalize_record("S1-1", "Star Logistics", "45 Industrial Way", "US")
    cands = comp.query(q)
    _eq(len(cands), len(set(cands)), "Candidates must be strictly deduplicated")
    assert "S2-1" in cands
    assert "S3-2" in cands


def test_composite_volume_capping():
    # Ensure cap works
    comp = CompositeBlocker(max_total_candidates=5)
    targets = [
        normalize_record(f"S2-{i}", f"Star Logistics Branch {i}", "Address", "US")
        for i in range(20)
    ]
    comp.index_target_records(targets)

    q = normalize_record("S1-1", "Star Logistics", "Address", "US")
    cands = comp.query(q)
    assert len(cands) <= 5, f"Expected <= 5 candidates, got {len(cands)}"


def test_source_preservation():
    comp = CompositeBlocker()
    t1 = normalize_record("S2-888", "Delta Airlines", "Atlanta", "US")
    t2 = normalize_record("S3-999", "Delta Airlines", "Atlanta", "US")
    comp.index_target_records([t1, t2])

    q = normalize_record("S1-1", "Delta Airlines", "Atlanta", "US")
    cands = comp.query(q)
    assert "S2-888" in cands
    assert "S3-999" in cands


def test_attribution_tracking():
    comp = CompositeBlocker()
    t1 = normalize_record("S2-1", "Zenith Corp", "100 Park Ave", "US")
    comp.index_target_records([t1])

    q = normalize_record("S1-1", "Zenith Corp", "100 Park Ave", "US")
    cands, attr = comp.query(q, track_attribution=True)
    assert "S2-1" in cands
    assert "exact_normalized_name" in attr
    assert "S2-1" in attr["exact_normalized_name"]


def test_blocking_evaluator_metrics():
    evaluator = BlockingEvaluator()
    q1 = normalize_record("S1-1", "Alpha", "Addr", "US")
    q2 = normalize_record("S1-2", "Beta", "Addr", "India")

    queries = [q1, q2]
    # S1-1 retrieved S2-10 (true match) and S2-11 (false positive)
    # S1-2 retrieved S3-20 (true match)
    retrieved = [["S2-10", "S2-11"], ["S3-20"]]
    ground_truth = {
        "S1-1": {"S2-10"},
        "S1-2": {"S3-20"},
    }

    metrics = evaluator.evaluate(
        query_records=queries,
        retrieved_candidates=retrieved,
        ground_truth=ground_truth,
        total_target_pool_size=1000,
    )
    _eq(metrics.overall_recall, 1.0)
    _eq(metrics.total_candidates_generated, 3)
    _eq(metrics.mean_candidates_per_query, 1.5)
    assert metrics.reduction_ratio > 0.99


def test_blocking_evaluator_partial_recall():
    evaluator = BlockingEvaluator()
    q1 = normalize_record("S1-1", "Alpha", "Addr", "US")
    queries = [q1]
    # Ground truth has 2 matches, we retrieved only 1
    retrieved = [["S2-10"]]
    ground_truth = {"S1-1": {"S2-10", "S3-11"}}

    metrics = evaluator.evaluate(
        query_records=queries,
        retrieved_candidates=retrieved,
        ground_truth=ground_truth,
        total_target_pool_size=1000,
    )
    _eq(metrics.overall_recall, 0.5)
    _eq(metrics.s2_recall, 1.0)
    _eq(metrics.s3_recall, 0.0)


def test_deterministic_ordering():
    comp = CompositeBlocker()
    targets = [
        normalize_record(f"S2-{i}", f"Company {i}", "Address", "US")
        for i in range(10)
    ]
    comp.index_target_records(targets)
    q = normalize_record("S1-1", "Company 5", "Address", "US")
    run1 = comp.query(q)
    run2 = comp.query(q)
    _eq(run1, run2, "Query results must be deterministic across repeated calls")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests() -> int:
    tests = [
        ("01 exact_normalized_name", test_exact_normalized_name),
        ("02 legal_suffix_normalization", test_legal_suffix_normalization_matching),
        ("03 compact_name_retrieval", test_compact_name_retrieval),
        ("04 website_style_name_retrieval", test_website_style_name_retrieval),
        ("05 name_token_retrieval", test_name_token_retrieval),
        ("06 name_token_stopword_filtering", test_name_token_stopword_filtering),
        ("07 name_token_max_df_pruning", test_name_token_max_df_pruning),
        ("08 char_ngram_typo_retrieval", test_char_ngram_typo_retrieval),
        ("09 address_token_retrieval", test_address_token_retrieval),
        ("10 address_stopword_filtering", test_address_stopword_filtering),
        ("11 address_structural_house_postal", test_address_structural_house_postal),
        ("12 address_structural_house_name", test_address_structural_house_name),
        ("13 non_latin_fallback", test_non_latin_fallback),
        ("14 corrupted_name_fallback", test_corrupted_name_fallback),
        ("15 empty_address_query", test_empty_address_query),
        ("16 empty_name_query", test_empty_name_query),
        ("17 composite_union_and_deduplication", test_composite_union_and_deduplication),
        ("18 composite_volume_capping", test_composite_volume_capping),
        ("19 source_preservation", test_source_preservation),
        ("20 attribution_tracking", test_attribution_tracking),
        ("21 blocking_evaluator_metrics", test_blocking_evaluator_metrics),
        ("22 blocking_evaluator_partial_recall", test_blocking_evaluator_partial_recall),
        ("23 deterministic_ordering", test_deterministic_ordering),
    ]

    print("\n" + "=" * 64)
    print("  Running Phase 3 Blocking Engine Unit Tests")
    print("=" * 64)

    for name, fn in tests:
        _run(name, fn)

    print("-" * 64)
    n_pass = len(_PASSED)
    total = len(tests)
    print(f"  {n_pass}/{total} tests passed")
    if _FAILED:
        print(f"  FAILED: {', '.join(_FAILED)}")
        return 1
    print("  All tests passed [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
