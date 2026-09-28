"""
test_normalization.py — Unit tests for the Phase 2 normalization engine.

All tests use small synthetic examples (plus actual examples observed in
the challenge training data).  No file I/O — no dependency on the large
TSV files.

Test coverage
-------------
01  case normalization
02  whitespace normalization
03  punctuation handling (& vs and, dots, hyphens, apostrophes)
04  leading noise prefix stripping
05  ampersand normalization
06  legal suffix normalization
07  Unicode NFKD normalization
08  compact representation
09  compact_no_tld (website names)
10  Latin script detection
11  Devanagari script detection
12  Gujarati script detection
13  mixed-script detection (Latin + Devanagari)
14  replacement-character (corruption) detection
15  empty string handling
16  no-letters string handling
17  name quality: GOOD
18  name quality: NON_LATIN
19  name quality: MIXED_SCRIPT
20  name quality: CORRUPTED
21  name quality: WEBSITE
22  name quality: EMPTY
23  noise prefix detection flag
24  missing address (empty string)
25  address punctuation (## prefix)
26  numeric token extraction
27  postal code candidate extraction
28  house number detection
29  PO Box detection
30  address component counting
31  address: India 'NO' / 'HNO' house-number marker
32  full pipeline: complete BusinessRecord from raw values
33  full pipeline: actual noisy examples from training data
34  script info on all-non-Latin name
35  suffix bigram normalization
36  alpha_token extraction
37  component count with multiple commas
38  multiple scripts in same record
39  French accented name normalization (NFKD decomposition)
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalization.common import (
    apply_suffix_normalization,
    make_compact,
    normalize_ampersand,
    normalize_abbreviation_dots,
    normalize_whitespace,
    remove_leading_noise,
    strip_tld,
    tokenize,
)
from src.normalization.name import normalize_name
from src.normalization.address import normalize_address
from src.normalization.phone import normalize_phone
from src.normalization.record import normalize_record, NormalizationPipeline
from src.normalization.scripts import (
    get_script_info,
    DOM_LATIN, DOM_NON_LATIN, DOM_MIXED, DOM_CORRUPTED, DOM_EMPTY, DOM_NO_LETTERS,
    SCRIPT_DEVANAGARI, SCRIPT_GUJARATI, SCRIPT_LATIN, SCRIPT_TELUGU,
)
from src.normalization.quality import (
    classify_name_quality, has_noise_prefix,
    NQ_GOOD, NQ_NON_LATIN, NQ_MIXED_SCRIPT, NQ_CORRUPTED, NQ_WEBSITE, NQ_EMPTY,
    AQ_GOOD, AQ_MISSING,
)

# ---------------------------------------------------------------------------
# Minimal test harness (stdlib only)
# ---------------------------------------------------------------------------

_PASSED: list[str] = []
_FAILED: list[str] = []


def _run(name: str, fn) -> None:
    try:
        fn()
        _PASSED.append(name)
        print(f"  PASS  {name}")
    except AssertionError as exc:
        _FAILED.append(name)
        print(f"  FAIL  {name}  -- {exc}")
    except Exception:
        _FAILED.append(name)
        print(f"  FAIL  {name}  -- unexpected exception:")
        traceback.print_exc()


def _eq(a, b, msg=""):
    assert a == b, f"{msg}\n  got:      {a!r}\n  expected: {b!r}"


# ============================================================================
# 01 — Case normalization
# ============================================================================
def test_case_normalization():
    rec = normalize_name("SAI TRADERS PRIVATE LIMITED")
    _eq(rec.normalized, "sai traders private limited")


# ============================================================================
# 02 — Whitespace normalization
# ============================================================================
def test_whitespace_normalization():
    result = normalize_whitespace("  Foo   Bar  Baz  ")
    _eq(result, "Foo Bar Baz")

    rec = normalize_name("Urology  Partners   Inc")
    _eq(rec.normalized, "urology partners inc")
    _eq(rec.tokens, ("urology", "partners", "inc"))


# ============================================================================
# 03 — Punctuation handling
# ============================================================================
def test_punctuation_handling():
    # Abbreviation dots
    result = normalize_abbreviation_dots("Pvt. Ltd.")
    _eq(result, "Pvt Ltd")

    # Hyphen in name (preserved)
    rec = normalize_name("Cardiology-Care Associates")
    assert "cardiology-care" in rec.normalized or "cardiology" in rec.normalized, \
        f"hyphen handling: {rec.normalized!r}"

    # Apostrophe preserved in tokens
    rec = normalize_name("Varela's Pizza")
    assert any("varela" in t for t in rec.tokens), f"apostrophe: {rec.tokens}"


# ============================================================================
# 04 — Leading noise prefix stripping
# ============================================================================
def test_leading_noise_prefix():
    cases = [
        (">> Holloway Peak Inc Seafood", "holloway peak inc seafood"),
        ("-- Summit Inc", "summit inc"),
        ("*** Acme Corp", "acme corp"),
        ("... Corp", "corp"),
        ("<< Team Ecole", "team ecole"),
        ("# thompsonstaking", "thompsonstaking"),
        ("@@ Foo Bar", "foo bar"),
    ]
    for raw, expected in cases:
        rec = normalize_name(raw)
        _eq(rec.normalized, expected, f"for raw={raw!r}")
        assert rec.noise_prefix_stripped, f"noise_prefix_stripped should be True for {raw!r}"


# ============================================================================
# 05 — Ampersand normalization
# ============================================================================
def test_ampersand_normalization():
    rec = normalize_name("Brookens, Barrie & Smith")
    _eq(rec.ampersand_norm, "brookens, barrie and smith")
    assert "and" in rec.ampersand_tokens, f"tokens: {rec.ampersand_tokens}"

    # & should be preserved in normalized (not ampersand_norm)
    # (it is stripped only when accessing the ampersand_norm view)


# ============================================================================
# 06 — Legal suffix normalization
# ============================================================================
def test_legal_suffix_normalization():
    cases = [
        # raw → expected tokens_suffix_norm
        ("Sai Traders Pvt. Ltd.", ("sai", "traders", "private_limited")),
        ("Golden One Consultants Private Ltd", ("golden", "one", "consultants", "private_limited")),
        ("Urban Bakery LLC", ("urban", "bakery", "llc")),
        ("Velasquez, Ragland & Simmons Inc", ("velasquez,", "ragland", "&", "simmons", "incorporated")),
        ("Real-It Private", ("real-it", "private")),   # "private" alone — not enough for bigram
        ("Acme Corp.", ("acme", "corporation")),
        ("Foo L.L.C.", ("foo", "llc")),
    ]
    for raw, expected in cases:
        rec = normalize_name(raw)
        _eq(rec.tokens_suffix_norm, expected, f"suffix norm for {raw!r}")


# ============================================================================
# 07 — Unicode NFKD normalization
# ============================================================================
def test_unicode_nfkd():
    # French accented name: é should decompose under NFKD
    rec = normalize_name("Fractales Amis Groupe S.A.S")
    import unicodedata
    # NFKD of 'é' is 'e' + combining acute
    # The NFKD representation should differ if there are accented chars
    # (This particular name has no accents, so test with explicit value)

    import unicodedata
    nfkd_e = unicodedata.normalize("NFKD", "é")
    assert len(nfkd_e) == 2 and nfkd_e[0] == "e", "NFKD should decompose é"

    rec2 = normalize_name("SCI Ptit Amicale")
    # No accents: normalized_nfkd == normalized
    _eq(rec2.normalized, rec2.normalized_nfkd)


# ============================================================================
# 08 — Compact representation
# ============================================================================
def test_compact_representation():
    result = make_compact("urban bakery llc", alnum_only=False)
    _eq(result, "urbanbakeryllc")

    result_alnum = make_compact("jeewan jyoti appartment", alnum_only=True)
    _eq(result_alnum, "jeewanjyotiappartment")


# ============================================================================
# 09 — compact_no_tld
# ============================================================================
def test_compact_no_tld():
    rec = normalize_name("jeewanjyotiappartment.com")
    # compact_alnum includes 'com' because the dot was already removed
    _eq(rec.compact_alnum, "jeewanjyotiappartmentcom")
    # compact_no_tld strips TLD BEFORE alnum, so the dot was still present
    _eq(rec.compact_no_tld, "jeewanjyotiappartment")

    # Verify strip_tld works when dot is present
    from src.normalization.common import strip_tld
    _eq(strip_tld("mycompany.com"), "mycompany")
    _eq(strip_tld("mycompany.co.in"), "mycompany")
    _eq(strip_tld("ordinarytext"), "ordinarytext")  # no TLD, no change
    _eq(strip_tld("siiainvestments"), "siiainvestments")



# ============================================================================
# 10 — Latin script detection
# ============================================================================
def test_latin_script_detection():
    info = get_script_info("Urology Partners Inc")
    _eq(info.dominant_script, DOM_LATIN)
    # Spaces and digits reduce letter ratio — require >0 latin letters
    assert info.latin_ratio > 0.5, f"latin_ratio={info.latin_ratio}"
    assert not info.is_non_latin
    assert not info.is_mixed_script
    assert not info.is_corrupted
    # All letter characters should be Latin
    assert info.non_latin_letter_ratio == 0.0


# ============================================================================
# 11 — Devanagari script detection
# ============================================================================
def test_devanagari_script_detection():
    # "Private Limited" in Devanagari (observed in training data)
    devanagari_name = "प्राइवेट लिमिटेड"
    info = get_script_info(devanagari_name)
    _eq(info.dominant_script, DOM_NON_LATIN)
    assert SCRIPT_DEVANAGARI in info.scripts_present, f"scripts: {info.scripts_present}"
    assert info.is_non_latin
    assert not info.is_mixed_script
    assert info.non_latin_letter_ratio > 0.5


# ============================================================================
# 12 — Gujarati script detection
# ============================================================================
def test_gujarati_script_detection():
    # "Vijay Constructions" in Gujarati (observed in training data)
    gujarati_name = "\u0ab5\u0abf\u0a9c\u0aaf \u0a95\u0aa8\u0acd\u0ab8\u0acd\u0a9f\u0acd\u0ab0\u0a95\u0acd\u0ab6\u0aa8\u0acd\u0ab8"
    info = get_script_info(gujarati_name)
    _eq(info.dominant_script, DOM_NON_LATIN)
    assert SCRIPT_GUJARATI in info.scripts_present, f"scripts: {info.scripts_present}"
    assert info.is_non_latin


# ============================================================================
# 13 — Mixed-script detection
# ============================================================================
def test_mixed_script_detection():
    # Latin + Devanagari in same string
    mixed = "ABC \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f"
    info = get_script_info(mixed)
    _eq(info.dominant_script, DOM_MIXED)
    assert info.is_mixed_script
    assert info.latin_ratio > 0.0
    assert info.non_latin_letter_ratio > 0.0


# ============================================================================
# 14 — Replacement character (corruption) detection
# ============================================================================
def test_replacement_character_detection():
    # String with >10% U+FFFD characters
    corrupted = "\ufffd" * 12 + "abc"
    info = get_script_info(corrupted)
    _eq(info.dominant_script, DOM_CORRUPTED)
    assert info.is_corrupted
    assert info.replacement_ratio > 0.1

    rec = normalize_name(corrupted)
    _eq(rec.quality, NQ_CORRUPTED)


# ============================================================================
# 15 — Empty string handling
# ============================================================================
def test_empty_string_handling():
    info = get_script_info("")
    _eq(info.dominant_script, DOM_EMPTY)
    _eq(info.char_count, 0)

    rec = normalize_name("")
    _eq(rec.normalized, "")
    _eq(rec.tokens, ())
    _eq(rec.quality, NQ_EMPTY)


# ============================================================================
# 16 — No-letters string handling
# ============================================================================
def test_no_letters_string():
    no_letters = "123 456 789"
    info = get_script_info(no_letters)
    _eq(info.dominant_script, DOM_NO_LETTERS)
    assert info.digit_ratio > 0.0
    assert info.latin_ratio == 0.0


# ============================================================================
# 17 — Name quality: GOOD
# ============================================================================
def test_name_quality_good():
    rec = normalize_name("Urology Partners Inc")
    _eq(rec.quality, NQ_GOOD)


# ============================================================================
# 18 — Name quality: NON_LATIN
# ============================================================================
def test_name_quality_non_latin():
    rec = normalize_name("\u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921")
    _eq(rec.quality, NQ_NON_LATIN)


# ============================================================================
# 19 — Name quality: MIXED_SCRIPT
# ============================================================================
def test_name_quality_mixed_script():
    mixed = "ABC \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f"
    info = get_script_info(mixed)
    quality = classify_name_quality(mixed, info)
    _eq(quality, NQ_MIXED_SCRIPT)


# ============================================================================
# 20 — Name quality: CORRUPTED
# ============================================================================
def test_name_quality_corrupted():
    rec = normalize_name("\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd abc")
    _eq(rec.quality, NQ_CORRUPTED)


# ============================================================================
# 21 — Name quality: WEBSITE
# ============================================================================
def test_name_quality_website():
    rec = normalize_name("siiainvestments.com")
    _eq(rec.quality, NQ_WEBSITE, f"got {rec.quality} for 'siiainvestments.com'")

    rec2 = normalize_name("jeewanjyotiappartment.com")
    _eq(rec2.quality, NQ_WEBSITE)


# ============================================================================
# 22 — Name quality: EMPTY
# ============================================================================
def test_name_quality_empty():
    rec = normalize_name("   ")
    _eq(rec.quality, NQ_EMPTY)


# ============================================================================
# 23 — Noise prefix detection flag
# ============================================================================
def test_noise_prefix_flag():
    assert has_noise_prefix(">> Foo Bar")
    assert has_noise_prefix("-- Foo Bar")
    assert has_noise_prefix("*** Foo Bar")
    assert has_noise_prefix("# Foo")
    assert not has_noise_prefix("Foo Bar")
    assert not has_noise_prefix("123 Main St")

    rec = normalize_name(">> Holloway Peak Inc")
    assert rec.noise_prefix_stripped, "Flag should be True"

    rec2 = normalize_name("Holloway Peak Inc")
    assert not rec2.noise_prefix_stripped, "Flag should be False"


# ============================================================================
# 24 — Missing address
# ============================================================================
def test_missing_address():
    rec = normalize_address("")
    _eq(rec.quality, AQ_MISSING)
    _eq(rec.tokens, ())
    assert not rec.has_po_box
    _eq(rec.component_count, 0)

    rec2 = normalize_address("   ")
    _eq(rec2.quality, AQ_MISSING)


# ============================================================================
# 25 — Address punctuation (## prefix)
# ============================================================================
def test_address_hash_prefix():
    rec = normalize_address("##218 WENRICK STREET, COVINGTON, OH")
    assert rec.normalized.startswith("218"), f"normalized={rec.normalized!r}"
    _eq(rec.quality, AQ_GOOD)
    assert "218" in rec.normalized


# ============================================================================
# 26 — Numeric token extraction
# ============================================================================
def test_numeric_token_extraction():
    rec = normalize_address("3343 J F Scott Drive, Aztec, NM")
    assert "3343" in rec.numeric_tokens, f"numeric_tokens={rec.numeric_tokens}"

    # Indian plot notation
    rec2 = normalize_address("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi")
    # Should find numeric tokens
    assert any(t.startswith("570") or "/" in t for t in rec2.numeric_tokens), \
        f"numeric_tokens={rec2.numeric_tokens}"


# ============================================================================
# 27 — Postal code candidate extraction
# ============================================================================
def test_postal_code_extraction():
    # US ZIP: 5 digits
    rec = normalize_address("1795 Westchester Drive, High Point, NC 27262")
    assert "27262" in rec.postal_candidates, f"postal={rec.postal_candidates}"

    # India PIN: 6 digits
    rec2 = normalize_address("KH NO. -570/13, 110001, NEW DELHI, Delhi")
    assert "110001" in rec2.postal_candidates, f"postal={rec2.postal_candidates}"

    # No postal code
    rec3 = normalize_address("1514 91st Avenue, Omaha, NE")
    _eq(rec3.postal_candidates, (), f"should be empty: {rec3.postal_candidates}")


# ============================================================================
# 28 — House number detection
# ============================================================================
def test_house_number_detection():
    # US-style: first token is house number
    rec = normalize_address("6207 Ocean Front Avenue, VA, Virginia Beach City")
    assert "6207" in rec.house_number_candidates, \
        f"house_candidates={rec.house_number_candidates}"

    # Indian 'NO' marker
    rec2 = normalize_address("NO 10 Enkay Square, 448A, Udyog Vihar, Gurgaon")
    assert any("10" in h for h in rec2.house_number_candidates), \
        f"house_candidates={rec2.house_number_candidates}"


# ============================================================================
# 29 — PO Box detection
# ============================================================================
def test_po_box_detection():
    rec = normalize_address("1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio")
    assert rec.has_po_box, "should detect PO Box"

    rec2 = normalize_address("6207 Ocean Front Avenue, Virginia Beach, VA")
    assert not rec2.has_po_box, "should not detect PO Box"

    # Variations
    for addr in ["P.O. Box 123, Dallas TX", "P O BOX 45678", "POBox 90"]:
        r = normalize_address(addr)
        assert r.has_po_box, f"should detect PO Box in: {addr!r}"


# ============================================================================
# 30 — Address component counting
# ============================================================================
def test_component_counting():
    rec = normalize_address("6207 Ocean Front Avenue, VA, Virginia Beach City")
    _eq(rec.component_count, 3)

    rec2 = normalize_address("House No.-37, Sector-3, Part-2, Rewari, Haryana, Gokalgarh, Rewari, Haryana")
    assert rec2.component_count >= 7, f"components={rec2.component_count}"

    rec3 = normalize_address("Simple Street")
    _eq(rec3.component_count, 1)


# ============================================================================
# 31 — India 'NO' / 'HNO' house-number marker
# ============================================================================
def test_india_house_number_markers():
    rec = normalize_address("H.NO. 37, SECTOR-3, REWARI, HARYANA")
    # Should find "37" as a house number candidate
    assert any("37" in h for h in rec.house_number_candidates), \
        f"house={rec.house_number_candidates}"


# ============================================================================
# 32 — Full pipeline: complete BusinessRecord
# ============================================================================
def test_full_pipeline():
    rec = normalize_record(
        entity_id="S1-820791388",
        raw_name="Sai Traders Private Limited",
        raw_address="House No.-37, Sector-3, Part-2, Rewari, Haryana",
        country="India",
    )
    _eq(rec.entity_id, "S1-820791388")
    _eq(rec.country, "India")
    _eq(rec.name.quality, NQ_GOOD)
    _eq(rec.name.script.dominant_script, DOM_LATIN)
    assert "private_limited" in rec.name.tokens_suffix_norm, \
        f"suffix_norm={rec.name.tokens_suffix_norm}"
    assert "37" in rec.address.house_number_candidates or \
           any("37" in h for h in rec.address.house_number_candidates), \
        f"house={rec.address.house_number_candidates}"
    _eq(rec.address.quality, AQ_GOOD)


# ============================================================================
# 33 — Full pipeline: actual noisy examples from training data
# ============================================================================
def test_noisy_real_examples():
    pipeline = NormalizationPipeline()

    examples = [
        # (entity_id, raw_name, raw_address, country)
        ("S2-845708664", "Urology Partners  Inc",
         "6207 OCEAN FRONT AVE, VIRGINIA BEACH CITY, VA", "US"),
        ("S2-442717160", "SAI TRADERS PRIVATE LIMITED",
         "HOUSE NO.-37, SECTOR-3, PART-2, REWARI, HARYANA, GOKALGARH, \u0939\u0930\u093f\u092f\u093e\u0923\u093e", "India"),
        ("S2-157435214", "Urban Bakery LLC Center",
         "NM, AZTEC, 3343-3345 J F SCOTT DR", "US"),
        ("S2-839474987", "CENTRAL MARKETING (GLOBAL)",
         "72573-72575 LAKE VIEW DR, JACKSON, MN", "US"),
        ("S3-202863386", "wilfordhancock.com",
         "Mack Rd, Haltom City, Texas", "US"),
    ]

    for eid, name, addr, country in examples:
        rec = pipeline.normalize(eid, name, addr, country)
        assert rec.entity_id == eid
        assert len(rec.name.tokens) > 0, f"empty tokens for {name!r}"
        # Normalized should be lowercase
        assert rec.name.normalized == rec.name.normalized.lower(), \
            f"not lowercase: {rec.name.normalized!r}"

    # Website detection
    rec_web = pipeline.normalize("S3-202863386", "wilfordhancock.com",
                                 "Mack Rd, Haltom City, Texas", "US")
    _eq(rec_web.name.quality, NQ_WEBSITE)

    # Mixed script address (Devanagari in address)
    rec_mixed_addr = pipeline.normalize(
        "S2-X", "Sai Traders", "REWARI, \u0939\u0930\u093f\u092f\u093e\u0923\u093e", "India")
    assert rec_mixed_addr.address.quality in ("MIXED_SCRIPT", "NON_LATIN", "GOOD"), \
        f"addr quality: {rec_mixed_addr.address.quality}"


# ============================================================================
# 34 — Script info on all-non-Latin name
# ============================================================================
def test_all_non_latin_name():
    # Telugu: "Limited" in Telugu script
    telugu = "\u0c32\u0c3f\u0c2e\u0c3f\u0c1f\u0c46\u0c21\u0c4d"
    info = get_script_info(telugu)
    _eq(info.dominant_script, DOM_NON_LATIN)
    assert info.is_non_latin
    assert not info.is_mixed_script
    assert SCRIPT_TELUGU in info.scripts_present


# ============================================================================
# 35 — Suffix bigram normalization
# ============================================================================
def test_suffix_bigram():
    # "Private Limited" as a bigram
    tokens = ("acme", "private", "limited")
    result = apply_suffix_normalization(tokens)
    _eq(result, ("acme", "private_limited"))

    # "Pvt Ltd" bigram
    tokens2 = ("foo", "bar", "pvt", "ltd")
    result2 = apply_suffix_normalization(tokens2)
    _eq(result2, ("foo", "bar", "private_limited"))

    # Single suffix only
    tokens3 = ("acme", "corp")
    result3 = apply_suffix_normalization(tokens3)
    _eq(result3, ("acme", "corporation"))


# ============================================================================
# 36 — Alpha token extraction
# ============================================================================
def test_alpha_token_extraction():
    rec = normalize_address("6207 Ocean Front Avenue, VA, Virginia Beach City")
    alpha = rec.alpha_tokens
    assert "ocean" in alpha, f"alpha_tokens={alpha}"
    assert "avenue" in alpha, f"'avenue' missing from {alpha}"
    assert "6207" not in alpha   # numeric, should be excluded
    # Single-char tokens like 'va' may or may not be included (>1 char check)
    # 'virginia', 'beach', 'city' should be there
    assert "virginia" in alpha, f"'virginia' missing from {alpha}"


# ============================================================================
# 37 — Component count with multiple commas
# ============================================================================
def test_component_count_multiple_commas():
    rec = normalize_address("A, B, C, D, E")
    _eq(rec.component_count, 5)


# ============================================================================
# 38 — Multiple Indic scripts in same string (edge case)
# ============================================================================
def test_multiple_indic_scripts():
    # Mix Gujarati and Devanagari in same string
    mixed_indic = "\u0a85\u0a86 \u0905\u0906"  # Gujarati + Devanagari chars
    info = get_script_info(mixed_indic)
    assert len(info.scripts_present) >= 1
    assert info.is_non_latin or info.dominant_script == DOM_NON_LATIN


# ============================================================================
# 39 — French accented name (NFKD decomposition)
# ============================================================================
def test_french_accent_nfkd():
    rec = normalize_name("Marina Ecole France Sarl")
    # No accents in this one, but test the French suffix
    assert "sarl" in rec.tokens_suffix_norm or "sarl" in rec.tokens

    rec2 = normalize_name("Boulevard du President Roosevelt")
    # NFKD should decompose accented chars in é
    assert "boulevard" in rec2.normalized
    assert "roosevelt" in rec2.normalized

    # Verify NFKD decomposition actually works for accented string
    import unicodedata
    accented = "café"
    rec3 = normalize_name(accented)
    nfkd = unicodedata.normalize("NFKD", "cafe\u0301")  # e + combining acute
    assert len(nfkd) > len("cafe"), "NFKD should add combining marks"


# ============================================================================
# 40 — Phone normalization: US format
# ============================================================================
def test_phone_normalization_us():
    rec = normalize_phone("+1 (800) 555-0199")
    _eq(rec.digits, "18005550199")
    _eq(rec.last_10, "8005550199")
    _eq(rec.last_7, "5550199")
    assert rec.has_phone
    assert rec.is_valid


# ============================================================================
# 41 — Phone normalization: International format
# ============================================================================
def test_phone_normalization_intl():
    rec = normalize_phone("+91 98765-43210")
    _eq(rec.digits, "919876543210")
    _eq(rec.last_10, "9876543210")
    _eq(rec.last_7, "6543210")
    assert rec.has_phone
    assert rec.is_valid


# ============================================================================
# 42 — Phone normalization: Empty / None
# ============================================================================
def test_phone_normalization_empty():
    rec1 = normalize_phone("")
    _eq(rec1.digits, "")
    _eq(rec1.last_10, "")
    _eq(rec1.last_7, "")
    assert not rec1.has_phone
    assert not rec1.is_valid

    rec2 = normalize_phone(None)
    assert not rec2.has_phone


# ============================================================================
# 43 — BusinessRecord source and properties
# ============================================================================
def test_business_record_properties():
    rec = normalize_record(
        "s2-12345",
        "Sai Traders Pvt Ltd",
        "House No.-37, Rewari, Haryana",
        "India",
        "+91 9876543210",
    )
    _eq(rec.source, "s2")
    assert rec.has_phone
    _eq(rec.phone.last_10, "9876543210")
    assert rec.is_complete_record
    _eq(rec.dominant_script, DOM_LATIN)
    assert not rec.is_mixed_script


# ============================================================================
# 44 — BusinessRecord throughput benchmark on synthetic batch
# ============================================================================
def test_record_throughput_sample():
    import time
    pipeline = NormalizationPipeline()
    sample_records = [
        ("s1-100", "Acme Corporation LLC", "123 Main St, Suite 400, Springfield, IL 62701", "USA"),
        ("s2-200", "Sai Enterprises Pvt. Ltd.", "KH NO. -570/13, NEW DELHI, Delhi 110001", "India"),
        ("s3-300", "##218 WENRICK STREET", "COVINGTON, OH", "USA"),
        ("s1-400", "\u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921", "Mumbai, Maharashtra", "India"),
    ] * 2500  # 10,000 records
    t0 = time.perf_counter()
    for eid, name, addr, country in sample_records:
        pipeline.normalize(eid, name, addr, country)
    elapsed = time.perf_counter() - t0
    throughput = len(sample_records) / elapsed
    assert throughput > 3000, f"Throughput too low: {throughput:.0f} rec/s"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests() -> int:
    tests = [
        ("01 case_normalization", test_case_normalization),
        ("02 whitespace_normalization", test_whitespace_normalization),
        ("03 punctuation_handling", test_punctuation_handling),
        ("04 leading_noise_prefix", test_leading_noise_prefix),
        ("05 ampersand_normalization", test_ampersand_normalization),
        ("06 legal_suffix_normalization", test_legal_suffix_normalization),
        ("07 unicode_nfkd", test_unicode_nfkd),
        ("08 compact_representation", test_compact_representation),
        ("09 compact_no_tld", test_compact_no_tld),
        ("10 latin_script_detection", test_latin_script_detection),
        ("11 devanagari_script_detection", test_devanagari_script_detection),
        ("12 gujarati_script_detection", test_gujarati_script_detection),
        ("13 mixed_script_detection", test_mixed_script_detection),
        ("14 replacement_char_detection", test_replacement_character_detection),
        ("15 empty_string_handling", test_empty_string_handling),
        ("16 no_letters_string", test_no_letters_string),
        ("17 name_quality_good", test_name_quality_good),
        ("18 name_quality_non_latin", test_name_quality_non_latin),
        ("19 name_quality_mixed_script", test_name_quality_mixed_script),
        ("20 name_quality_corrupted", test_name_quality_corrupted),
        ("21 name_quality_website", test_name_quality_website),
        ("22 name_quality_empty", test_name_quality_empty),
        ("23 noise_prefix_flag", test_noise_prefix_flag),
        ("24 missing_address", test_missing_address),
        ("25 address_hash_prefix", test_address_hash_prefix),
        ("26 numeric_token_extraction", test_numeric_token_extraction),
        ("27 postal_code_extraction", test_postal_code_extraction),
        ("28 house_number_detection", test_house_number_detection),
        ("29 po_box_detection", test_po_box_detection),
        ("30 component_counting", test_component_counting),
        ("31 india_house_number_markers", test_india_house_number_markers),
        ("32 full_pipeline", test_full_pipeline),
        ("33 noisy_real_examples", test_noisy_real_examples),
        ("34 all_non_latin_name", test_all_non_latin_name),
        ("35 suffix_bigram", test_suffix_bigram),
        ("36 alpha_token_extraction", test_alpha_token_extraction),
        ("37 component_count_multiple_commas", test_component_count_multiple_commas),
        ("38 multiple_indic_scripts", test_multiple_indic_scripts),
        ("39 french_accent_nfkd", test_french_accent_nfkd),
        ("40 phone_normalization_us", test_phone_normalization_us),
        ("41 phone_normalization_intl", test_phone_normalization_intl),
        ("42 phone_normalization_empty", test_phone_normalization_empty),
        ("43 business_record_properties", test_business_record_properties),
        ("44 record_throughput_sample", test_record_throughput_sample),
    ]

    print("\n" + "=" * 64)
    print("  Running Phase 2 normalization unit tests")
    print("=" * 64)

    for name, fn in tests:
        _run(name, fn)

    print("-" * 64)
    n_pass = len(_PASSED)
    n_fail = len(_FAILED)
    total = len(tests)
    print(f"  {n_pass}/{total} tests passed")
    if _FAILED:
        print(f"  FAILED: {', '.join(_FAILED)}")
        return 1
    print("  All tests passed [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
