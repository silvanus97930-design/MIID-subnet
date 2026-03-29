# MIID/miner/kav_helpers.py — KAV (identity variation) helpers: DOB, address heuristics, debug.
# Validator scoring mirrors MIID.validator.reward where imported.

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

# Lazy imports for validator-side checks (heavy geonames on first use)
_LOOKS_LIKE = None
_VALIDATE_REGION = None


def _validator_imports():
    global _LOOKS_LIKE, _VALIDATE_REGION
    if _LOOKS_LIKE is None:
        from MIID.validator.reward import looks_like_address, validate_address_region

        _LOOKS_LIKE = looks_like_address
        _VALIDATE_REGION = validate_address_region
    return _LOOKS_LIKE, _VALIDATE_REGION


# Aliases aligned with miner.COUNTRY_ALIASES + common variants for parsing seed text
COUNTRY_ALIAS_NORMALIZE: Dict[str, str] = {
    "russian federation": "russia",
    "the netherlands": "netherlands",
    "holland": "netherlands",
    "burma": "myanmar",
    "usa": "united states",
    "u.s.a.": "united states",
    "u.s.": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "great britain": "united kingdom",
    "uae": "united arab emirates",
    "venezuela, bolivarian republic of": "venezuela",
    "bolivia, plurinational state of": "bolivia",
    "tanzania, united republic of": "tanzania",
    "lao people's democratic republic": "laos",
    "côte d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "cote d ivoire": "ivory coast",
    "the gambia": "gambia",
    "korea, south": "south korea",
    "korea, north": "north korea",
}


def compact_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


@dataclass
class SeedRegion:
    """Structured seed address for KAV region-aligned generation."""

    raw: str
    country_name: str  # canonical display for last segment (Geonames-style)
    country_code: Optional[str]
    city_hint: Optional[str]
    parts: List[str] = field(default_factory=list)
    single_token_seed: bool = False


def normalize_country_token(
    token: str,
    country_to_code: Dict[str, str],
    code_to_name: Dict[str, str],
    compact_to_code: Dict[str, str],
) -> Tuple[Optional[str], str]:
    """Return (country_code, canonical_country_name) for a free-text country token."""
    t = (token or "").strip().lower()
    if not t:
        return None, ""
    t = COUNTRY_ALIAS_NORMALIZE.get(t, t)
    if t in country_to_code:
        code = country_to_code[t]
        return code, code_to_name.get(code, token.strip())
    ct = compact_text(t)
    if ct in compact_to_code:
        code = compact_to_code[ct]
        return code, code_to_name.get(code, token.strip())
    best_code = None
    best_score = 0.0
    for name, code in country_to_code.items():
        if len(name) < 3:
            continue
        ratio = SequenceMatcher(None, t, name).ratio()
        cr = SequenceMatcher(None, ct, compact_text(name)).ratio()
        sc = max(ratio, cr)
        if sc > best_score:
            best_score = sc
            best_code = code
    if best_code and best_score >= 0.74:
        return best_code, code_to_name.get(best_code, token.strip())
    return None, token.strip()


def extract_seed_region(
    seed_address: str,
    country_to_code: Dict[str, str],
    code_to_name: Dict[str, str],
    compact_to_code: Dict[str, str],
) -> SeedRegion:
    """Parse seed into country + optional city; normalize country aliases."""
    raw = str(seed_address or "").strip()
    if not raw:
        return SeedRegion(
            raw="",
            country_name="United States",
            country_code=country_to_code.get("united states"),
            city_hint="New York",
            parts=[],
            single_token_seed=False,
        )

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) == 1:
        single = parts[0]
        low = single.lower()
        low = COUNTRY_ALIAS_NORMALIZE.get(low, low)
        # country-only token
        cc = country_to_code.get(low) or compact_to_code.get(compact_text(low))
        if cc:
            return SeedRegion(
                raw=raw,
                country_name=code_to_name.get(cc, single),
                country_code=cc,
                city_hint=None,
                parts=parts,
                single_token_seed=True,
            )
        # city-only / region token — still use as city hint; infer country via geonames city index if possible elsewhere
        return SeedRegion(
            raw=raw,
            country_name="United States",
            country_code=country_to_code.get("united states"),
            city_hint=single,
            parts=parts,
            single_token_seed=True,
        )

    # Multi-part: last segment = country, first = city/street hint
    last = parts[-1]
    city_hint = parts[0] if not any(ch.isdigit() for ch in parts[0]) else None
    cc, cname = normalize_country_token(last, country_to_code, code_to_name, compact_to_code)
    if not cc:
        cc = country_to_code.get(last.lower()) or compact_to_code.get(compact_text(last))
        if cc:
            cname = code_to_name.get(cc, last)
    return SeedRegion(
        raw=raw,
        country_name=cname or last,
        country_code=cc,
        city_hint=city_hint,
        parts=parts,
        single_token_seed=False,
    )


def _parse_seed_dob_to_datetime(raw: str) -> Optional[datetime]:
    """Parse validator-style or common date strings; return None if unusable."""
    s = (raw or "").strip()
    if not s:
        return None
    for fmt, maxlen in (
        ("%Y-%m-%d", 10),
        ("%Y/%m/%d", 10),
        ("%d/%m/%Y", 10),
        ("%m/%d/%Y", 10),
    ):
        try:
            return datetime.strptime(s[:maxlen], fmt)
        except Exception:
            continue
    # Year-month only (validator accepts %Y-%m for one bucket)
    if len(s) >= 7 and s[4] == "-":
        try:
            return datetime.strptime(s[:7] + "-15", "%Y-%m-%d")
        except Exception:
            pass
    return None


def build_dob_variations_deterministic(seed_dob: str, target_count: int) -> List[str]:
    """
    Deterministic DOB list: no LLM. Covers ±1, ±3, ±30, ±90, ±365, year-month, then unique offsets.
    """
    if target_count <= 0:
        return []
    raw = (seed_dob or "").strip()
    base = _parse_seed_dob_to_datetime(raw)
    if base is None:
        base = datetime(1990, 1, 15)

    ordered_ops: List[Tuple[str, Any]] = [
        ("day", timedelta(days=1)),
        ("day", timedelta(days=-1)),
        ("day", timedelta(days=3)),
        ("day", timedelta(days=-3)),
        ("day", timedelta(days=30)),
        ("day", timedelta(days=-30)),
        ("day", timedelta(days=90)),
        ("day", timedelta(days=-90)),
        ("day", timedelta(days=365)),
        ("day", timedelta(days=-365)),
        ("month", None),
    ]

    out: List[str] = []
    seen: set = set()

    def push_ds(ds: datetime, as_month: bool = False) -> None:
        nonlocal out
        s = ds.strftime("%Y-%m") if as_month else ds.strftime("%Y-%m-%d")
        if s not in seen:
            seen.add(s)
            out.append(s)

    for kind, delta in ordered_ops:
        if kind == "month":
            push_ds(base, as_month=True)
        elif isinstance(delta, timedelta):
            push_ds(base + delta)

    extra = [2, -7, 14, -21, 45, -120, 180, -240, 330, -360, 500, -500]
    for off in extra:
        if len(out) >= target_count:
            break
        push_ds(base + timedelta(days=off))

    step = 0
    while len(out) < target_count:
        step += 1
        push_ds(base + timedelta(days=step * 13))

    return out[:target_count]


def make_structured_address_line(
    idx: int,
    city: str,
    country: str,
    street_roots: List[str],
    street_suffixes: List[str],
    district_words: List[str],
) -> str:
    """Long comma-separated line tuned to pass looks_like_address heuristics."""
    street_root = street_roots[idx % len(street_roots)]
    street_suffix = street_suffixes[idx % len(street_suffixes)]
    district = district_words[idx % len(district_words)]
    house_num = 10 + (idx * 7 % 980)
    area_num = 1 + (idx * 3 % 17)
    postal = 10000 + (idx * 113 % 89999)
    return (
        f"{house_num} {street_root} {street_suffix}, "
        f"{district} {area_num}, "
        f"{city} {postal}, "
        f"{country}"
    )


def score_address_candidate(addr: str, seed_address: str) -> Tuple[float, bool, bool]:
    """
    Returns (score, looks_ok, region_ok). Higher is better.
    Uses validator functions when available.
    """
    looks_fn, region_fn = _validator_imports()
    lk = bool(looks_fn(str(addr)))
    rk = bool(region_fn(str(addr), str(seed_address)))
    score = (1000.0 if rk else 0.0) + (100.0 if lk else 0.0) + min(len(addr), 400) * 0.01
    return score, lk, rk


def generate_address_candidate_pool(
    seed_address: str,
    pool_size: int,
    country_display: str,
    pick_city: Callable[[int], str],
    street_roots: List[str],
    street_suffixes: List[str],
    district_words: List[str],
) -> List[str]:
    """Build diverse structured candidates; selection filters with validator."""
    out: List[str] = []
    seen = set()
    for i in range(pool_size * 3):
        if len(out) >= pool_size:
            break
        city = pick_city(i)
        line = make_structured_address_line(i, city, country_display, street_roots, street_suffixes, district_words)
        key = line.lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out


def select_best_addresses(
    candidates: List[str],
    seed_address: str,
    target_count: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Pick top addresses by validator-aligned score; dedupe."""
    scored: List[Tuple[float, str, bool, bool]] = []
    for c in candidates:
        sc, lk, rk = score_address_candidate(c, seed_address)
        scored.append((sc, c, lk, rk))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen: List[str] = []
    meta: List[Dict[str, Any]] = []
    seen = set()
    for sc, c, lk, rk in scored:
        if len(chosen) >= target_count:
            break
        k = c.lower().strip()
        if k in seen:
            continue
        seen.add(k)
        chosen.append(c)
        meta.append({"score": sc, "looks_like": lk, "region_match": rk})
    return chosen, meta


def debug_score_identity_output(
    seed_name: str,
    seed_dob: str,
    seed_address: str,
    rows: List[List[str]],
    target_count: int,
) -> Dict[str, Any]:
    """Lightweight local metrics for logging / JSON (not validator ground truth)."""
    names = [r[0] for r in rows if len(r) > 0]
    dobs = [r[1] for r in rows if len(r) > 1]
    addrs = [r[2] for r in rows if len(r) > 2]
    looks_fn, region_fn = _validator_imports()

    def _dup_risk(xs: List[str]) -> float:
        if not xs:
            return 1.0
        u = len({x.lower().strip() for x in xs if x})
        return 1.0 - (u / max(len(xs), 1))

    lk_addr = [bool(looks_fn(a)) for a in addrs if a]
    rk_addr = [bool(region_fn(a, seed_address)) for a in addrs if a]

    return {
        "row_count_ok": len(rows) >= target_count,
        "missing_row_risk": max(0.0, 1.0 - (len(rows) / max(target_count, 1))),
        "empty_name_fields": sum(1 for n in names if not str(n).strip()),
        "empty_dob_fields": sum(1 for d in dobs if not str(d).strip()),
        "empty_address_fields": sum(1 for a in addrs if not str(a).strip()),
        "duplicate_name_risk": _dup_risk([str(n) for n in names]),
        "looks_like_address_rate": (sum(1 for x in lk_addr if x) / max(len(lk_addr), 1)) if lk_addr else 0.0,
        "region_match_rate": (sum(1 for x in rk_addr if x) / max(len(rk_addr), 1)) if rk_addr else 0.0,
        "target_count": target_count,
        "actual_count": len(rows),
    }


def debug_score_run(identities_debug: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "identities": identities_debug,
        "worst_by_missing": sorted(
            identities_debug,
            key=lambda d: d.get("metrics", {}).get("missing_row_risk", 0),
            reverse=True,
        )[:5],
    }


def write_kav_debug_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
