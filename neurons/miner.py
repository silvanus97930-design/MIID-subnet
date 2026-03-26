# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): YANEZ - MIID Team
# Copyright © 2025 YANEZ

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Name Variation Miner Module

This module implements a Bittensor miner that generates alternative spellings for names
using a local LLM (via Ollama). 
######### Ollama should be installed and running on the machine. ########
The miner receives requests from validators containing
a list of names and a query template, processes each name through the LLM, extracts
the variations from the LLM's response, and returns them to the validator.

The miner follows these steps:
1. Receive a request with names and a query template
2. For each name, query the LLM to generate variations
3. Process the LLM responses to extract clean variations
4. Return the variations to the validator

The processing logic handles different response formats from LLMs, including:
- Comma-separated lists
- Line-separated lists
- Space-separated lists with numbering

For debugging and analysis, the miner also saves:
- Raw LLM responses
- Processed variations in JSON format
- A pandas DataFrame with the variations

Each mining run is saved with a unique timestamp identifier to distinguish between
different runs and facilitate analysis of results over time.
"""

import time
import typing
import io
import subprocess
import random
import re
import threading
import unicodedata
import json
import hashlib
from difflib import SequenceMatcher
import bittensor as bt
import ollama
import pandas as pd
import os
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image

try:
    import geonamescache
except ImportError:
    geonamescache = None

# Bittensor Miner Template:
from MIID.protocol import IdentitySynapse, S3Submission

# import base miner class which takes care of most of the boilerplate
from MIID.base.miner import BaseMinerNeuron

from bittensor.core.errors import NotVerifiedException

# Phase 4 imports
from MIID.miner.image_generator import (
    decode_base_image,
    generate_variations,
    validate_variation,
    prewarm_flux_pipeline,
)
from MIID.miner.drand_encrypt import encrypt_image_for_drand, is_timelock_available
from MIID.miner.s3_upload import upload_to_s3


class Miner(BaseMinerNeuron):
    """
    Name Variation Miner Neuron
    
    This miner receives requests from validators to generate alternative spellings for names,
    and responds with variations generated using a local LLM (via Ollama).
    
    The miner handles the following tasks:
    - Processing incoming requests for name variations
    - Querying a local LLM to generate variations
    - Extracting and cleaning variations from LLM responses
    - Returning the processed variations to the validator
    - Saving intermediate results for debugging and analysis
    
    Each mining run is saved with a unique timestamp identifier to distinguish between
    different runs and facilitate analysis of results over time.
    
    Configuration:
    - model_name: The Ollama model to use (default: 'tinyllama:latest')
    - output_path: Directory for saving mining results (default: logging_dir/mining_results)
    """
    # Base whitelist of known validators
    WHITELISTED_VALIDATORS = {
        "5C4qiYkqKjqGDSvzpf6YXCcnBgM6punh8BQJRP78bqMGsn54": "RoundTable21",
        "5DUB7kNLvvx8Dj7D8tn54N1C7Xok6GodNPQE2WECCaL9Wgpr": "Yanez",
        "5GWzXSra6cBM337nuUU7YTjZQ6ewT2VakDpMj8Pw2i8v8PVs": "Yuma",
        "5HbUFHW4XVhbQvMbSy7WDjvhHb62nuYgP1XBsmmz9E2E2K6p": "OpenTensor",
        "5GQqAhLKVHRLpdTqRg1yc3xu7y47DicJykSpggE2GuDbfs54": "Rizzo",
        "5HK5tp6t2S59DywmHRWPBVJeJ86T61KjurYqeooqj8sREpeN": "Tensora",
        "5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u": "Tao.bot",
        "5GuPvuyKBJAWQbEGAkMbfRpG5qDqqhML8uDVSWoFjqcKKvDU": "Testnet_omar",
        "5CnkkjPdfsA6jJDHv2U6QuiKiivDuvQpECC13ffdmSDbkgtt": "Testnet_asem"
    }

    COUNTRY_ALIASES = {
        "usa": "united states",
        "us": "united states",
        "united states of america": "united states",
        "u.s.a.": "united states",
        "u.s.": "united states",
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "great britain": "united kingdom",
        "britain": "united kingdom",
        "uae": "united arab emirates",
        "korea, south": "south korea",
        "korea, north": "north korea",
        "cote d ivoire": "ivory coast",
        "côte d'ivoire": "ivory coast",
        "cote d'ivoire": "ivory coast",
        "the gambia": "gambia",
        "holland": "the netherlands",
        "burma": "myanmar",
        "drc": "democratic republic of the congo",
        "czech republic": "czechia",
        "russian federation": "russia",
        "iran, islamic republic of": "iran",
        "syrian arab republic": "syria",
        "viet nam": "vietnam",
        "moldova, republic of": "moldova",
        "lao people's democratic republic": "laos",
        "tanzania, united republic of": "tanzania",
        "bolivia, plurinational state of": "bolivia",
        "venezuela, bolivarian republic of": "venezuela",
        "palestine, state of": "palestinian territory",
    }

    STREET_SUFFIXES = ["Street", "Avenue", "Road", "Boulevard", "Lane", "Drive", "Way"]
    DISTRICT_WORDS = ["District", "Quarter", "Ward", "Heights", "Center", "Commons"]
    STREET_ROOTS = [
        "Oak", "Maple", "Cedar", "Pine", "River", "Hill", "Market", "Liberty",
        "Central", "Park", "Garden", "Lake", "Harbor", "Sunrise", "West", "East",
    ]

    NON_NAME_TOKENS = {
        "street", "st", "avenue", "ave", "road", "rd", "boulevard", "blvd",
        "lane", "drive", "district", "quarter", "ward", "center", "commons",
        "city", "state", "province", "country", "postal", "zip", "code",
        "address", "region", "unknown", "none", "null",
    }

    def _register_dynamic_validator(self, hotkey: str) -> bool:
        """Allow any currently validator-permitted hotkey from metagraph."""
        if hotkey in self.WHITELISTED_VALIDATORS:
            return True

        try:
            uid = self.metagraph.hotkeys.index(hotkey)
        except ValueError:
            return False

        try:
            has_permit = bool(self.metagraph.validator_permit[uid])
        except Exception:
            has_permit = False

        if has_permit:
            self.WHITELISTED_VALIDATORS[hotkey] = f"DynamicValidator_UID{uid}"
            bt.logging.info(
                f"Added dynamic validator allowlist entry: uid={uid}, hotkey={hotkey[:16]}..."
            )
            return True

        return False

    def _initialize_geodata(self) -> None:
        """Build country/city indexes used for reward-aligned address generation."""
        self._geo_available = False
        self._country_name_to_code: Dict[str, str] = {}
        self._country_compact_to_code: Dict[str, str] = {}
        self._country_code_to_name: Dict[str, str] = {}
        self._city_names_by_country: Dict[str, List[str]] = {}
        self._city_to_country_codes: Dict[str, List[str]] = {}

        if geonamescache is None:
            bt.logging.warning("geonamescache unavailable. Falling back to generic address generation.")
            return

        try:
            gc = geonamescache.GeonamesCache()
            countries = gc.get_countries()
            cities = gc.get_cities()

            for code, data in countries.items():
                country_name = str(data.get("name", "")).strip()
                if not country_name:
                    continue
                lower_name = country_name.lower()
                self._country_name_to_code[lower_name] = code
                self._country_compact_to_code[self._compact_text(lower_name)] = code
                self._country_code_to_name[code] = country_name

                iso = str(data.get("iso", "")).lower().strip()
                iso3 = str(data.get("iso3", "")).lower().strip()
                if iso:
                    self._country_name_to_code[iso] = code
                if iso3:
                    self._country_name_to_code[iso3] = code

            for alias, canonical in self.COUNTRY_ALIASES.items():
                canonical_code = self._country_name_to_code.get(canonical.lower())
                if canonical_code:
                    self._country_name_to_code[alias] = canonical_code
                    self._country_compact_to_code[self._compact_text(alias)] = canonical_code

            grouped_cities: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
            city_country_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for city_data in cities.values():
                code = str(city_data.get("countrycode", "")).strip()
                city_name = str(city_data.get("name", "")).strip()
                if not code or not city_name:
                    continue
                try:
                    population = int(city_data.get("population", 0) or 0)
                except Exception:
                    population = 0
                grouped_cities[code].append((population, city_name))
                city_country_counts[city_name.lower()][code] += max(population, 1)

            for code, records in grouped_cities.items():
                records.sort(reverse=True, key=lambda x: x[0])
                seen = set()
                ordered = []
                for _, city_name in records:
                    city_norm = city_name.lower().strip()
                    if not city_norm or city_norm in seen:
                        continue
                    seen.add(city_norm)
                    ordered.append(city_name)
                    if len(ordered) >= 300:
                        break
                if ordered:
                    self._city_names_by_country[code] = ordered

            for city_name, code_counts in city_country_counts.items():
                ranked_codes = sorted(code_counts.items(), key=lambda kv: kv[1], reverse=True)
                self._city_to_country_codes[city_name] = [code for code, _ in ranked_codes]

            self._geo_available = bool(self._country_name_to_code and self._city_names_by_country)
            bt.logging.info(
                f"Geodata initialized: {len(self._country_name_to_code)} country aliases, "
                f"{len(self._city_names_by_country)} countries with cities"
            )
        except Exception as e:
            bt.logging.warning(f"Failed to initialize geodata: {e}")
            self._geo_available = False

    def _extract_expected_variation_count(self, query_template: str, default: int = 10) -> int:
        """Parse expected variation count from validator query text."""
        if not query_template:
            return default

        text = str(query_template)
        candidates: List[int] = []
        patterns = [
            r"\bexact(?:ly)?\s+(\d{1,2})\s+variations?\b",
            r"\bgenerate(?:\s+exactly)?\s+(\d{1,2})\s+variations?\b",
            r"\btotal(?:\s+of)?\s+(\d{1,2})\s+variations?\b",
            r"\bexact\s+number\s+of\s+variations?\s*[:=]?\s*(\d{1,2})\b",
            r"\bvariation[_\s-]*count\s*[:=]?\s*(\d{1,2})\b",
            r'"variation_count"\s*:\s*(\d{1,2})',
            r'"count"\s*:\s*(\d{1,2})',
            r"\b(\d{1,2})\s+variations?\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                try:
                    value = int(match.group(1))
                except Exception:
                    continue
                if 4 <= value <= 20:
                    candidates.append(value)

        if not candidates:
            return default
        return Counter(candidates).most_common(1)[0][0]

    def _extract_rule_percentage(self, query_template: str, default: float = 0.0) -> float:
        """Extract the target percentage of rule-based variations from query text."""
        if not query_template:
            return default
        lowered = str(query_template).lower()
        candidates: List[int] = []
        for match in re.finditer(r"(\d{1,3})\s*%", lowered):
            try:
                pct = int(match.group(1))
            except Exception:
                continue
            if pct < 0 or pct > 100:
                continue
            window = lowered[max(0, match.start() - 48): min(len(lowered), match.end() + 96)]
            if any(token in window for token in ("rule", "transform", "variation")):
                candidates.append(pct)
        if not candidates:
            return default
        pct = Counter(candidates).most_common(1)[0][0]
        return max(0.0, min(0.9, pct / 100.0))

    def _extract_requested_transformations(self, query_template: str) -> List[str]:
        """Infer requested transformation types from natural-language or snake_case rules."""
        lowered = str(query_template or "").lower()
        if not lowered.strip():
            return []

        rules: List[str] = []

        def add_rule(name: str) -> None:
            if name not in rules:
                rules.append(name)

        if re.search(r"\binsert[_\s-]*random[_\s-]*letter\b", lowered) or ("insert" in lowered and "letter" in lowered):
            add_rule("insert_random_letter")
        if re.search(r"\bswap[_\s-]*adjacent[_\s-]*(?:syllables?|letters?)\b", lowered) or ("swap" in lowered and "adjacent" in lowered):
            add_rule("swap_adjacent_syllables")
        if re.search(r"\bdelete[_\s-]*random[_\s-]*letter\b", lowered) or ("delete" in lowered and "letter" in lowered):
            add_rule("delete_random_letter")
        if re.search(r"\bremove[_\s-]*random[_\s-]*vowel\b", lowered) or ("remove" in lowered and "vowel" in lowered):
            add_rule("remove_random_vowel")
        if re.search(r"\bremove[_\s-]*random[_\s-]*consonant\b", lowered) or ("remove" in lowered and "consonant" in lowered):
            add_rule("remove_random_consonant")
        if re.search(r"\bduplicate[_\s-]*random[_\s-]*letter\b", lowered) or ("duplicate" in lowered and "letter" in lowered):
            add_rule("duplicate_random_letter_as_double_letter")
        if re.search(r"\babbreviat|abbreviate_name_parts|shorten[_\s-]*name[_\s-]*to[_\s-]*abbreviations\b", lowered):
            add_rule("shorten_name_to_abbreviations")
        if re.search(r"\bname[_\s-]*parts[_\s-]*permut", lowered) or ("permute" in lowered and "name" in lowered):
            add_rule("name_parts_permutations")

        # If a rule percentage is specified but specific rules are phrased loosely,
        # use a safe default trio of letter-only transforms.
        if not rules and ("rule-based" in lowered or "transform" in lowered):
            return [
                "insert_random_letter",
                "swap_adjacent_syllables",
                "shorten_name_to_abbreviations",
            ]
        return rules

    @staticmethod
    def _normalize_name_text(raw_name: str) -> str:
        """Unicode-normalize and keep name-like text only."""
        normalized = unicodedata.normalize("NFKC", str(raw_name or ""))
        normalized = normalized.replace("’", "'").replace("`", "'").replace("´", "'")
        cleaned = "".join(
            ch if (ch.isalpha() or ch.isspace() or ch in {"'", "-"}) else " "
            for ch in normalized
        )
        cleaned = cleaned.replace("-", " ").replace("'", " ")
        return " ".join(cleaned.split()).strip()

    @staticmethod
    def _canonical_name_key(raw_name: str) -> str:
        normalized = Miner._normalize_name_text(raw_name)
        if not normalized:
            return ""
        decomposed = unicodedata.normalize("NFKD", normalized)
        without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        return " ".join(without_marks.casefold().split())

    @staticmethod
    def _token_overlap_ratio(left: str, right: str) -> float:
        left_tokens = {token for token in left.split() if token}
        right_tokens = {token for token in right.split() if token}
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / float(max(len(left_tokens), len(right_tokens)))

    @staticmethod
    def _compact_text(value: str) -> str:
        compact = re.sub(r"[^a-z]", "", value.lower())
        compact = compact.replace("republic", "").replace("kingdom", "").replace("state", "")
        compact = compact.replace("federation", "").replace("democratic", "").replace("islamic", "")
        return compact

    def _is_geo_or_noise_token(self, token: str, seed_tokens: set[str]) -> bool:
        lowered = token.casefold().strip()
        if not lowered:
            return True
        if lowered in seed_tokens:
            return False
        if lowered in self.NON_NAME_TOKENS:
            return True
        if lowered in self.COUNTRY_ALIASES:
            return True

        country_map = getattr(self, "_country_name_to_code", {}) or {}
        if lowered in country_map:
            return True

        country_compact = getattr(self, "_country_compact_to_code", {}) or {}
        if self._compact_text(lowered) in country_compact:
            return True

        return False

    def _is_name_like_candidate(self, candidate: str, seed_name: str, is_multipart: bool) -> bool:
        candidate_norm = self._normalize_name_text(candidate)
        seed_norm = self._normalize_name_text(seed_name)
        if not candidate_norm or not seed_norm:
            return False

        candidate_key = self._canonical_name_key(candidate_norm)
        seed_key = self._canonical_name_key(seed_norm)
        if not candidate_key or candidate_key == seed_key:
            return False

        candidate_parts = candidate_norm.split()
        seed_parts = seed_norm.split()

        if is_multipart:
            if len(candidate_parts) < 2:
                return False
            if len(candidate_parts) > max(len(seed_parts) + 1, 4):
                return False
        else:
            if len(candidate_parts) != 1:
                return False

        seed_tokens = set(seed_key.split())
        allowed_initials = {token[0] for token in seed_tokens if token}
        for part in candidate_parts:
            if any(ch.isdigit() for ch in part):
                return False
            if len(part) > 36:
                return False
            if len(part) == 1 and part.casefold() not in allowed_initials:
                return False
            if self._is_geo_or_noise_token(part, seed_tokens):
                return False

        dedup_parts = {part.casefold() for part in candidate_parts if part.strip()}
        if len(candidate_parts) > 1 and len(dedup_parts) == 1:
            return False

        similarity = SequenceMatcher(None, seed_key, candidate_key).ratio()
        if is_multipart and similarity < 0.40:
            return False
        if not is_multipart and similarity < 0.30:
            return False

        return True

    def _resolve_identity_match(
        self,
        seed_name: str,
        identity_by_name: Dict[str, List[str]],
        identity_by_normalized_name: Dict[str, List[str]],
        identity_candidates: List[Tuple[str, str, List[str]]],
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        direct = identity_by_name.get(seed_name)
        if direct is not None:
            return seed_name, direct

        normalized_seed = self._normalize_name_text(seed_name)
        if normalized_seed:
            normalized_match = identity_by_normalized_name.get(normalized_seed)
            if normalized_match is not None:
                canonical_name = str(normalized_match[0]).strip() if normalized_match else seed_name
                return canonical_name, normalized_match

        canonical_seed = self._canonical_name_key(seed_name)
        if not canonical_seed:
            return None, None

        fuzzy_rows: List[Tuple[float, float, float, str, List[str]]] = []
        sorted_seed = " ".join(sorted(canonical_seed.split()))

        for candidate_name, candidate_key, identity in identity_candidates:
            if not candidate_key:
                continue
            seq_score = SequenceMatcher(None, canonical_seed, candidate_key).ratio()
            sorted_candidate = " ".join(sorted(candidate_key.split()))
            sort_score = SequenceMatcher(None, sorted_seed, sorted_candidate).ratio()
            overlap = self._token_overlap_ratio(canonical_seed, candidate_key)
            score = max(seq_score, sort_score * 0.96 + overlap * 0.04, (seq_score + overlap) / 2.0)
            fuzzy_rows.append((score, seq_score, overlap, candidate_name, identity))

        if not fuzzy_rows:
            return None, None

        fuzzy_rows.sort(key=lambda item: item[0], reverse=True)
        best_score, best_seq, best_overlap, best_name, best_identity = fuzzy_rows[0]
        second_score = fuzzy_rows[1][0] if len(fuzzy_rows) > 1 else 0.0

        token_count = max(len(canonical_seed.split()), 1)
        threshold = 0.94 if token_count == 1 else 0.88
        if best_score >= threshold and (best_score - second_score) >= 0.03:
            bt.logging.info(
                f"Identity fallback matched '{seed_name}' -> '{best_name}' "
                f"(score={best_score:.3f}, seq={best_seq:.3f}, overlap={best_overlap:.3f})"
            )
            return best_name, best_identity

        return None, None

    def _clean_name_candidate(self, raw_name: str, seed_name: str, is_multipart: bool) -> Optional[str]:
        """Normalize and validate name candidates before scoring."""
        if not raw_name:
            return None
        candidate = self._normalize_name_text(str(raw_name))
        if not candidate:
            return None
        if not self._is_name_like_candidate(candidate, seed_name, is_multipart):
            return None
        return candidate

    @staticmethod
    def _apply_word_case(original: str, mutated: str) -> str:
        if not mutated:
            return mutated
        if original.isupper():
            return mutated.upper()
        if original and original[0].isupper():
            return mutated.capitalize()
        return mutated.lower()

    def _mutate_name_part(self, part: str) -> List[str]:
        """Generate lightweight name mutations for fallback diversity."""
        part_clean = "".join(ch for ch in str(part) if ch.isalpha())
        if len(part_clean) < 3:
            return []
        lower = part_clean.lower()
        candidates = []

        # Swap adjacent characters near the center.
        center = max(1, min(len(lower) - 2, len(lower) // 2))
        swapped = list(lower)
        swapped[center], swapped[center + 1] = swapped[center + 1], swapped[center]
        candidates.append("".join(swapped))

        # Remove one vowel if possible.
        vowel_idx = next((i for i, ch in enumerate(lower[:-1]) if ch in "aeiou"), None)
        if vowel_idx is not None:
            candidates.append(lower[:vowel_idx] + lower[vowel_idx + 1 :])

        # Duplicate one consonant for typo-like behavior.
        consonant_idx = next((i for i, ch in enumerate(lower[:-1]) if ch.isalpha() and ch not in "aeiou"), None)
        if consonant_idx is not None:
            candidates.append(lower[: consonant_idx + 1] + lower[consonant_idx] + lower[consonant_idx + 1 :])

        # Simple phonetic substitutions.
        substitutions = [("ph", "f"), ("f", "ph"), ("c", "k"), ("k", "c"), ("i", "y"), ("y", "i"), ("v", "w"), ("w", "v")]
        for src, dst in substitutions:
            if src in lower:
                candidates.append(lower.replace(src, dst, 1))

        unique = []
        seen = set()
        for cand in candidates:
            normalized = cand.strip().lower()
            if len(normalized) < 2 or normalized in seen or normalized == lower:
                continue
            seen.add(normalized)
            unique.append(self._apply_word_case(part_clean, normalized))
        return unique

    def _apply_rule_to_seed_name(self, seed_name: str, rule: str, attempt: int) -> Optional[str]:
        """
        Build one deterministic candidate intended to satisfy a requested transformation rule.
        Uses letter-only transforms to avoid non-letter penalties.
        """
        parts = [p for p in str(seed_name).split() if p.strip()]
        if not parts:
            return None

        mutable_parts = ["".join(ch for ch in p if ch.isalpha()) for p in parts]
        mutable_parts = [p if p else raw for p, raw in zip(mutable_parts, parts)]
        target_idx = attempt % len(mutable_parts)
        base_part = mutable_parts[target_idx]
        if len(base_part) < 2:
            return None

        if rule == "insert_random_letter":
            letter_pool = "aeiourstlnm"
            insert_at = max(1, min(len(base_part) - 1, len(base_part) // 2))
            inserted = letter_pool[(attempt + len(base_part)) % len(letter_pool)]
            mutated = base_part[:insert_at] + inserted + base_part[insert_at:]
            result = mutable_parts.copy()
            result[target_idx] = self._apply_word_case(base_part, mutated)
            return " ".join(result)

        if rule in ("swap_adjacent_syllables", "swap_random_letter"):
            if len(base_part) < 3:
                return None
            swap_at = max(0, min(len(base_part) - 2, len(base_part) // 2 - 1 + (attempt % 2)))
            chars = list(base_part)
            if chars[swap_at].lower() == chars[swap_at + 1].lower():
                swap_at = max(0, min(len(base_part) - 2, swap_at + 1))
            chars[swap_at], chars[swap_at + 1] = chars[swap_at + 1], chars[swap_at]
            result = mutable_parts.copy()
            result[target_idx] = "".join(chars)
            return " ".join(result)

        if rule in ("delete_random_letter", "remove_random_vowel", "remove_random_consonant"):
            if len(base_part) < 4:
                return None
            idx_pool = list(range(len(base_part)))
            if rule == "remove_random_vowel":
                idx_pool = [i for i, ch in enumerate(base_part) if ch.lower() in "aeiou"]
            elif rule == "remove_random_consonant":
                idx_pool = [i for i, ch in enumerate(base_part) if ch.isalpha() and ch.lower() not in "aeiou"]
            if not idx_pool:
                return None
            rm_at = idx_pool[attempt % len(idx_pool)]
            mutated = base_part[:rm_at] + base_part[rm_at + 1:]
            if len(mutated) < 2:
                return None
            result = mutable_parts.copy()
            result[target_idx] = mutated
            return " ".join(result)

        if rule == "duplicate_random_letter_as_double_letter":
            dup_at = max(0, min(len(base_part) - 1, len(base_part) // 2))
            mutated = base_part[:dup_at] + base_part[dup_at] + base_part[dup_at:]
            result = mutable_parts.copy()
            result[target_idx] = mutated
            return " ".join(result)

        if rule == "shorten_name_to_abbreviations":
            if len(mutable_parts) == 1:
                short_len = max(2, min(len(base_part) - 1, len(base_part) // 2 + 1))
                return mutable_parts[0][:short_len]
            abbreviated = []
            for word in mutable_parts:
                if len(word) <= 2:
                    abbreviated.append(word)
                else:
                    short_len = max(1, min(len(word) - 1, int(round(len(word) * 0.6))))
                    abbreviated.append(word[:short_len])
            return " ".join(abbreviated)

        if rule == "name_parts_permutations" and len(mutable_parts) >= 2:
            shift = (attempt % (len(mutable_parts) - 1)) + 1
            return " ".join(mutable_parts[shift:] + mutable_parts[:shift])

        return None

    def _build_rule_aware_candidates(self, seed_name: str, rules: List[str], target_count: int) -> List[str]:
        """Generate a bounded set of rule-targeted name candidates."""
        if target_count <= 0 or not rules:
            return []

        is_multipart = len(seed_name.split()) > 1
        candidates: List[str] = []
        seen = {seed_name.lower()}
        max_attempts = max(24, target_count * max(3, len(rules)) * 6)
        attempt = 0
        while len(candidates) < target_count and attempt < max_attempts:
            rule = rules[attempt % len(rules)]
            raw_candidate = self._apply_rule_to_seed_name(seed_name, rule, attempt)
            cleaned = self._clean_name_candidate(raw_candidate or "", seed_name, is_multipart)
            attempt += 1
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cleaned)
        return candidates

    def _build_name_fallback_candidates(self, seed_name: str, is_multipart: bool) -> List[str]:
        """Generate structured fallback names when LLM output is too short."""
        seed_parts = seed_name.split()
        if not seed_parts:
            return []
        if not is_multipart:
            return self._mutate_name_part(seed_parts[0])

        first = seed_parts[0]
        last = seed_parts[-1]
        middle = seed_parts[1:-1]
        first_mut = self._mutate_name_part(first)
        last_mut = self._mutate_name_part(last)
        variants = []

        for fm in first_mut:
            variants.append(" ".join([fm] + middle + [last]))
        for lm in last_mut:
            variants.append(" ".join([first] + middle + [lm]))
        for idx, fm in enumerate(first_mut[:4]):
            if idx < len(last_mut):
                variants.append(" ".join([fm] + middle + [last_mut[idx]]))

        # Add an initials-style variant for rule coverage.
        if len(first) > 1:
            variants.append(" ".join([f"{first[0]}", *middle, last]))

        deduped = []
        seen = set()
        for variant in variants:
            normalized = " ".join(variant.split()).strip().lower()
            if not normalized or normalized in seen or normalized == seed_name.lower():
                continue
            seen.add(normalized)
            deduped.append(" ".join(variant.split()))
        return deduped

    def _ensure_target_name_count(
        self,
        seed_name: str,
        extracted_names: List[str],
        target_count: int,
        query_template: str = "",
    ) -> List[str]:
        """Ensure we return a stable count of unique name variations."""
        is_multipart = len(seed_name.split()) > 1
        cleaned = []
        seen = set()
        for raw in extracted_names:
            candidate = self._clean_name_candidate(raw, seed_name, is_multipart)
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(candidate)
            if len(cleaned) >= target_count:
                return cleaned[:target_count]

        # Proactively inject a bounded set of rule-targeted candidates when requested.
        rule_percentage = self._extract_rule_percentage(query_template, default=0.0)
        requested_rules = self._extract_requested_transformations(query_template)
        if requested_rules and rule_percentage > 0:
            rule_target_count = int(round(target_count * rule_percentage))
            if target_count > 1:
                rule_target_count = min(target_count - 1, max(1, rule_target_count))
            else:
                rule_target_count = 1
            rule_candidates = self._build_rule_aware_candidates(seed_name, requested_rules, rule_target_count)
            prioritized = []
            prioritized_seen = set()
            for candidate in rule_candidates + cleaned:
                key = candidate.lower()
                if key in prioritized_seen:
                    continue
                prioritized_seen.add(key)
                prioritized.append(candidate)
                if len(prioritized) >= target_count:
                    break
            cleaned = prioritized
            seen = prioritized_seen

        fallback_candidates = self._build_name_fallback_candidates(seed_name, is_multipart)
        for fallback in fallback_candidates:
            key = fallback.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(fallback)
            if len(cleaned) >= target_count:
                return cleaned[:target_count]

        # Last resort: deterministic perturbation to avoid empty responses.
        seed_parts = seed_name.split()
        counter = 0
        while len(cleaned) < target_count and counter < target_count * 6:
            counter += 1
            if not seed_parts:
                break
            part_idx = counter % len(seed_parts)
            part = seed_parts[part_idx]
            if len(part) < 2:
                continue
            if all(ord(ch) < 128 for ch in part):
                pool = "aeiouwy"
                replacement = pool[(counter + part_idx) % len(pool)]
            else:
                replacement = part[0]
            perturb = part[:-1] + replacement
            perturb = self._apply_word_case(part, perturb)
            generated = seed_parts.copy()
            generated[part_idx] = perturb
            candidate = " ".join(generated)
            key = candidate.lower()
            if key == seed_name.lower() or key in seen:
                continue
            seen.add(key)
            cleaned.append(candidate)
        return cleaned[:target_count]

    def _build_dob_variations(self, seed_dob: str, target_count: int) -> List[str]:
        """Build DOB variations covering validator scoring buckets."""
        if target_count <= 0:
            return []
        try:
            base = datetime.strptime(seed_dob.strip(), "%Y-%m-%d")
        except Exception:
            fallback = seed_dob.strip() if seed_dob and seed_dob.strip() else "1990-01-01"
            return [fallback for _ in range(target_count)]

        categories = [
            base + timedelta(days=1),
            base - timedelta(days=3),
            base + timedelta(days=30),
            base - timedelta(days=90),
            base + timedelta(days=365),
        ]
        dob_values = [d.strftime("%Y-%m-%d") for d in categories]
        dob_values.append(base.strftime("%Y-%m"))

        extra_offsets = [2, -7, 14, -21, 45, -120, 180, -240, 330, -360]
        for offset in extra_offsets:
            dob_values.append((base + timedelta(days=offset)).strftime("%Y-%m-%d"))

        deduped = []
        seen = set()
        for dob in dob_values:
            if dob in seen:
                continue
            seen.add(dob)
            deduped.append(dob)
            if len(deduped) >= target_count:
                return deduped[:target_count]

        while len(deduped) < target_count:
            jitter = (len(deduped) + 1) * 11
            deduped.append((base + timedelta(days=jitter)).strftime("%Y-%m-%d"))
        return deduped[:target_count]

    def _resolve_region_target(self, seed_address: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Resolve seed address into country name/code and optional preferred city.
        Returns: (country_name, country_code, preferred_city)
        """
        seed = str(seed_address or "").strip()
        if not seed:
            return "United States", self._country_name_to_code.get("united states"), "New York"

        normalized_seed = seed.lower()
        parts = [p.strip() for p in seed.split(",") if p.strip()]

        country_code = None
        country_name = None
        preferred_city = None
        seed_is_country_like = False

        # Try direct full-seed country match first.
        if normalized_seed in self._country_name_to_code:
            country_code = self._country_name_to_code[normalized_seed]
            country_name = self._country_code_to_name.get(country_code, seed)
            seed_is_country_like = True
        elif self._compact_text(normalized_seed) in self._country_compact_to_code:
            country_code = self._country_compact_to_code[self._compact_text(normalized_seed)]
            country_name = self._country_code_to_name.get(country_code, seed)
            seed_is_country_like = True

        # Try last comma-separated segment as country.
        if not country_code and parts:
            last_part = parts[-1].lower()
            if last_part in self._country_name_to_code:
                country_code = self._country_name_to_code[last_part]
                country_name = self._country_code_to_name.get(country_code, parts[-1])
            elif self._compact_text(last_part) in self._country_compact_to_code:
                country_code = self._country_compact_to_code[self._compact_text(last_part)]
                country_name = self._country_code_to_name.get(country_code, parts[-1])

        # Fuzzy fallback for alternate country naming.
        if not country_code and self._country_code_to_name:
            best_code = None
            best_score = 0.0
            for code, canonical_name in self._country_code_to_name.items():
                ratio = SequenceMatcher(None, normalized_seed, canonical_name.lower()).ratio()
                compact_ratio = SequenceMatcher(
                    None, self._compact_text(normalized_seed), self._compact_text(canonical_name)
                ).ratio()
                score = max(ratio, compact_ratio)
                if score > best_score:
                    best_score = score
                    best_code = code
            if best_code and best_score >= 0.74:
                country_code = best_code
                country_name = self._country_code_to_name.get(best_code, seed)
                seed_is_country_like = True

        # If still unresolved, interpret seed as city.
        if not country_code:
            city_key = normalized_seed
            if city_key in self._city_to_country_codes:
                ranked = self._city_to_country_codes.get(city_key, [])
                if ranked:
                    country_code = ranked[0]
                    country_name = self._country_code_to_name.get(country_code, seed)
                    preferred_city = seed

        # Derive preferred city from seed parts if possible.
        if parts and len(parts) >= 2 and not seed_is_country_like:
            city_candidate = parts[0]
            if not any(ch.isdigit() for ch in city_candidate):
                preferred_city = city_candidate

        if not country_name:
            country_name = seed
        return country_name, country_code, preferred_city

    def _pick_city_for_country(self, country_code: Optional[str], preferred_city: Optional[str], idx: int) -> str:
        """Pick a deterministic city for the resolved country."""
        if preferred_city and idx == 0:
            return preferred_city
        if country_code and country_code in self._city_names_by_country:
            candidates = self._city_names_by_country[country_code]
            if candidates:
                return candidates[(idx + (1 if preferred_city else 0)) % len(candidates)]
        return "New York"

    def _make_address_line(self, city: str, country: str, idx: int) -> str:
        """Create a realistic, validator-friendly address line."""
        street_root = self.STREET_ROOTS[idx % len(self.STREET_ROOTS)]
        street_suffix = self.STREET_SUFFIXES[idx % len(self.STREET_SUFFIXES)]
        district = self.DISTRICT_WORDS[idx % len(self.DISTRICT_WORDS)]
        house_num = 10 + (idx * 7 % 980)
        area_num = 1 + (idx * 3 % 17)
        postal = 10000 + (idx * 113 % 89999)
        return (
            f"{house_num} {street_root} {street_suffix}, "
            f"{district} {area_num}, "
            f"{city} {postal}, "
            f"{country}"
        )

    def _build_address_variations(self, seed_address: str, target_count: int) -> List[str]:
        """Build address variations that satisfy validator heuristics and region checks."""
        if target_count <= 0:
            return []
        country_name, country_code, preferred_city = self._resolve_region_target(seed_address)
        addresses = []
        seen = set()
        i = 0
        while len(addresses) < target_count and i < target_count * 8:
            city = self._pick_city_for_country(country_code, preferred_city, i)
            candidate = self._make_address_line(city, country_name, i)
            normalized = candidate.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                addresses.append(candidate)
            i += 1

        if not addresses:
            fallback = "120 Central Avenue, District 4, New York 10001, United States"
            addresses = [fallback for _ in range(target_count)]
        elif len(addresses) < target_count:
            addresses.extend([addresses[-1]] * (target_count - len(addresses)))
        return addresses[:target_count]

    @staticmethod
    def _increment_fail_reason(fail_reasons: Dict[str, int], reason: str, amount: int = 1) -> None:
        if not reason:
            return
        if amount <= 0:
            return
        fail_reasons[reason] = int(fail_reasons.get(reason, 0)) + int(amount)

    @staticmethod
    def _merge_fail_reasons(target: Dict[str, int], source: Dict[str, int]) -> None:
        for reason, count in (source or {}).items():
            Miner._increment_fail_reason(target, str(reason), int(count))

    def _emit_run_metrics(
        self,
        run_id: int,
        requested_names: int,
        returned_names: int,
        s3_submissions: int,
        fail_reasons: Dict[str, int],
    ) -> None:
        payload = {
            "run_id": int(run_id),
            "requested_names": int(max(requested_names, 0)),
            "returned_names": int(max(returned_names, 0)),
            "s3_submissions": int(max(s3_submissions, 0)),
            "fail_reasons": {k: int(v) for k, v in sorted((fail_reasons or {}).items()) if int(v) > 0},
        }
        bt.logging.info(f"RUN_METRICS {json.dumps(payload, ensure_ascii=False, sort_keys=True)}")

    def _emit_telegram_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Emit structured log events consumed by monitoring/telegram_notifier.py."""
        envelope = {
            "event": str(event),
            "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payload": payload or {},
        }
        try:
            bt.logging.info(
                f"TELEGRAM_EVENT {json.dumps(envelope, ensure_ascii=False, default=str, sort_keys=True)}"
            )
        except Exception as e:
            bt.logging.debug(f"Failed to emit TELEGRAM_EVENT {event}: {e}")

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _summarize_identity_payload(self, identity_rows: List[List[str]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for index, identity in enumerate(identity_rows or [], start=1):
            raw = [str(part) for part in list(identity or [])]
            rows.append(
                {
                    "index": index,
                    "name": raw[0] if len(raw) > 0 else "",
                    "dob": raw[1] if len(raw) > 1 else "",
                    "address": raw[2] if len(raw) > 2 else "",
                    "raw": raw,
                }
            )
        return rows

    def _summarize_image_request(self, image_request: Any) -> Optional[Dict[str, Any]]:
        if not image_request:
            return None

        variation_requests_raw = list(getattr(image_request, "variation_requests", []) or [])
        variation_requests: List[Dict[str, Any]] = []
        for index, request in enumerate(variation_requests_raw, start=1):
            variation_requests.append(
                {
                    "index": index,
                    "type": str(getattr(request, "type", "") or ""),
                    "intensity": str(getattr(request, "intensity", "") or ""),
                    "description": str(getattr(request, "description", "") or ""),
                    "detail": str(getattr(request, "detail", "") or ""),
                }
            )

        base_image = str(getattr(image_request, "base_image", "") or "")
        base_image_hash = hashlib.sha256(base_image.encode("utf-8")).hexdigest() if base_image else ""

        return {
            "image_filename": str(getattr(image_request, "image_filename", "") or ""),
            "challenge_id": str(getattr(image_request, "challenge_id", "") or ""),
            "target_drand_round": self._safe_int(getattr(image_request, "target_drand_round", 0), default=0),
            "reveal_timestamp": self._safe_int(getattr(image_request, "reveal_timestamp", 0), default=0),
            "base_image_base64_chars": len(base_image),
            "base_image_sha256": base_image_hash,
            "variation_requests": variation_requests,
            "requested_variations": len(variation_requests),
        }

    def _emit_validator_request_event(
        self,
        run_id: int,
        synapse: IdentitySynapse,
        requested_names: int,
        timeout_seconds: float,
    ) -> None:
        dendrite = getattr(synapse, "dendrite", None)
        validator_hotkey = str(getattr(dendrite, "hotkey", "") or "")
        validator_name = self.WHITELISTED_VALIDATORS.get(validator_hotkey, "UnknownValidator")

        payload = {
            "run_id": int(run_id),
            "requested_names": int(max(requested_names, 0)),
            "timeout_seconds": float(timeout_seconds),
            "validator_name": validator_name,
            "validator_hotkey": validator_hotkey,
            "nonce": str(getattr(dendrite, "nonce", "") or ""),
            "uuid": str(getattr(dendrite, "uuid", "") or ""),
            "body_hash": str(getattr(synapse, "computed_body_hash", "") or ""),
            "query_template": str(getattr(synapse, "query_template", "") or ""),
            "identity": self._summarize_identity_payload(list(getattr(synapse, "identity", []) or [])),
            "image_request": self._summarize_image_request(getattr(synapse, "image_request", None)),
        }
        self._emit_telegram_event("validator_request", payload)

    def _emit_submission_status_event(
        self,
        run_id: int,
        requested_names: int,
        returned_names: int,
        s3_submissions: int,
        fail_reasons: Dict[str, int],
        note: str = "",
    ) -> None:
        payload = {
            "run_id": int(run_id),
            "submitted_to_validator": True,
            "requested_names": int(max(requested_names, 0)),
            "returned_names": int(max(returned_names, 0)),
            "s3_submissions": int(max(s3_submissions, 0)),
            "fail_reasons": {k: int(v) for k, v in sorted((fail_reasons or {}).items()) if int(v) > 0},
            "note": str(note or ""),
        }
        self._emit_telegram_event("validator_submission_status", payload)

    def _should_prewarm_flux(self) -> bool:
        value = os.environ.get("SN54_FLUX_PREWARM", "true").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _run_flux_prewarm(self) -> None:
        try:
            bt.logging.info("FLUX prewarm: starting background warm-up")
            prewarm_flux_pipeline()
            bt.logging.info("FLUX prewarm: completed")
        except Exception as e:
            bt.logging.warning(f"FLUX prewarm skipped: {e}")

    def _schedule_flux_prewarm(self) -> None:
        if not self._should_prewarm_flux():
            bt.logging.info("FLUX prewarm disabled via SN54_FLUX_PREWARM")
            return
        thread = threading.Thread(target=self._run_flux_prewarm, name="sn54-flux-prewarm", daemon=True)
        thread.start()

    def _sync_dynamic_validator_whitelist(self):
        """Sync allowlist with current validator-permit hotkeys on chain."""
        added = 0
        try:
            for uid, hotkey in enumerate(self.metagraph.hotkeys):
                if hotkey in self.WHITELISTED_VALIDATORS:
                    continue
                if bool(self.metagraph.validator_permit[uid]):
                    self.WHITELISTED_VALIDATORS[hotkey] = f"DynamicValidator_UID{uid}"
                    added += 1
            bt.logging.info(
                f"Dynamic validator sync complete. Added {added} validator-permitted hotkeys. "
                f"Total allowlist size: {len(self.WHITELISTED_VALIDATORS)}"
            )
        except Exception as e:
            bt.logging.warning(f"Dynamic validator sync failed: {e}")

    def _add_local_validators_to_whitelist(self):
        """Add all registered neurons as whitelisted validators in local_test mode."""
        if not getattr(self.config, 'local_test', False):
            return

        bt.logging.info("Local test mode: Adding all registered neurons to whitelist")
        for i, hotkey in enumerate(self.metagraph.hotkeys):
            if hotkey not in self.WHITELISTED_VALIDATORS:
                self.WHITELISTED_VALIDATORS[hotkey] = f"LocalValidator_UID{i}"
                bt.logging.info(f"  Added {hotkey[:16]}... as LocalValidator_UID{i}")

    def __init__(self, config=None):
        """
        Initialize the Name Variation Miner.
        
        Sets up the LLM client and creates directories for storing mining results.
        Each run will be saved in a separate directory with a unique timestamp.
        
        Args:
            config: Configuration object for the miner
        """
        super(Miner, self).__init__(config=config)
        
        # Initialize the LLM client
        # You can override this in your config by setting model_name
        # Ensure we have a valid model name, defaulting to llama3.2:1b if not specified
        self.model_name = getattr(self.config.neuron, 'model_name', None) if hasattr(self.config, 'neuron') else None
        if self.model_name is None:
            #self.model_name = 'llama3.2:1b'
            # self.model_name = 'tinyllama:latest'
            self.model_name = 'mistral:latest'
            bt.logging.info(f"No model specified in config, using default model: {self.model_name}")
        
        bt.logging.info(f"Using LLM model: {self.model_name}")
        
        # Check if Ollama is available
        try:
            # Check if model exists locally first
            models = ollama.list().get('models', [])
            model_exists = any(model.get('name') == self.model_name for model in models)
            
            if model_exists:
                bt.logging.info(f"Model {self.model_name} already pulled")
            else:
                # Model not found locally, pull it
                bt.logging.info(f"Pulling model {self.model_name}...")
                ollama.pull(self.model_name)
        except Exception as e:
            bt.logging.error(f"Error with Ollama: {str(e)}")
            bt.logging.error("Make sure Ollama is installed and running on this machine")
            bt.logging.error("Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
            bt.logging.error("Start Ollama: ollama serve")
            raise RuntimeError("Ollama is required for this miner. Please install and start Ollama.")
        
        # Create a directory for storing mining results
        # This helps with debugging and analysis
        self.output_path = os.path.join(self.config.logging.logging_dir, "mining_results")
        os.makedirs(self.output_path, exist_ok=True)
        bt.logging.info(f"Mining results will be saved to: {self.output_path}")

        # Initialize geodata indexes for reward-aligned address generation.
        self._initialize_geodata()

        # Keep static trusted keys, but also include live validator-permitted hotkeys.
        self._sync_dynamic_validator_whitelist()
        self.axon.verify_fns[IdentitySynapse.__name__] = self._verify_validator_request

        # Optional startup optimization to reduce first Phase-4 latency.
        self._schedule_flux_prewarm()

        if not is_timelock_available():
            bt.logging.warning(
                "Phase 4 timelock unavailable. Encrypted submissions are disabled "
                "until timelock_wasm_wrapper is installed."
            )

    def _unload_ollama_model(self) -> None:
        """Release Ollama model from GPU memory before Phase 4 image generation."""
        try:
            result = subprocess.run(
                ["ollama", "stop", self.model_name],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode == 0:
                bt.logging.info(f"Released Ollama model {self.model_name} before Phase 4")
            elif result.stderr:
                bt.logging.debug(
                    f"Ollama stop returned {result.returncode}: {result.stderr.strip()[:200]}"
                )
        except Exception as e:
            bt.logging.debug(f"Ollama model unload skipped: {e}")

    async def _verify_validator_request(self, synapse: IdentitySynapse) -> None:
        """
        Rejects any RPC that is not cryptographically proven to come from
        one of the whitelisted validator hotkeys.

        Signature *must* be present and valid.  If anything is missing or
        incorrect we raise `NotVerifiedException`, which the Axon middleware
        converts into a 401 reply.
        """
        # ----------  basic sanity checks  ----------
        if synapse.dendrite is None:
            raise NotVerifiedException("Missing dendrite terminal in request")

        hotkey    = synapse.dendrite.hotkey
        # signature = synapse.dendrite.signature
        nonce     = synapse.dendrite.nonce
        uuid      = synapse.dendrite.uuid
        body_hash = synapse.computed_body_hash

        # 1 — is the sender even on our allow‑list?
        if not self._register_dynamic_validator(hotkey):
            raise NotVerifiedException(f"{hotkey} is not a whitelisted validator")

        # 3 — run all the standard Bittensor checks (nonce window, replay,
        #     timeout, signature, …).  This *does not* insist on a signature,
        #     so we still do step 4 afterwards.
        message = (
            f"nonce: {nonce}. "
            f"hotkey {hotkey}. "
            f"self hotkey {self.wallet.hotkey.ss58_address}. "
            f"uuid {uuid}. "
            f"body hash {body_hash} "
        )
        bt.logging.info(
            f"Verifying message: {message}"
        )

        await self.axon.default_verify(synapse)

        # 5 — all good ➜ let the middleware continue
        bt.logging.info(
            f"Verified call from {self.WHITELISTED_VALIDATORS[hotkey]} ({hotkey})"
        )

    async def forward(self, synapse: IdentitySynapse) -> IdentitySynapse:
        """
        Process a name variation request by generating variations for each name.

        This is the main entry point for the miner's functionality. It:
        1. Receives a request with names and a query template
        2. Processes each name through the LLM
        3. Extracts variations from the LLM responses
        4. Returns the variations to the validator

        Each run is assigned a unique timestamp ID and results are saved in a
        dedicated directory for that run.

        Args:
            synapse: The IdentitySynapse containing names and query template

        Returns:
            The synapse with variations field populated with name variations
        """
        run_id = int(time.time())
        requested_names = len(synapse.identity)
        bt.logging.info(f"Starting run {run_id} for {requested_names} names")

        timeout = getattr(synapse, 'timeout', 120.0)
        bt.logging.info(f"Request timeout: {timeout:.1f}s for {requested_names} names")
        start_time = time.time()

        run_fail_reasons: Dict[str, int] = {}
        self._emit_validator_request_event(
            run_id=run_id,
            synapse=synapse,
            requested_names=requested_names,
            timeout_seconds=timeout,
        )

        run_dir = os.path.join(self.output_path, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

        response_list: List[str] = []
        processed_names: List[str] = []
        timeout_skips = 0

        for identity in tqdm(synapse.identity, desc="Processing identities"):
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            time_buffer = timeout * 0.15

            if remaining < time_buffer:
                timeout_skips = max(0, requested_names - len(processed_names))
                bt.logging.warning(
                    f"Time limit approaching ({elapsed:.1f}/{timeout:.1f}s), "
                    f"processed {len(processed_names)}/{requested_names} identities. "
                    f"Skipping remaining identities to ensure timely response."
                )
                break

            name = identity[0] if len(identity) > 0 else "Unknown"
            dob = identity[1] if len(identity) > 1 else "Unknown"
            address = identity[2] if len(identity) > 2 else "Unknown"

            response_list.append("Respond")
            response_list.append("---")
            response_list.append("Query-" + name)
            response_list.append("---")

            formatted_query = synapse.query_template.replace("{name}", name)
            formatted_query = formatted_query.replace("{address}", address)
            formatted_query = formatted_query.replace("{dob}", dob)

            try:
                bt.logging.info(f"Generating variations for name: {name}, remaining time: {remaining:.1f}s")
                name_respond = self.Get_Respond_LLM(formatted_query)
                response_list.append(name_respond)
                processed_names.append(name)
            except Exception as e:
                bt.logging.error(f"Error querying LLM for name {name}: {str(e)}")
                response_list.append("Error: " + str(e))
                self._increment_fail_reason(run_fail_reasons, "llm_query_error")

        if timeout_skips > 0:
            self._increment_fail_reason(run_fail_reasons, "name_generation_timeout_skips", timeout_skips)

        if not processed_names:
            bt.logging.error("Could not process any names within the timeout period")
            synapse.variations = {}
            synapse.s3_submissions = []
            self._increment_fail_reason(run_fail_reasons, "no_names_processed")
            self._emit_run_metrics(
                run_id=run_id,
                requested_names=requested_names,
                returned_names=0,
                s3_submissions=0,
                fail_reasons=run_fail_reasons,
            )
            self._emit_submission_status_event(
                run_id=run_id,
                requested_names=requested_names,
                returned_names=0,
                s3_submissions=0,
                fail_reasons=run_fail_reasons,
                note="no_names_processed",
            )
            return synapse

        remaining = timeout - (time.time() - start_time)
        bt.logging.info(f"Processing responses with {remaining:.1f}s remaining of {timeout:.1f}s timeout")

        if remaining > 1.0:
            variations = self.process_variations(
                response_list,
                run_id,
                run_dir,
                synapse.identity,
                synapse.query_template,
                run_fail_reasons=run_fail_reasons,
            )
            bt.logging.info(f"======== FINAL VARIATIONS===============================================: {variations}")
            synapse.variations = variations
        else:
            bt.logging.warning("Insufficient time for processing responses, returning empty result")
            synapse.variations = {}
            self._increment_fail_reason(run_fail_reasons, "insufficient_postprocess_time")

        returned_names = len(synapse.variations or {})
        if returned_names < requested_names:
            self._increment_fail_reason(
                run_fail_reasons,
                "identity_return_shortfall",
                requested_names - returned_names,
            )

        total_time = time.time() - start_time
        bt.logging.info(
            f"Request completed in {total_time:.2f}s of {timeout:.1f}s allowed. "
            f"Processed {len(processed_names)}/{requested_names} names."
        )

        bt.logging.info(f"======== SYNAPSE VARIATIONS===============================================: {synapse.variations}")
        bt.logging.info(f"==========================Processed variations for {returned_names} names in run {run_id}")
        bt.logging.info("========================================================================================")

        s3_submissions_count = 0
        synapse.s3_submissions = []

        if hasattr(synapse, 'image_request') and synapse.image_request is not None:
            try:
                self._unload_ollama_model()
                s3_submissions, phase4_fail_reasons = self.process_image_request(synapse)
                synapse.s3_submissions = s3_submissions
                s3_submissions_count = len(s3_submissions)
                self._merge_fail_reasons(run_fail_reasons, phase4_fail_reasons)
                bt.logging.info(f"Phase 4: Generated {s3_submissions_count} S3 submissions")
            except Exception as e:
                bt.logging.error(f"Phase 4: Failed to process image request: {e}")
                self._increment_fail_reason(run_fail_reasons, "phase4_unhandled_exception")
                synapse.s3_submissions = []

        self._emit_run_metrics(
            run_id=run_id,
            requested_names=requested_names,
            returned_names=returned_names,
            s3_submissions=s3_submissions_count,
            fail_reasons=run_fail_reasons,
        )
        self._emit_submission_status_event(
            run_id=run_id,
            requested_names=requested_names,
            returned_names=returned_names,
            s3_submissions=s3_submissions_count,
            fail_reasons=run_fail_reasons,
            note="forward_return",
        )

        return synapse

    def is_valid_image_bytes(self, image_bytes: bytes) -> bool:
        """
        Validate whether raw bytes represent a valid image of any supported format.

        This uses Pillow's image decoder to verify integrity without fully loading
        pixel data.

        Args:
            image_bytes: Raw image bytes

        Returns:
            True if the bytes represent a valid image, False otherwise
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()  # Verifies file integrity without decoding pixels
            return True
        except Exception:
            return False

    def process_image_request(self, synapse: IdentitySynapse) -> Tuple[List[S3Submission], Dict[str, int]]:
        """Process Phase 4 image variation request.

        Generates image variations, encrypts them with drand timelock,
        uploads to S3, and returns S3 submission references.

        Args:
            synapse: IdentitySynapse with image_request

        Returns:
            (s3_submissions, fail_reasons)
        """
        image_request = synapse.image_request
        fail_reasons: Dict[str, int] = {}
        if not image_request:
            return [], fail_reasons

        def req_type(req: Any) -> str:
            return str(getattr(req, "type", None) or (req.get("type") if isinstance(req, dict) else "") or "unknown").strip()

        def req_intensity(req: Any) -> str:
            return str(getattr(req, "intensity", None) or (req.get("intensity") if isinstance(req, dict) else "") or "standard").strip()

        def req_label(req: Any) -> str:
            return f"{req_type(req)}({req_intensity(req)})"

        try:
            bt.logging.info(f"Phase 4: Decoding base image: {image_request.image_filename}")
            base_image = decode_base_image(image_request.base_image)

            seed_image_name = image_request.image_filename
            if seed_image_name.endswith('.png'):
                seed_image_name = seed_image_name[:-4]
            elif seed_image_name.endswith('.jpg') or seed_image_name.endswith('.jpeg'):
                seed_image_name = seed_image_name.rsplit('.', 1)[0]

            variation_requests = list(image_request.variation_requests or [])
            if not variation_requests:
                self._increment_fail_reason(fail_reasons, "phase4_no_variation_requests")
                return [], fail_reasons

            labels = [req_label(req) for req in variation_requests]
            bt.logging.info(
                f"Phase 4: Generating {len(variation_requests)} variations "
                f"(from validator: {labels})"
            )

            configured_attempts = os.environ.get("PHASE4_RETRY_ATTEMPTS")
            if configured_attempts is None and hasattr(self.config, "neuron"):
                configured_attempts = getattr(self.config.neuron, "phase4_retry_attempts", 2)
            max_attempts = max(1, int(configured_attempts or 2))

            target_round = int(image_request.target_drand_round)
            if target_round <= 0:
                self._increment_fail_reason(fail_reasons, "phase4_invalid_target_round", len(variation_requests))
                bt.logging.error(f"Phase 4: Invalid drand target round: {target_round}")
                return [], fail_reasons

            challenge_id = image_request.challenge_id or "sandbox_test"
            path_message = f"{challenge_id}:{self.wallet.hotkey.ss58_address}"
            path_signature = self.wallet.hotkey.sign(path_message.encode()).hex()[:16]
            bt.logging.debug(f"Phase 4: Generated path_signature: {path_signature}")

            if not is_timelock_available():
                bt.logging.error("Phase 4: Timelock unavailable; refusing unencrypted submissions")
                self._increment_fail_reason(fail_reasons, "phase4_timelock_unavailable", len(variation_requests))
                return [], fail_reasons

            s3_submissions: List[S3Submission] = []

            for req_index, request in enumerate(variation_requests, start=1):
                base_variation_type = req_type(request)
                label = req_label(request)
                success = False
                last_reason = "phase4_unknown_failure"

                for attempt in range(1, max_attempts + 1):
                    try:
                        generated = generate_variations(base_image, [request])
                        if not generated:
                            last_reason = "phase4_generation_empty"
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} returned no image"
                            )
                            continue

                        var = generated[0]
                        image_bytes = var.get("image_bytes", b"")
                        if not image_bytes or not self.is_valid_image_bytes(image_bytes):
                            last_reason = "phase4_invalid_image"
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} produced invalid image"
                            )
                            continue

                        if not validate_variation(var, base_image, min_similarity=0.7):
                            last_reason = "phase4_identity_not_preserved"
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} failed face-identity check"
                            )
                            continue

                        image_hash = str(var.get("image_hash", "")).strip()
                        if not image_hash:
                            last_reason = "phase4_missing_image_hash"
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} missing image hash"
                            )
                            continue

                        message = f"challenge:{challenge_id}:hash:{image_hash}"
                        signature = self.wallet.hotkey.sign(message.encode()).hex()

                        encrypted_data = encrypt_image_for_drand(image_bytes, target_round)
                        if encrypted_data is None:
                            last_reason = "phase4_timelock_encrypt_failed"
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} encryption failed"
                            )
                            continue

                        s3_key = upload_to_s3(
                            encrypted_data=encrypted_data,
                            miner_hotkey=self.wallet.hotkey.ss58_address,
                            signature=signature,
                            image_hash=image_hash,
                            target_round=target_round,
                            challenge_id=challenge_id,
                            variation_type=base_variation_type,
                            path_signature=path_signature,
                            seed_image_name=seed_image_name,
                        )

                        if not s3_key:
                            last_reason = "phase4_s3_upload_failed"
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} upload failed"
                            )
                            continue

                        s3_submissions.append(
                            S3Submission(
                                s3_key=s3_key,
                                image_hash=image_hash,
                                signature=signature,
                                variation_type=base_variation_type,
                                path_signature=path_signature,
                            )
                        )
                        bt.logging.debug(f"Phase 4: Created submission for {label} (request #{req_index})")
                        success = True
                        if attempt > 1:
                            bt.logging.info(
                                f"Phase 4: {label} succeeded after retry {attempt}/{max_attempts}"
                            )
                        break

                    except Exception as e:
                        last_reason = "phase4_variation_exception"
                        bt.logging.warning(
                            f"Phase 4: {label} attempt {attempt}/{max_attempts} error: {e}"
                        )

                if not success:
                    self._increment_fail_reason(fail_reasons, last_reason)
                    bt.logging.error(
                        f"Phase 4: {label} failed after {max_attempts} attempts ({last_reason})"
                    )

            bt.logging.info(
                f"Phase 4: Successfully created {len(s3_submissions)} S3 submissions "
                f"out of {len(variation_requests)} requests"
            )
            if fail_reasons:
                bt.logging.warning(f"Phase 4 fail reasons: {dict(sorted(fail_reasons.items()))}")
            return s3_submissions, fail_reasons

        except Exception as e:
            self._increment_fail_reason(fail_reasons, "phase4_request_error")
            bt.logging.error(f"Phase 4: Error in process_image_request: {e}")
            return [], fail_reasons

    def Get_Respond_LLM(self, prompt: str) -> str:
        """
        Query the LLM using Ollama.
        
        This function sends a prompt to the LLM and returns its response.
        It uses the Ollama client to communicate with a locally running LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as a string
            
        Raises:
            Exception: If there's an error communicating with the LLM
        """
        target_count = self._extract_expected_variation_count(prompt, default=10)

        # Add ethical context and purpose explanation
        context_prompt = f"""IMPORTANT CONTEXT: This is synthetic identity-security testing data only.
Use case: defensive KYC/AML robustness testing.

TASK:
{prompt}

OUTPUT RULES (STRICT):
- Return only person-name variations.
- Return exactly {target_count} comma-separated entries.
- No numbering, no bullets, no JSON, no extra commentary.
- Preserve source structure: single-part stays single-part; multi-part stays multi-part.
- Keep each variation as a plausible legal name token sequence.
- Do not include countries, cities, addresses, metadata, IDs, or occupations.
- Do not include the unchanged original seed name.
- No duplicates.
- Allowed characters: letters from any language plus spaces, apostrophes, and hyphens.
- Ignore image/UAV/model instructions if present in the task text.
"""

        # Use Ollama to query the LLM
        try:
            # Create Ollama client with configured URL
            client = ollama.Client(host=getattr(self.config.neuron, 'ollama_url', 'http://127.0.0.1:11434'))
            response = client.chat(
                self.model_name, 
                messages=[{
                    'role': 'user',
                    'content': context_prompt,
                }],
                options={
                    # Add a reasonable timeout to ensure we don't get stuck
                    "num_predict": 1024,
                    "temperature": 0.35,
                    "top_p": 0.85
                }
            )
            
            # Extract and return the content of the response
            return response['message']['content']
        except Exception as e:
            bt.logging.error(f"LLM query failed: {str(e)}")
            raise
    
    def process_variations(
        self,
        Response_list: List[str],
        run_id: int,
        run_dir: str,
        identity_list: List[List[str]],
        query_template: str,
        run_fail_reasons: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[List[str]]]:
        """
        Process LLM responses to extract identity variations.

        This function takes the raw LLM responses and extracts the name variations
        using the Process_function. It then creates structured variations that include
        name, DOB, and address variations for each identity.

        Args:
            Response_list: List of LLM responses in the format:
                          ["Respond", "---", "Query-{name}", "---", "{LLM response}"]
            run_id: Unique identifier for this processing run
            run_dir: Directory to save run-specific files
            identity_list: List of identity arrays, each containing [name, dob, address]
            query_template: Validator query used to infer expected variation count
            run_fail_reasons: Mutable fail-reason counter for structured run metrics

        Returns:
            Dictionary mapping each name to its list of [name, dob, address] variations
        """
        if run_fail_reasons is None:
            run_fail_reasons = {}

        bt.logging.info(f"Processing {len(Response_list)} responses")
        responds = "".join(Response_list).split("Respond")
        expected_count = self._extract_expected_variation_count(query_template, default=10)
        bt.logging.info(f"Parsed expected variation count from query: {expected_count}")

        identity_by_name: Dict[str, List[str]] = {}
        identity_by_normalized_name: Dict[str, List[str]] = {}
        identity_candidates: List[Tuple[str, str, List[str]]] = []

        for identity in identity_list:
            if not identity:
                continue
            seed_name = str(identity[0]).strip()
            if not seed_name:
                continue
            identity_by_name[seed_name] = identity
            norm_seed = self._normalize_name_text(seed_name)
            if norm_seed:
                identity_by_normalized_name[norm_seed] = identity
            canonical_seed = self._canonical_name_key(seed_name)
            if canonical_seed:
                identity_candidates.append((seed_name, canonical_seed, identity))

        name_variations: Dict[str, List[List[str]]] = {}

        for i in range(1, len(responds)):
            try:
                llm_respond = self.Process_function(responds[i], False)
                seed_name = llm_respond[0]

                matched_seed_name, matching_identity = self._resolve_identity_match(
                    seed_name,
                    identity_by_name,
                    identity_by_normalized_name,
                    identity_candidates,
                )
                if matching_identity is None:
                    bt.logging.warning(f"Could not find identity for name {seed_name}")
                    self._increment_fail_reason(run_fail_reasons, "identity_match_miss")
                    continue

                canonical_seed_name = matched_seed_name or seed_name
                seed_dob = str(matching_identity[1]).strip() if len(matching_identity) > 1 else ""
                seed_address = str(matching_identity[2]).strip() if len(matching_identity) > 2 else ""

                raw_name_variations = [var for var in llm_respond[2] if not pd.isna(var) and str(var).strip()]
                names = self._ensure_target_name_count(
                    canonical_seed_name,
                    raw_name_variations,
                    expected_count,
                    query_template=query_template,
                )
                if not names:
                    bt.logging.warning(f"No valid name variations for {canonical_seed_name}; skipping")
                    self._increment_fail_reason(run_fail_reasons, "no_valid_name_variations")
                    name_variations[canonical_seed_name] = []
                    continue

                target_count = min(expected_count, len(names))
                names = names[:target_count]
                if target_count < expected_count:
                    self._increment_fail_reason(
                        run_fail_reasons,
                        "variation_shortfall",
                        expected_count - target_count,
                    )

                dob_variants = self._build_dob_variations(seed_dob, target_count)
                address_variants = self._build_address_variations(seed_address, target_count)

                structured = []
                for idx in range(target_count):
                    structured.append([
                        names[idx],
                        dob_variants[idx] if idx < len(dob_variants) else (seed_dob or ""),
                        address_variants[idx] if idx < len(address_variants) else (seed_address or ""),
                    ])

                name_variations[canonical_seed_name] = structured
                bt.logging.info(
                    f"Processed {len(structured)} structured variations for {canonical_seed_name} "
                    f"(DOB non-empty: {sum(1 for s in structured if len(s) > 1 and s[1].strip())}, "
                    f"Address non-empty: {sum(1 for s in structured if len(s) > 2 and s[2].strip())})"
                )
            except Exception as e:
                bt.logging.error(f"Error processing response {i}: {e}")
                self._increment_fail_reason(run_fail_reasons, "response_parse_error")

        bt.logging.info(f"Generated structured variations for {len(name_variations)} seed names")
        return name_variations

    def save_variations_to_json(self, name_variations: Dict[str, List[str]], run_id: int, run_dir: str) -> None:
        """
        Save processed variations to JSON and DataFrame for debugging and analysis.
        
        This function saves the processed variations in multiple formats:
        1. A pandas DataFrame saved as a pickle file in the run-specific directory
        2. A JSON file with the name variations in the run-specific directory
        3. A JSON file with the model name and run ID in the main output directory
        
        Each file is named with the run ID to distinguish between different runs.
        
        Args:
            name_variations: Dictionary mapping names to variations
            run_id: Unique identifier for this processing run
            run_dir: Directory to save run-specific files
        """
        bt.logging.info(f"=================== Name variations: {name_variations}")
        bt.logging.info(f"=================== Run ID: {run_id}")
        bt.logging.info(f"=================== Run directory: {run_dir}")
        bt.logging.info("Saving variations to JSON and DataFrame")

        # Find the maximum number of variations for any name
        max_variations = max([len(vars) for vars in name_variations.values()]) if name_variations else 0
        bt.logging.info(f"Maximum number of variations found: {max_variations}")
        
        # Create a DataFrame with columns for the name and each variation
        columns = ['Name'] + [f'Var_{i+1}' for i in range(max_variations)]
        result_df = pd.DataFrame(columns=columns)
        
        # Fill the DataFrame with names and their variations, padding with empty strings if needed
        for i, (name, variations) in enumerate(name_variations.items()):
            row_data = [name] + variations + [''] * (max_variations - len(variations))
            result_df.loc[i] = row_data
        
        # Note: We no longer need to clean the data here since it's already cleaned
        # in the process_variations function
        
        # Save DataFrame to pickle for backup and analysis
        # Include run_id in the filename
        #df_path = os.path.join(run_dir, f"variations_df_{run_id}.pkl")
        #result_df.to_pickle(df_path)
        
        # Convert DataFrame to JSON format
        json_data = {}
        for i, row in result_df.iterrows():
            name = row['Name']
            # Extract non-empty variations
            variations = [var for var in row[1:] if var != ""]
            json_data[name] = variations
        
        # Save to JSON file
        # Include run_id in the filename
        # json_path = os.path.join(run_dir, f"variations_{run_id}.json")
        # import json
        # with open(json_path, 'w', encoding='utf-8') as f:
        #     json.dump(json_data, f, indent=4)
        # bt.logging.info(f"Saved variations to: {json_path}")
        # bt.logging.info(f"DataFrame shape: {result_df.shape} with {max_variations} variation columns")
    
    def Clean_extra(self, payload: str, comma: bool, line: bool, space: bool, preserve_name_spaces: bool = False) -> str:
        """
        Clean the LLM output by removing unwanted characters.
        
        Args:
            payload: The text to clean
            comma: Whether to remove commas
            line: Whether to remove newlines
            space: Whether to remove spaces
            preserve_name_spaces: Whether to preserve spaces between names (for multi-part names)
        """
        # Remove punctuation and quotes
        payload = payload.replace(".", "")
        payload = payload.replace('"', "")
        payload = payload.replace("'", "")
        payload = payload.replace("-", "")
        payload = payload.replace("and ", "")
        
        # Handle spaces based on preservation flag
        if space:
            if preserve_name_spaces:
                # Replace multiple spaces with single space
                while "  " in payload:
                    payload = payload.replace("  ", " ")
            else:
                # Original behavior - remove all spaces
                payload = payload.replace(" ", "")
        
        if comma:
            payload = payload.replace(",", "")
        if line:
            payload = payload.replace("\\n", "")
        
        return payload.strip()

    def validate_variation(self, name: str, seed: str, is_multipart_name: bool) -> str:
        """
        Helper function to validate if a variation matches the seed name structure.

        Args:
            name: The variation to validate
            seed: The original seed name
            is_multipart_name: Whether the seed is a multi-part name

        Returns:
            str: The validated and cleaned variation, or np.nan if invalid
        """
        name = (name or "").strip()
        if not name or name.isspace():
            return np.nan

        # Handle cases with colons (e.g., "Here are variations: Name")
        if ":" in name:
            name = name.split(":")[-1].strip()

        name = self._normalize_name_text(name)
        if not name:
            return np.nan

        # Guard against malformed long strings / payload leaks.
        if len(name) > max(2 * len(seed), 64):
            return np.nan

        name_parts = name.split()
        if is_multipart_name:
            if len(name_parts) < 2:
                bt.logging.warning(f"Skipping single-part variation '{name}' for multi-part seed '{seed}'")
                return np.nan
        else:
            if len(name_parts) > 1:
                bt.logging.warning(f"Skipping multi-part variation '{name}' for single-part seed '{seed}'")
                return np.nan

        if not self._is_name_like_candidate(name, seed, is_multipart_name):
            return np.nan

        return name

    def Process_function(self, string: str, debug: bool) -> Tuple[str, str, List[str], Optional[str]]:
        """
        Process the LLM response to extract the seed name and variations.
        
        This function parses the LLM response to extract:
        1. The original seed name
        2. The list of name variations
        
        It handles different response formats from LLMs:
        - Comma-separated lists (preferred format)
        - Line-separated lists
        - Space-separated lists with numbering
        
        The function ensures variations match the structure of the seed name:
        - Single-part seed names (e.g., "John") only get single-part variations
        - Multi-part seed names (e.g., "John Smith") only get multi-part variations
        
        Args:
            string: The LLM response in the format:
                   "---\nQuery-{name}\n---\n{response}"
            debug: Whether to return debug information
            
        Returns:
            Tuple containing:
            - seed_name: The original name
            - processing_method: The method used to process the response (r1, r2, or r3)
            - variations_list: The list of extracted variations
            - payload: (if debug=True) The processed payload
        """
        # Split the response by "---" to extract the query and response parts
        splits = string.split('---')
        
        # Extract and analyze the seed name structure
        seed = splits[1].split("-")[1].replace(".", "").replace(",", "").replace("'", "")
        seed_parts = seed.split()
        is_multipart_name = len(seed_parts) > 1
        seed = self.Clean_extra(seed, True, True, True, preserve_name_spaces=is_multipart_name)
        
        bt.logging.info(f"Processing seed name: '{seed}' (multipart: {is_multipart_name})")
        
        # Extract the response payload
        payload = splits[-1]
        
        # Case 1: Comma-separated list (preferred format)
        if len(payload.split(",")) > 3:  # Check if we have at least 3 commas
            # Clean the payload but keep commas for splitting
            payload = self.Clean_extra(payload, False, True, True, preserve_name_spaces=is_multipart_name)
            
            # Remove numbering prefixes
            for num in range(10):
                payload = payload.replace(str(num), "")
            
            # Split by comma and process each variation
            variations = []
            for name in payload.split(","):
                cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                if not pd.isna(cleaned_var):
                    variations.append(cleaned_var)
            
            if debug:
                return seed, "r1", variations, payload
            return seed, "r1", variations
        
        # Case 2 & 3: Non-comma separated formats
        else:
            # Case 2: Line-separated list
            len_ans = len(payload.split("\\n"))
            if len_ans > 2:  # Multiple lines indicate line-separated format
                # Clean the payload but preserve newlines for splitting
                payload = self.Clean_extra(payload, True, False, True, preserve_name_spaces=is_multipart_name)
                
                # Remove numbering prefixes
                for num in range(10):
                    payload = payload.replace(str(num), "")
                
                # Process line-separated variations
                variations = []
                for name in payload.split("\\n"):
                    cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                    if not pd.isna(cleaned_var):
                        variations.append(cleaned_var)
            
                if debug:
                    return seed, "r2", variations, payload
                return seed, "r2", variations
            
            # Case 3: Space-separated list
            else:
                # Clean the payload but preserve spaces for multi-part names
                payload = self.Clean_extra(payload, True, True, False, preserve_name_spaces=is_multipart_name)
                
                # Remove numbering prefixes
                for num in range(10):
                    payload = payload.replace(str(num), "")
                
                variations = []
                if is_multipart_name:
                    # For multi-part names, we need to carefully group the parts
                    current_variation = []
                    parts = payload.split()
                    
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        
                        if ":" in part:  # New variation starts after colon
                            if current_variation:
                                # Process completed variation
                                cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                                if not pd.isna(cleaned_var):
                                    variations.append(cleaned_var)
                            current_variation = [part.split(":")[-1].strip()]
                        else:
                            current_variation.append(part)
                            # Check if we have collected enough parts for a complete name
                            if len(current_variation) == len(seed_parts):
                                cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                                if not pd.isna(cleaned_var):
                                    variations.append(cleaned_var)
                                current_variation = []
                
                    # Handle any remaining parts
                    if current_variation:
                        cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                        if not pd.isna(cleaned_var):
                            variations.append(cleaned_var)
                else:
                    # For single-part names, simple space splitting is sufficient
                    for name in payload.split():
                        cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                        if not pd.isna(cleaned_var):
                            variations.append(cleaned_var)
                
                if debug:
                    return seed, "r3", variations, payload
                return seed, "r3", variations

    async def blacklist(
        self, synapse: IdentitySynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored.
        
        This function implements security checks to ensure that only authorized
        validators can query this miner. It verifies:
        1. Whether the request has a valid dendrite and hotkey
        2. Whether the hotkey is one of the ones on the white list
        
        Args:
            synapse: A IdentitySynapse object constructed from the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: Whether the request should be blacklisted
                - str: The reason for the decision
        """
        # Check if the request has a valid dendrite and hotkey
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        if not self._register_dynamic_validator(synapse.dendrite.hotkey):
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # If all checks pass, allow the request
        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: IdentitySynapse) -> float:
        """
        The priority function determines the order in which requests are handled.
        
        This function assigns a priority to each request based on the stake of the
        calling entity. Requests with higher priority are processed first, which
        ensures that validators with more stake get faster responses.
        
        Args:
            synapse: The IdentitySynapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.
                  Higher values indicate higher priority.
        """
        # Check if the request has a valid dendrite and hotkey
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # Get the UID of the caller
        try:
            caller_uid = self.metagraph.hotkeys.index(
                synapse.dendrite.hotkey
            )
        except ValueError:
            bt.logging.warning(
                f"Priority fallback: hotkey {synapse.dendrite.hotkey} not found in metagraph"
            )
            return 0.0
        
        # Use the stake as the priority
        # Higher stake = higher priority
        priority = float(
            self.metagraph.S[caller_uid]
        )
        
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"----------------------------------Name Variation Miner running... {time.time()}")
            time.sleep(30)
