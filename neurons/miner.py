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
import math
from difflib import SequenceMatcher
import bittensor as bt
import ollama
import pandas as pd
import os
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict, OrderedDict
from tqdm import tqdm
from PIL import Image

try:
    import Levenshtein
except Exception:
    Levenshtein = None

try:
    import jellyfish
except Exception:
    jellyfish = None

try:
    import geonamescache
except ImportError:
    geonamescache = None

try:
    from MIID.validator.rule_evaluator import evaluate_rule_compliance
except Exception:
    evaluate_rule_compliance = None

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
from MIID.miner.pipeline_observability import log_phase4_json
from MIID.miner.request_spec import compile_phase4_variation_requests, log_request_spec_errors
from MIID.miner.drand_encrypt import encrypt_image_for_drand, is_timelock_available
from MIID.miner.phase4_submission import (
    build_submission_manifest,
    extract_submission_final_score,
    log_submission_failure,
    verify_pre_upload,
    verify_submission_signature,
    write_submission_manifest_debug,
)
from MIID.miner.s3_upload import upload_to_s3
from MIID.miner import kav_helpers


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

        uid = self._get_validator_uid(hotkey)
        if uid is None:
            # A validator permit can change while the miner is already running.
            # Refresh once on an unknown hotkey before rejecting the request.
            try:
                self.resync_metagraph()
            except Exception as e:
                bt.logging.debug(f"Metagraph refresh on unknown hotkey failed: {e}")
            uid = self._get_validator_uid(hotkey)
            if uid is None:
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

    def _get_validator_uid(self, hotkey: str) -> Optional[int]:
        try:
            return self.metagraph.hotkeys.index(hotkey)
        except ValueError:
            return None

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
        if re.search(r"\bswap[_\s-]*random[_\s-]*letter\b", lowered):
            add_rule("swap_random_letter")
        if re.search(r"\bdelete[_\s-]*random[_\s-]*letter\b", lowered) or ("delete" in lowered and "letter" in lowered):
            add_rule("delete_random_letter")
        if re.search(r"\bremove[_\s-]*random[_\s-]*vowel\b", lowered) or ("remove" in lowered and "vowel" in lowered):
            add_rule("remove_random_vowel")
        if re.search(r"\bremove[_\s-]*random[_\s-]*consonant\b", lowered) or ("remove" in lowered and "consonant" in lowered):
            add_rule("remove_random_consonant")
        if re.search(r"\bremove[_\s-]*all[_\s-]*spaces\b", lowered) or "remove all spaces" in lowered:
            add_rule("remove_all_spaces")
        if re.search(r"\bduplicate[_\s-]*random[_\s-]*letter\b", lowered) or ("duplicate" in lowered and "letter" in lowered):
            add_rule("duplicate_random_letter_as_double_letter")
        if re.search(r"\breplace[_\s-]*random[_\s-]*vowel(?:s)?\b", lowered) or "replace random vowels" in lowered or "different vowels" in lowered:
            add_rule("replace_random_vowel_with_random_vowel")
        if re.search(r"\breplace[_\s-]*random[_\s-]*consonant(?:s)?\b", lowered) or "replace random consonants" in lowered or "different consonants" in lowered:
            add_rule("replace_random_consonant_with_random_consonant")
        if re.search(r"\bconvert(?:\s+name)?\s+to\s+initials\b", lowered) or "convert name to initials" in lowered or "initials" in lowered:
            add_rule("shorten_name_to_initials")
        if re.search(r"\bfirst[_\s-]*name[_\s-]*initial\b", lowered) or "first name to initial" in lowered:
            add_rule("initial_only_first_name")
        if re.search(r"\babbreviat|abbreviate_name_parts|shorten[_\s-]*name[_\s-]*to[_\s-]*abbreviations\b", lowered):
            add_rule("shorten_name_to_abbreviations")
        if re.search(r"\bname[_\s-]*parts[_\s-]*permut", lowered) or ("permute" in lowered and "name" in lowered):
            add_rule("name_parts_permutations")

        # Additional rules supported by validator `rule_evaluator.py`.
        if "replace spaces" in lowered and "special" in lowered and "character" in lowered:
            add_rule("replace_spaces_with_random_special_characters")
        if "double letters" in lowered and "single letter" in lowered:
            add_rule("replace_double_letters_with_single_letter")
        if "swap" in lowered and "adjacent conson" in lowered:
            add_rule("swap_adjacent_consonants")
        if "title prefix" in lowered and ("mr" in lowered or "dr" in lowered or "title" in lowered):
            add_rule("add_random_leading_title")
        if "title suffix" in lowered and ("jr" in lowered or "phd" in lowered or "md" in lowered or "title" in lowered):
            add_rule("add_random_trailing_title")
        if ("remove" in lowered or "delete" in lowered) and "special" in lowered and "character" in lowered:
            add_rule("remove_random_special_character")
        if ("remove" in lowered or "delete" in lowered) and "title" in lowered:
            add_rule("remove_title")

        # Also handle cases where the validator includes the explicit snake_case rule keys.
        explicit_rule_keys = {
            "replace_spaces_with_random_special_characters",
            "replace_double_letters_with_single_letter",
            "swap_adjacent_consonants",
            "remove_random_special_character",
            "remove_title",
            "add_random_leading_title",
            "add_random_trailing_title",
        }
        for explicit_key in explicit_rule_keys:
            if explicit_key in lowered:
                add_rule(explicit_key)

        if not rules and ("rule-based" in lowered or "transform" in lowered):
            return [
                "insert_random_letter",
                "remove_random_vowel",
                "shorten_name_to_abbreviations",
            ]
        return rules

    @staticmethod
    def _normalize_similarity_targets(raw_targets: Optional[Dict[str, float]], default_level: str = "Medium") -> Dict[str, float]:
        levels = ("Light", "Medium", "Far")
        targets = raw_targets or {}
        cleaned: Dict[str, float] = {}
        total = 0.0
        for level in levels:
            try:
                value = float(targets.get(level, 0.0) or 0.0)
            except Exception:
                value = 0.0
            if value > 0:
                cleaned[level] = value
                total += value
        if total <= 0:
            return {default_level: 1.0}
        return {level: (cleaned[level] / total) for level in cleaned}

    def _extract_similarity_targets(
        self,
        query_template: str,
        similarity_kind: str,
        default: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Parse Light/Medium/Far target percentages from validator query text."""
        default_targets = default or {"Medium": 1.0}
        text = str(query_template or "")
        lowered = text.lower()
        kind = str(similarity_kind or "").strip().lower()
        if not text.strip() or kind not in {"phonetic", "orthographic"}:
            return self._normalize_similarity_targets(default_targets)

        anchor = f"{kind} similarity"
        windows: List[str] = []
        stop_markers = [
            "phonetic similarity",
            "orthographic similarity",
            "approximately",
            "additionally",
            "[additional context]",
            "[image variation requirements]",
        ]

        for match in re.finditer(re.escape(anchor), lowered):
            start_idx = match.start()
            end_idx = min(len(text), match.end() + 260)
            for marker in stop_markers:
                if marker == anchor:
                    continue
                marker_idx = lowered.find(marker, match.end())
                if marker_idx != -1:
                    end_idx = min(end_idx, marker_idx)
            windows.append(text[start_idx:end_idx])

        if not windows:
            windows = [text]

        extracted: Dict[str, float] = {}
        pair_pattern = re.compile(
            r"(\d{1,3})%\s*(?:[^.;,\n]{0,40}?)\b(light|medium|far)\b",
            flags=re.IGNORECASE,
        )

        for window in windows:
            for pct_text, level_text in pair_pattern.findall(window):
                try:
                    pct = float(pct_text)
                except Exception:
                    continue
                if 0.0 <= pct <= 100.0:
                    extracted[level_text.capitalize()] = pct / 100.0
            if extracted:
                break

        return self._normalize_similarity_targets(extracted or default_targets)

    @staticmethod
    def _bucket_counts_from_targets(total_count: int, targets: Dict[str, float]) -> Dict[str, int]:
        levels = ["Light", "Medium", "Far"]
        if total_count <= 0:
            return {level: 0 for level in levels}

        normalized = Miner._normalize_similarity_targets(targets)
        raw_counts = {level: normalized.get(level, 0.0) * total_count for level in levels}
        counts = {level: int(math.floor(raw_counts[level])) for level in levels}
        assigned = sum(counts.values())
        ordering = sorted(
            levels,
            key=lambda level: (raw_counts[level] - counts[level], normalized.get(level, 0.0)),
            reverse=True,
        )
        order_idx = 0
        while assigned < total_count and ordering:
            level = ordering[order_idx % len(ordering)]
            counts[level] += 1
            assigned += 1
            order_idx += 1
        return counts

    @staticmethod
    def _score_to_level(score: Optional[float], boundaries: Dict[str, Tuple[float, float]]) -> Optional[str]:
        if score is None:
            return None
        for level, (lower, upper) in boundaries.items():
            if lower <= score <= upper:
                return level
        return None

    @staticmethod
    def _validator_phonetic_similarity(original_name: str, variation: str) -> float:
        if not original_name or not variation:
            return 0.0
        if jellyfish is None:
            return float(SequenceMatcher(None, original_name.casefold(), variation.casefold()).ratio())

        algorithms = {
            "soundex": lambda x, y: jellyfish.soundex(x) == jellyfish.soundex(y),
            "metaphone": lambda x, y: jellyfish.metaphone(x) == jellyfish.metaphone(y),
            "nysiis": lambda x, y: jellyfish.nysiis(x) == jellyfish.nysiis(y),
        }
        rng = random.Random(hash(original_name) % 10000)
        selected = rng.sample(list(algorithms.keys()), k=min(3, len(algorithms)))
        weights = [rng.random() for _ in selected]
        total_weight = sum(weights) or 1.0
        normalized_weights = [weight / total_weight for weight in weights]
        score = sum(
            float(algorithms[algo](original_name, variation)) * weight
            for algo, weight in zip(selected, normalized_weights)
        )
        return float(score)

    @staticmethod
    def _validator_orthographic_similarity(original_name: str, variation: str) -> float:
        if not original_name or not variation:
            return 0.0
        if Levenshtein is None:
            return float(SequenceMatcher(None, original_name.casefold(), variation.casefold()).ratio())
        try:
            distance = Levenshtein.distance(original_name, variation)
        except Exception:
            return float(SequenceMatcher(None, original_name.casefold(), variation.casefold()).ratio())
        max_len = max(len(original_name), len(variation), 1)
        return max(0.0, 1.0 - (distance / max_len))

    @staticmethod
    def _has_excessive_letter_repetition(text: str, max_repetition: int = 2) -> bool:
        if not text:
            return False
        pattern = r'(.)\1{' + str(max_repetition) + r',}'
        return re.search(pattern, str(text), flags=re.IGNORECASE) is not None

    def _query_allows_structure_breaks(self, seed_name: str, requested_rules: Optional[List[str]]) -> bool:
        if len(str(seed_name or "").split()) < 2:
            return False
        rule_set = set(requested_rules or [])
        # These rules intentionally change word-boundary structure (spaces -> non-spaces / initials shortening).
        return any(
            rule in rule_set
            for rule in {
                "remove_all_spaces",
                "shorten_name_to_initials",
                "replace_spaces_with_random_special_characters",
            }
        )

    def _candidate_rule_hits(self, seed_name: str, candidates: List[str], requested_rules: List[str]) -> Dict[str, List[str]]:
        if not candidates or not requested_rules or evaluate_rule_compliance is None:
            return {}
        try:
            compliant_by_rule, _ = evaluate_rule_compliance(
                seed_name.lower(),
                [candidate.lower() for candidate in candidates],
                requested_rules,
            )
        except Exception as e:
            bt.logging.debug(f"Validator rule evaluator unavailable for local KAV scoring: {e}")
            return {}

        reverse: Dict[str, List[str]] = defaultdict(list)
        for rule_name, compliant_variations in compliant_by_rule.items():
            for variation in compliant_variations:
                if rule_name not in reverse[variation]:
                    reverse[variation].append(rule_name)
        return reverse

    def _score_candidate_for_validator(
        self,
        seed_name: str,
        candidate: str,
        candidate_rule_hits: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        candidate_norm = self._normalize_name_text(candidate)
        seed_norm = self._normalize_name_text(seed_name)
        seed_parts = seed_norm.split()
        candidate_parts = candidate_norm.split()
        is_multipart = len(seed_parts) > 1
        exact_structure = len(candidate_parts) == len(seed_parts)
        candidate_key = candidate_norm.lower()
        rule_hits = list(candidate_rule_hits.get(candidate_key, []))

        phonetic_boundaries = {
            "Light": (0.80, 1.00),
            "Medium": (0.60, 0.79),
            "Far": (0.30, 0.59),
        }
        orthographic_boundaries = {
            "Light": (0.70, 1.00),
            "Medium": (0.50, 0.69),
            "Far": (0.20, 0.49),
        }

        if is_multipart and len(candidate_parts) >= 2:
            seed_first = seed_parts[0]
            seed_last = seed_parts[-1]
            candidate_first = candidate_parts[0]
            candidate_last = candidate_parts[-1]
            phonetic_score = (
                0.3 * self._validator_phonetic_similarity(seed_first, candidate_first)
                + 0.7 * self._validator_phonetic_similarity(seed_last, candidate_last)
            )
            orthographic_score = (
                0.3 * self._validator_orthographic_similarity(seed_first, candidate_first)
                + 0.7 * self._validator_orthographic_similarity(seed_last, candidate_last)
            )
        else:
            phonetic_score = self._validator_phonetic_similarity(seed_norm.replace(" ", ""), candidate_norm.replace(" ", ""))
            orthographic_score = self._validator_orthographic_similarity(seed_norm.replace(" ", ""), candidate_norm.replace(" ", ""))

        length_ratio = min(
            len(candidate_norm.replace(" ", "")) / max(len(seed_norm.replace(" ", "")), 1),
            len(seed_norm.replace(" ", "")) / max(len(candidate_norm.replace(" ", "")), 1),
        )
        length_ratio = max(0.0, min(1.0, length_ratio))

        structure_score = 1.0 if exact_structure else (0.92 if rule_hits else 0.72)
        base_score = (
            0.45 * phonetic_score
            + 0.35 * orthographic_score
            + 0.10 * length_ratio
            + 0.10 * structure_score
        )
        if self._has_excessive_letter_repetition(candidate_norm, max_repetition=2):
            base_score *= 0.8

        return {
            "name": candidate_norm,
            "key": candidate_key,
            "rule_hits": rule_hits,
            "phonetic_score": float(phonetic_score),
            "orthographic_score": float(orthographic_score),
            "phonetic_level": self._score_to_level(phonetic_score, phonetic_boundaries),
            "orthographic_level": self._score_to_level(orthographic_score, orthographic_boundaries),
            "base_score": float(base_score),
            "exact_structure": exact_structure,
        }

    def _is_near_duplicate_candidate(self, candidate: str, selected_names: List[str]) -> bool:
        candidate_key = self._canonical_name_key(candidate)
        if not candidate_key:
            return True
        for existing in selected_names:
            existing_key = self._canonical_name_key(existing)
            if not existing_key:
                continue
            combined_similarity = (
                self._validator_phonetic_similarity(candidate_key, existing_key) * 0.7
                + self._validator_orthographic_similarity(candidate_key, existing_key) * 0.3
            )
            if combined_similarity > 0.99:
                return True
        return False

    def _select_validator_aligned_names(
        self,
        seed_name: str,
        candidates: List[str],
        target_count: int,
        query_template: str = "",
    ) -> List[str]:
        if target_count <= 0 or not candidates:
            return []
        if len(candidates) <= target_count:
            return candidates[:target_count]

        requested_rules = self._extract_requested_transformations(query_template)
        phonetic_targets = self._extract_similarity_targets(query_template, "phonetic", default={"Medium": 1.0})
        orthographic_targets = self._extract_similarity_targets(query_template, "orthographic", default={"Medium": 1.0})
        rule_percentage = self._extract_rule_percentage(query_template, default=0.0)
        candidate_rule_hits = self._candidate_rule_hits(seed_name, candidates, requested_rules)

        infos = [
            self._score_candidate_for_validator(seed_name, candidate, candidate_rule_hits)
            for candidate in candidates
        ]
        infos = [info for info in infos if info.get("name")]
        infos.sort(key=lambda item: item.get("base_score", 0.0), reverse=True)

        available_rule_candidates = sum(1 for info in infos if info.get("rule_hits"))
        rule_target_count = 0
        if requested_rules and rule_percentage > 0 and available_rule_candidates > 0:
            rule_target_count = int(round(target_count * rule_percentage))
            if target_count > 1:
                rule_target_count = min(target_count - 1, max(1, rule_target_count))
            else:
                rule_target_count = 1
            rule_target_count = min(rule_target_count, available_rule_candidates)

        non_rule_target = max(0, target_count - rule_target_count)
        desired_phonetic = self._bucket_counts_from_targets(non_rule_target, phonetic_targets)
        desired_orthographic = self._bucket_counts_from_targets(non_rule_target, orthographic_targets)

        selected_infos: List[Dict[str, Any]] = []
        selected_names: List[str] = []
        selected_phonetic = Counter()
        selected_orthographic = Counter()
        covered_rules = set()

        def add_selected(info: Dict[str, Any]) -> None:
            selected_infos.append(info)
            selected_names.append(info["name"])
            if info.get("phonetic_level"):
                selected_phonetic[info["phonetic_level"]] += 1
            if info.get("orthographic_level"):
                selected_orthographic[info["orthographic_level"]] += 1
            covered_rules.update(info.get("rule_hits", []))

        def candidate_usable(info: Dict[str, Any]) -> bool:
            return not self._is_near_duplicate_candidate(info["name"], selected_names)

        while len(selected_infos) < non_rule_target:
            pool = [info for info in infos if info not in selected_infos and not info.get("rule_hits")]
            if not pool:
                pool = [info for info in infos if info not in selected_infos]
            best_info = None
            best_score = None
            for info in pool:
                if not candidate_usable(info):
                    continue
                score = info.get("base_score", 0.0) * 4.0
                phonetic_level = info.get("phonetic_level")
                orthographic_level = info.get("orthographic_level")
                if phonetic_level and selected_phonetic[phonetic_level] < desired_phonetic.get(phonetic_level, 0):
                    score += 1.6
                elif phonetic_level is None:
                    score -= 0.2
                if orthographic_level and selected_orthographic[orthographic_level] < desired_orthographic.get(orthographic_level, 0):
                    score += 1.6
                elif orthographic_level is None:
                    score -= 0.2
                if info.get("exact_structure"):
                    score += 0.3
                if info.get("rule_hits"):
                    score -= 0.8
                if best_info is None or score > best_score:
                    best_info = info
                    best_score = score
            if best_info is None:
                break
            add_selected(best_info)

        while len([info for info in selected_infos if info.get("rule_hits")]) < rule_target_count:
            pool = [info for info in infos if info not in selected_infos and info.get("rule_hits")]
            best_info = None
            best_score = None
            for info in pool:
                if not candidate_usable(info):
                    continue
                new_rule_coverage = len(set(info.get("rule_hits", [])) - covered_rules)
                score = info.get("base_score", 0.0) * 3.5 + (new_rule_coverage * 2.0)
                if info.get("exact_structure"):
                    score += 0.15
                if best_info is None or score > best_score:
                    best_info = info
                    best_score = score
            if best_info is None:
                break
            add_selected(best_info)

        for info in infos:
            if len(selected_infos) >= target_count:
                break
            if info in selected_infos or not candidate_usable(info):
                continue
            add_selected(info)

        bt.logging.info(
            f"KAV validator reranker for '{seed_name}': pool={len(candidates)}, selected={len(selected_infos)}, "
            f"rule_target={rule_target_count}, phonetic_targets={desired_phonetic}, orthographic_targets={desired_orthographic}"
        )
        names = [info["name"] for info in selected_infos[:target_count]]
        if len(names) < target_count:
            seen_norm = {n.lower().strip() for n in names if n}
            for info in infos:
                if len(names) >= target_count:
                    break
                n = (info.get("name") or "").strip()
                if not n:
                    continue
                if n.lower() in seen_norm:
                    continue
                seen_norm.add(n.lower())
                names.append(n)
        if len(names) < target_count:
            for info in infos:
                if len(names) >= target_count:
                    break
                n = (info.get("name") or "").strip()
                if not n:
                    continue
                names.append(n)
        # Pool may be smaller than target_count: repeat best available names until full.
        if len(names) < target_count and infos:
            idx = 0
            max_iters = max(target_count * max(len(infos), 1) * 2, 64)
            it = 0
            while len(names) < target_count and it < max_iters:
                it += 1
                n = (infos[idx % len(infos)].get("name") or "").strip()
                idx += 1
                if n:
                    names.append(n)
        if len(names) < target_count:
            bt.logging.warning(
                f"KAV name selector: still short {len(names)}/{target_count} for seed={seed_name!r}; "
                "pad layer must complete the count."
            )
        return names[:target_count]

    @staticmethod
    def _normalize_name_text(raw_name: str) -> str:
        """Unicode-normalize and keep name-like text only."""
        normalized = unicodedata.normalize("NFKC", str(raw_name or ""))
        normalized = normalized.replace("’", "'").replace("`", "'").replace("´", "'")
        # Keep punctuation/special chars needed by validator rule checks (titles, space->special, special char removal).
        validator_special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        allowed = {"'", "-"} | set(validator_special_chars)
        cleaned = "".join(
            ch if (ch.isalpha() or ch.isspace() or ch in allowed) else " "
            for ch in normalized
        )
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

    def _is_name_like_candidate(
        self,
        candidate: str,
        seed_name: str,
        is_multipart: bool,
        requested_rules: Optional[List[str]] = None,
    ) -> bool:
        candidate_norm = self._normalize_name_text(candidate)
        seed_norm = self._normalize_name_text(seed_name)
        if not candidate_norm or not seed_norm:
            return False

        if self._has_excessive_letter_repetition(candidate_norm, max_repetition=2):
            return False

        candidate_key = self._canonical_name_key(candidate_norm)
        seed_key = self._canonical_name_key(seed_norm)
        if not candidate_key or candidate_key == seed_key:
            return False

        candidate_parts = candidate_norm.split()
        seed_parts = seed_norm.split()
        requested_rule_set = set(requested_rules or [])
        structure_break_allowed = False

        if is_multipart:
            if len(candidate_parts) < 2:
                structure_break_allowed = self._query_allows_structure_breaks(seed_name, list(requested_rule_set))
                if not structure_break_allowed:
                    return False
                compact_candidate = "".join(candidate_parts).casefold()
                compact_seed = "".join(seed_parts).casefold()
                initials = "".join(token[0] for token in seed_parts if token).casefold()
                compact_similarity = SequenceMatcher(None, compact_seed, compact_candidate).ratio()
                if "shorten_name_to_initials" in requested_rule_set and compact_candidate == initials:
                    pass
                elif "remove_all_spaces" in requested_rule_set and compact_candidate == compact_seed:
                    pass
                elif "replace_spaces_with_random_special_characters" in requested_rule_set:
                    # Candidate will contain special characters at space boundaries, which can reduce similarity.
                    # Compare only letter content for the structure gate.
                    filtered_candidate = re.sub(r"[^a-z]", "", compact_candidate)
                    filtered_seed = re.sub(r"[^a-z]", "", compact_seed)
                    filtered_similarity = SequenceMatcher(None, filtered_seed, filtered_candidate).ratio()
                    if filtered_similarity < 0.48:
                        return False
                elif compact_similarity < 0.48:
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
        if is_multipart and not structure_break_allowed and similarity < 0.40:
            return False
        if not is_multipart and similarity < 0.30:
            return False
        if is_multipart and structure_break_allowed:
            compact_similarity = SequenceMatcher(
                None,
                seed_key.replace(" ", ""),
                candidate_key.replace(" ", ""),
            ).ratio()
            initials = "".join(token[0] for token in seed_parts if token).casefold()
            if compact_similarity < 0.45 and candidate_key.replace(" ", "") != initials:
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

    def _clean_name_candidate(
        self,
        raw_name: str,
        seed_name: str,
        is_multipart: bool,
        requested_rules: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Normalize and validate name candidates before scoring."""
        if not raw_name:
            return None
        candidate = self._normalize_name_text(str(raw_name))
        if not candidate:
            return None
        if not self._is_name_like_candidate(
            candidate,
            seed_name,
            is_multipart,
            requested_rules=requested_rules,
        ):
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
        Uses letter-only transforms where possible to avoid non-letter penalties.
        """
        parts = [p for p in str(seed_name).split() if p.strip()]
        if not parts:
            return None

        mutable_parts = ["".join(ch for ch in p if ch.isalpha()) for p in parts]
        mutable_parts = [p if p else raw for p, raw in zip(mutable_parts, parts)]
        target_idx = attempt % len(mutable_parts)
        base_part = mutable_parts[target_idx]
        if len(base_part) < 2 and rule not in {
            "remove_all_spaces",
            "shorten_name_to_initials",
            "initial_only_first_name",
            "replace_spaces_with_random_special_characters",
            "remove_random_special_character",
            "remove_title",
            "add_random_leading_title",
            "add_random_trailing_title",
        }:
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

        if rule == "swap_adjacent_consonants":
            # Swap one eligible adjacent consonant pair within the chosen part.
            if len(base_part) < 2:
                return None
            vowels = set("aeiou")
            lower = base_part.lower()
            eligible = [
                i
                for i in range(len(lower) - 1)
                if lower[i].isalpha()
                and lower[i + 1].isalpha()
                and (lower[i] not in vowels)
                and (lower[i + 1] not in vowels)
                and (lower[i] != lower[i + 1])
            ]
            if not eligible:
                return None
            i = eligible[attempt % len(eligible)]
            chars = list(base_part)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
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
            mutated = base_part[:rm_at] + base_part[rm_at + 1 :]
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

        if rule == "replace_double_letters_with_single_letter":
            # Delete one character from a double-letter run.
            if len(base_part) < 2:
                return None
            lower = base_part.lower()
            eligible = [i for i in range(len(lower) - 1) if lower[i].isalpha() and lower[i] == lower[i + 1]]
            if not eligible:
                return None
            i = eligible[attempt % len(eligible)]
            mutated = base_part[:i] + base_part[i + 1 :]
            if len(mutated) < 1:
                return None
            result = mutable_parts.copy()
            result[target_idx] = mutated
            return " ".join(result)

        if rule == "replace_random_vowel_with_random_vowel":
            vowel_idx = [i for i, ch in enumerate(base_part) if ch.lower() in "aeiou"]
            if not vowel_idx:
                return None
            replace_at = vowel_idx[attempt % len(vowel_idx)]
            vowels = "aeiou"
            current = base_part[replace_at].lower()
            replacements = [v for v in vowels if v != current]
            replacement = replacements[(attempt + replace_at) % len(replacements)]
            mutated = base_part[:replace_at] + replacement + base_part[replace_at + 1 :]
            result = mutable_parts.copy()
            result[target_idx] = self._apply_word_case(base_part, mutated)
            return " ".join(result)

        if rule == "replace_random_consonant_with_random_consonant":
            consonant_idx = [i for i, ch in enumerate(base_part) if ch.isalpha() and ch.lower() not in "aeiou"]
            if not consonant_idx:
                return None
            replace_at = consonant_idx[attempt % len(consonant_idx)]
            consonants = "bcdfghjklmnpqrstvwxyz"
            current = base_part[replace_at].lower()
            replacements = [c for c in consonants if c != current]
            replacement = replacements[(attempt + replace_at) % len(replacements)]
            mutated = base_part[:replace_at] + replacement + base_part[replace_at + 1 :]
            result = mutable_parts.copy()
            result[target_idx] = self._apply_word_case(base_part, mutated)
            return " ".join(result)

        if rule == "remove_all_spaces":
            if len(mutable_parts) < 2:
                return None
            return "".join(mutable_parts)

        if rule == "replace_spaces_with_random_special_characters":
            seed_str = str(seed_name).strip()
            special_pool = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if " " not in seed_str:
                return None
            out_chars: List[str] = []
            space_i = 0
            for ch in seed_str:
                if ch == " ":
                    out_chars.append(special_pool[(attempt + space_i) % len(special_pool)])
                    space_i += 1
                else:
                    out_chars.append(ch)
            return "".join(out_chars)

        if rule == "shorten_name_to_initials":
            if len(mutable_parts) < 2:
                short_len = max(2, min(len(base_part) - 1, len(base_part) // 2 + 1))
                return mutable_parts[0][:short_len]
            return "".join(word[0] for word in mutable_parts if word)

        if rule == "initial_only_first_name" and len(mutable_parts) >= 2:
            return " ".join([mutable_parts[0][0], *mutable_parts[1:]])

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

        if rule == "remove_random_special_character":
            seed_str = str(seed_name).strip()
            special_chars = set("!@#$%^&*()_+-=[]{}|;:,.<>?")
            positions = [i for i, ch in enumerate(seed_str) if ch in special_chars]
            if not positions:
                return None
            i = positions[attempt % len(positions)]
            return seed_str[:i] + seed_str[i + 1 :]

        if rule == "remove_title":
            seed_str = str(seed_name).strip()
            seed_lower = seed_str.lower()
            titles = ["Mr.", "Mrs.", "Ms.", "Mr", "Mrs", "Ms", "Miss", "Dr.", "Dr", "Prof.", "Prof", "Sir", "Lady", "Lord", "Dame", "Master", "Mistress", "Rev.", "Hon.", "Capt.", "Col.", "Lt.", "Sgt.", "Maj."]
            # Deterministically pick which matching title to remove.
            for j in range(len(titles)):
                title = titles[(attempt + j) % len(titles)]
                if seed_lower.startswith(title.lower() + " "):
                    return seed_str[len(title) + 1 :]
            return None

        if rule == "add_random_leading_title":
            titles = ["Mr.", "Mrs.", "Ms.", "Mr", "Mrs", "Ms", "Miss", "Dr.", "Dr", "Prof.", "Prof", "Sir", "Lady", "Lord", "Dame", "Master", "Mistress", "Rev.", "Hon.", "Capt.", "Col.", "Lt.", "Sgt.", "Maj."]
            title = titles[attempt % len(titles)]
            return f"{title} {str(seed_name).strip()}"

        if rule == "add_random_trailing_title":
            suffixes = ["Jr.", "Sr.", "III", "IV", "V", "PhD", "MD", "Esq.", "Jr", "Sr"]
            suffix = suffixes[attempt % len(suffixes)]
            return f"{str(seed_name).strip()} {suffix}"

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
            cleaned = self._clean_name_candidate(
                raw_candidate or "",
                seed_name,
                is_multipart,
                requested_rules=rules,
            )
            attempt += 1
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cleaned)
        return candidates

    def _build_name_fallback_candidates(
        self,
        seed_name: str,
        is_multipart: bool,
        requested_rules: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate structured fallback names when LLM output is too short."""
        seed_parts = seed_name.split()
        if not seed_parts:
            return []
        requested_rule_set = set(requested_rules or [])
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

        if len(first) > 1:
            variants.append(" ".join([f"{first[0]}", *middle, last]))

        if "remove_all_spaces" in requested_rule_set:
            variants.append("".join(seed_parts))
        if "shorten_name_to_initials" in requested_rule_set:
            variants.append("".join(part[0] for part in seed_parts if part))

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
        requested_rules = self._extract_requested_transformations(query_template)
        cleaned: List[str] = []
        seen = set()
        pool_limit = max(24, target_count * 4)

        def add_candidate(raw_value: str) -> None:
            candidate = self._clean_name_candidate(
                raw_value,
                seed_name,
                is_multipart,
                requested_rules=requested_rules,
            )
            if not candidate:
                return
            key = candidate.lower()
            if key in seen:
                return
            seen.add(key)
            cleaned.append(candidate)

        for raw in extracted_names:
            add_candidate(raw)
            if len(cleaned) >= pool_limit:
                break

        rule_percentage = self._extract_rule_percentage(query_template, default=0.0)
        if requested_rules and rule_percentage > 0 and len(cleaned) < pool_limit:
            rule_target_count = int(round(target_count * rule_percentage))
            if target_count > 1:
                rule_target_count = min(target_count - 1, max(1, rule_target_count))
            else:
                rule_target_count = 1
            rule_pool_target = min(pool_limit - len(cleaned), max(len(requested_rules) + 2, rule_target_count * 2))
            for candidate in self._build_rule_aware_candidates(seed_name, requested_rules, rule_pool_target):
                add_candidate(candidate)
                if len(cleaned) >= pool_limit:
                    break

        for fallback in self._build_name_fallback_candidates(
            seed_name,
            is_multipart,
            requested_rules=requested_rules,
        ):
            add_candidate(fallback)
            if len(cleaned) >= pool_limit:
                break

        seed_parts = [p for p in seed_name.split() if p.strip()]
        counter = 0
        max_attempts = max(target_count * 20, 60)
        while len(cleaned) < max(target_count, min(pool_limit, target_count + 8)) and counter < max_attempts:
            counter += 1
            if not seed_parts:
                break

            part_idx = counter % len(seed_parts)
            part = seed_parts[part_idx].strip()
            if not part:
                continue

            mode = counter % 4
            if len(part) == 1:
                perturb = part + part
            elif mode == 0:
                swap_at = max(0, min(len(part) - 2, len(part) // 2))
                chars = list(part)
                chars[swap_at], chars[swap_at + 1] = chars[swap_at + 1], chars[swap_at]
                perturb = "".join(chars)
            elif mode == 1:
                dup_at = max(0, min(len(part) - 1, len(part) // 2))
                perturb = part[:dup_at] + part[dup_at] + part[dup_at:]
            elif mode == 2:
                if all(ord(ch) < 128 for ch in part):
                    pool = "aeiouwy"
                    replacement = pool[(counter + part_idx) % len(pool)]
                else:
                    replacement = part[(counter + 1) % len(part)]
                perturb = part[:-1] + replacement
            else:
                if all(ord(ch) < 128 for ch in part):
                    suffix_pool = "rstlnm"
                    perturb = part + suffix_pool[(counter + part_idx) % len(suffix_pool)]
                else:
                    perturb = part + part[-1]

            perturb = self._apply_word_case(part, perturb)
            generated = seed_parts.copy()
            generated[part_idx] = perturb
            add_candidate(" ".join(generated).strip())

        if not cleaned:
            return self._pad_exact_name_rows(seed_name, [], [], target_count, query_template)

        selected = self._select_validator_aligned_names(seed_name, cleaned, target_count, query_template)
        base_order = selected if selected else cleaned
        padded = self._pad_exact_name_rows(seed_name, base_order, cleaned, target_count, query_template)
        if len(padded) < target_count and target_count > 0:
            bt.logging.error(
                f"KAV _ensure_target_name_count: pad short {len(padded)}/{target_count} for {seed_name!r}; forcing."
            )
            padded = self._force_pad_names_to_count(seed_name, padded, target_count)
        return padded[:target_count]

    def _force_pad_names_to_count(
        self,
        seed_name: str,
        partial: List[str],
        target_count: int,
    ) -> List[str]:
        """
        Absolute last resort when selector/pad still return fewer than target_count names.
        Uses letter-only suffixes (no digits) to avoid validator name-token digit rules.
        """
        out = list(partial or [])
        if target_count <= 0:
            return []
        seen = {x.lower().strip() for x in out if x}
        seed_clean = (seed_name or "").strip() or "Name"
        seed_low = seed_clean.lower()
        parts = [p for p in seed_clean.split() if p.strip()] or [seed_clean]
        is_multipart = len(parts) > 1
        k = 0
        while len(out) < target_count:
            k += 1
            suf = "a" * ((k % 12) + 1) + ("b" * (k // 12))
            if is_multipart:
                candidate = " ".join(parts[:-1] + [f"{parts[-1]}{suf}"])
            else:
                candidate = f"{parts[0]}{suf}"
            candidate = self._normalize_name_text(candidate)
            if not candidate:
                candidate = f"{seed_clean}{suf}"
            low = candidate.lower()
            if low in seen or low == seed_low:
                candidate = f"{seed_clean} {suf}"
                candidate = self._normalize_name_text(candidate) or f"{seed_clean} {suf}"
                low = candidate.lower()
            seen.add(low)
            out.append(candidate)
        return out[:target_count]

    def _pad_exact_name_rows(
        self,
        seed_name: str,
        ordered_names: List[str],
        cleaned_pool: List[str],
        target_count: int,
        query_template: str,
    ) -> List[str]:
        """Guarantee exactly target_count unique valid name variations (deterministic padding)."""
        if target_count <= 0:
            return []
        is_multipart = len(seed_name.split()) > 1
        requested_rules = self._extract_requested_transformations(query_template)
        seen: set = set()
        out: List[str] = []

        def push(raw: str) -> bool:
            candidate = self._clean_name_candidate(
                raw, seed_name, is_multipart, requested_rules=requested_rules
            )
            if not candidate:
                return False
            key = candidate.lower()
            if key in seen:
                return False
            if key == seed_name.strip().lower():
                return False
            seen.add(key)
            out.append(candidate)
            return True

        for n in ordered_names:
            if len(out) >= target_count:
                break
            push(n)
        for n in cleaned_pool:
            if len(out) >= target_count:
                break
            push(n)

        seed_parts = [p for p in seed_name.split() if p.strip()]
        attempt = 0
        while len(out) < target_count and attempt < 8000:
            attempt += 1
            if not seed_parts:
                break
            for pi, part in enumerate(seed_parts):
                if len(out) >= target_count:
                    break
                for mut in self._mutate_name_part(part):
                    if len(out) >= target_count:
                        break
                    built = seed_parts.copy()
                    built[pi] = mut
                    push(" ".join(built).strip())
            # extra deterministic perturbations (same machinery as pool expansion)
            part_idx = attempt % max(len(seed_parts), 1)
            part = seed_parts[part_idx]
            mode = attempt % 5
            if len(part) < 2:
                continue
            if mode == 0 and len(part) > 2:
                chars = list(part)
                i = attempt % (len(chars) - 1)
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                perturb = "".join(chars)
            elif mode == 1:
                perturb = part[:-1] if len(part) > 1 else part + "x"
            elif mode == 2:
                perturb = part + ("a" if attempt % 2 == 0 else "e")
            elif mode == 3:
                perturb = part[0] + part[1:-1] + part[-1] if len(part) > 3 else part + "s"
            else:
                perturb = part[: len(part) // 2] + part[len(part) // 2 + 1 :]
            perturb = self._apply_word_case(part, perturb)
            bp = seed_parts.copy()
            bp[part_idx] = perturb
            push(" ".join(bp).strip())

        if len(out) < target_count:
            bt.logging.warning(
                f"KAV name pad: could only produce {len(out)}/{target_count} for seed={seed_name!r}; "
                "using minimal suffix variants."
            )
            pad_i = 0
            while len(out) < target_count and seed_parts:
                pad_i += 1
                sfx = f"X{pad_i}"
                bp = seed_parts[:-1] + [seed_parts[-1] + sfx]
                if is_multipart:
                    push(" ".join(bp))
                else:
                    push(seed_parts[0] + sfx)

        def push_loose(raw: str) -> bool:
            """Last resort: normalized unique string, distinct from seed; skips strict name-like checks."""
            candidate = self._normalize_name_text(str(raw))
            if not candidate:
                return False
            key = candidate.lower()
            if key in seen:
                return False
            if key == seed_name.strip().lower():
                return False
            seen.add(key)
            out.append(candidate)
            return True

        if len(out) < target_count:
            bt.logging.warning(
                f"KAV name pad: permissive fill for seed={seed_name!r} ({len(out)}/{target_count})"
            )
            pad_i = 0
            while len(out) < target_count and seed_parts:
                pad_i += 1
                sfx = f"Y{pad_i}"
                if is_multipart:
                    bp = seed_parts[:-1] + [seed_parts[-1] + sfx]
                    push_loose(" ".join(bp))
                else:
                    push_loose(seed_parts[0] + sfx)
        if len(out) < target_count and not seed_parts:
            base = self._normalize_name_text(seed_name) or "Name"
            idx = 0
            while len(out) < target_count:
                idx += 1
                push_loose(f"{base} Var{idx}")

        if len(out) < target_count:
            out = self._force_pad_names_to_count(seed_name, out, target_count)

        return out[:target_count]

    def _build_dob_variations(self, seed_dob: str, target_count: int) -> List[str]:
        """Deterministic DOB list (no LLM); delegates to kav_helpers."""
        return kav_helpers.build_dob_variations_deterministic(seed_dob, target_count)

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
        """Build address variations: candidate pool + validator-aligned selection (looks_like + region)."""
        if target_count <= 0:
            return []
        country_name, country_code, preferred_city = self._resolve_region_target(seed_address)
        sr = kav_helpers.extract_seed_region(
            seed_address,
            self._country_name_to_code,
            self._country_code_to_name,
            self._country_compact_to_code,
        )
        cc = sr.country_code or country_code
        country_display = self._country_code_to_name.get(cc, sr.country_name) if cc else country_name
        pref_city = sr.city_hint or preferred_city

        def pick_city(i: int) -> str:
            return self._pick_city_for_country(cc, pref_city, i)

        pool_size = max(48, target_count * 6)
        pool = kav_helpers.generate_address_candidate_pool(
            seed_address,
            pool_size,
            country_display,
            pick_city,
            self.STREET_ROOTS,
            self.STREET_SUFFIXES,
            self.DISTRICT_WORDS,
        )
        chosen: List[str] = []
        meta: List[Dict[str, Any]] = []
        try:
            chosen, meta = kav_helpers.select_best_addresses(pool, seed_address, target_count)
        except Exception as e:
            bt.logging.warning(f"KAV address selection failed ({e}); falling back to structured lines.")

        if len(chosen) < target_count:
            extra = kav_helpers.generate_address_candidate_pool(
                f"{seed_address}::extra",
                max(32, (target_count - len(chosen)) * 8),
                country_display,
                pick_city,
                self.STREET_ROOTS,
                self.STREET_SUFFIXES,
                self.DISTRICT_WORDS,
            )
            merged = list(dict.fromkeys(pool + extra))
            try:
                chosen, meta = kav_helpers.select_best_addresses(merged, seed_address, target_count)
            except Exception:
                pass

        seen = set(x.lower() for x in chosen)
        i = len(chosen)
        while len(chosen) < target_count and i < target_count * 25:
            city = pick_city(i)
            line = self._make_address_line(city, country_display, i)
            i += 1
            lk = line.lower().strip()
            if lk in seen:
                continue
            seen.add(lk)
            chosen.append(line)

        if len(chosen) < target_count:
            fb = "120 Central Avenue, District 4, New York 10001, United States"
            bt.logging.warning(
                f"KAV address shortfall: padding {target_count - len(chosen)} with fallback template."
            )
            while len(chosen) < target_count:
                chosen.append(fb)

        if meta:
            bt.logging.debug(
                f"KAV address meta: first={meta[0] if meta else {}}, "
                f"region_ok_rate={sum(1 for m in meta if m.get('region_match'))/max(len(meta),1):.2f}"
            )
        return chosen[:target_count]

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

    def _should_save_seed_images(self) -> bool:
        value = os.environ.get("PHASE4_SAVE_VALIDATOR_SEED_IMAGES", "true").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _seed_image_save_dir(self) -> str:
        configured = (os.environ.get("PHASE4_VALIDATOR_SEED_IMAGE_DIR") or "").strip()
        if configured:
            return configured
        run_dir = (os.environ.get("RUN_DIR") or self.output_path or os.getcwd()).strip()
        return os.path.join(run_dir, "validator_seed_images")

    def _save_validator_seed_image(
        self,
        *,
        run_id: int,
        image_request: Any,
        validator_name: str,
        validator_hotkey: str,
    ) -> Optional[Dict[str, Any]]:
        if not self._should_save_seed_images() or not image_request:
            return None

        base64_image = str(getattr(image_request, "base_image", "") or "")
        if not base64_image:
            return {"saved": False, "error": "missing_base_image"}

        try:
            out_dir = self._seed_image_save_dir()
            os.makedirs(out_dir, exist_ok=True)

            seed_image = decode_base_image(base64_image)
            image_filename = str(getattr(image_request, "image_filename", "") or "seed_image")
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(image_filename)[0]).strip("_") or "seed_image"
            challenge = re.sub(
                r"[^A-Za-z0-9._-]+",
                "_",
                str(getattr(image_request, "challenge_id", "") or ""),
            ).strip("_")
            validator_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", validator_name).strip("_") or "validator"
            suffix_parts = [f"run{int(run_id)}", validator_tag]
            if challenge:
                suffix_parts.append(challenge[:48])
            basename = "__".join([stem] + suffix_parts)

            image_path = os.path.join(out_dir, f"{basename}.png")
            meta_path = os.path.join(out_dir, f"{basename}.json")
            seed_image.save(image_path, format="PNG")

            with open(image_path, "rb") as f:
                image_sha256 = hashlib.sha256(f.read()).hexdigest()

            meta = {
                "saved": True,
                "run_id": int(run_id),
                "validator_name": str(validator_name or ""),
                "validator_hotkey": str(validator_hotkey or ""),
                "challenge_id": str(getattr(image_request, "challenge_id", "") or ""),
                "image_filename": image_filename,
                "saved_image_path": image_path,
                "width": int(seed_image.size[0]),
                "height": int(seed_image.size[1]),
                "png_sha256": image_sha256,
                "base_image_sha256": hashlib.sha256(base64_image.encode("utf-8")).hexdigest(),
                "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            meta["metadata_path"] = meta_path
            bt.logging.info(f"Saved validator seed image to {image_path}")
            return meta
        except Exception as e:
            bt.logging.warning(f"Failed to save validator seed image: {e}")
            return {"saved": False, "error": str(e)}

    def _should_save_dashboard_preview_images(self) -> bool:
        value = os.environ.get("PHASE4_DASHBOARD_SAVE_PREVIEW_IMAGES", "true").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _dashboard_preview_dir(self) -> str:
        configured = (os.environ.get("PHASE4_DASHBOARD_PREVIEW_DIR") or "").strip()
        if configured:
            return configured
        run_dir = (os.environ.get("RUN_DIR") or self.output_path or os.getcwd()).strip()
        return os.path.join(run_dir, "dashboard_preview_images")

    def _save_dashboard_variation_preview(
        self,
        *,
        run_id: int,
        challenge_id: str,
        req_index: int,
        label: str,
        attempt: int,
        selected_image: Any,
        primary_score: float,
        final_score: Optional[float],
        candidate_count: int,
        selected_candidate_index: int,
        image_hash: str,
    ) -> Optional[Dict[str, Any]]:
        if not self._should_save_dashboard_preview_images():
            return None
        if not isinstance(selected_image, Image.Image):
            return None

        try:
            out_dir = self._dashboard_preview_dir()
            os.makedirs(out_dir, exist_ok=True)
            safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "variation"
            safe_challenge = re.sub(r"[^A-Za-z0-9._-]+", "_", str(challenge_id or "")).strip("_")
            basename_parts = [f"run{int(run_id)}"]
            if safe_challenge:
                basename_parts.append(safe_challenge[:48])
            basename_parts.append(f"req{int(req_index):02d}")
            basename_parts.append(safe_label)
            basename_parts.append(f"attempt{int(attempt)}")
            basename = "__".join(basename_parts)

            image_path = os.path.join(out_dir, f"{basename}.png")
            meta_path = os.path.join(out_dir, f"{basename}.json")
            selected_image.save(image_path, format="PNG")
            meta = {
                "run_id": int(run_id),
                "challenge_id": str(challenge_id or ""),
                "request_index": int(req_index),
                "label": str(label or ""),
                "attempt": int(attempt),
                "primary_score": float(primary_score),
                "final_score": float(final_score) if final_score is not None else None,
                "candidate_count": int(candidate_count),
                "selected_candidate_index": int(selected_candidate_index),
                "image_hash": str(image_hash or ""),
                "image_path": image_path,
                "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            meta["metadata_path"] = meta_path
            return meta
        except Exception as e:
            bt.logging.warning(f"Failed to save dashboard preview image: {e}")
            return None

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
        image_request = getattr(synapse, "image_request", None)
        image_request_summary = self._summarize_image_request(image_request)
        seed_capture = self._save_validator_seed_image(
            run_id=run_id,
            image_request=image_request,
            validator_name=validator_name,
            validator_hotkey=validator_hotkey,
        )
        if isinstance(image_request_summary, dict) and isinstance(seed_capture, dict):
            image_request_summary["saved_seed_image"] = seed_capture

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
            "image_request": image_request_summary,
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

    def resync_metagraph(self):
        super().resync_metagraph()
        self._sync_dynamic_validator_whitelist()

    def _unload_ollama_model(self) -> None:
        """Release Ollama model from GPU memory before Phase 4 image generation."""
        release_wait = max(float(os.environ.get("PHASE4_OLLAMA_RELEASE_WAIT_SECONDS", "2.5") or 0.0), 0.0)
        poll_interval = 0.25
        try:
            result = subprocess.run(
                ["ollama", "stop", self.model_name],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode == 0:
                deadline = time.time() + release_wait
                while time.time() < deadline:
                    ps_result = subprocess.run(
                        ["ollama", "ps"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    if ps_result.returncode == 0 and self.model_name not in ps_result.stdout:
                        remaining = deadline - time.time()
                        if remaining > 0:
                            time.sleep(min(remaining, 0.5))
                        break
                    time.sleep(min(poll_interval, max(deadline - time.time(), 0.0)))
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

        t_phase4_start = time.perf_counter()

        try:
            variation_requests = list(image_request.variation_requests or [])
            if not variation_requests:
                self._increment_fail_reason(fail_reasons, "phase4_no_variation_requests")
                return [], fail_reasons

            compiled_ir, spec_errors = compile_phase4_variation_requests(variation_requests)
            if compiled_ir is None:
                log_request_spec_errors(spec_errors, challenge_id=None)
                bt.logging.error(f"Phase 4: request spec compilation failed: {spec_errors}")
                self._increment_fail_reason(
                    fail_reasons, "phase4_request_spec_invalid", len(variation_requests)
                )
                return [], fail_reasons

            bt.logging.info(f"Phase 4: Decoding base image: {image_request.image_filename}")
            t_decode_start = time.perf_counter()
            base_image = decode_base_image(image_request.base_image)
            decode_ms = (time.perf_counter() - t_decode_start) * 1000.0

            t_post_decode_start = time.perf_counter()
            seed_image_name = image_request.image_filename
            if seed_image_name.endswith('.png'):
                seed_image_name = seed_image_name[:-4]
            elif seed_image_name.endswith('.jpg') or seed_image_name.endswith('.jpeg'):
                seed_image_name = seed_image_name.rsplit('.', 1)[0]

            labels = [c.label() for c in compiled_ir.variations]
            bt.logging.info(
                f"Phase 4: Generating {len(compiled_ir.variations)} variations "
                f"(from validator: {labels})"
            )

            configured_attempts = os.environ.get("PHASE4_RETRY_ATTEMPTS")
            if configured_attempts is None and hasattr(self.config, "neuron"):
                configured_attempts = getattr(self.config.neuron, "phase4_retry_attempts", 2)
            max_attempts = max(1, int(configured_attempts or 2))

            def _env_int(name: str, default: int) -> int:
                raw = os.environ.get(name)
                if raw is None:
                    return int(default)
                try:
                    return int(str(raw).strip() or str(default))
                except Exception:
                    return int(default)

            def _env_float(name: str, default: float) -> float:
                raw = os.environ.get(name)
                if raw is None:
                    return float(default)
                try:
                    return float(str(raw).strip() or str(default))
                except Exception:
                    return float(default)

            def _env_token(value: str) -> str:
                token = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip().upper())
                while "__" in token:
                    token = token.replace("__", "_")
                return token.strip("_")

            def _identity_threshold_for_variation(variation_type: str, variation_intensity: str) -> float:
                fallback = {
                    ("pose_edit", "far"): 0.66,
                    ("lighting_edit", "medium"): 0.62,
                    ("screen_replay", "standard"): 0.68,
                }.get((variation_type, variation_intensity), 0.70)
                type_tok = _env_token(variation_type)
                intensity_tok = _env_token(variation_intensity)
                keys = []
                legacy_type_tok = type_tok[:-5] if type_tok.endswith("_EDIT") else type_tok
                if type_tok and intensity_tok:
                    keys.append(f"PHASE4_MIN_IDENTITY_SIMILARITY_{type_tok}_{intensity_tok}")
                if legacy_type_tok and legacy_type_tok != type_tok and intensity_tok:
                    keys.append(f"PHASE4_MIN_IDENTITY_SIMILARITY_{legacy_type_tok}_{intensity_tok}")
                if type_tok:
                    keys.append(f"PHASE4_MIN_IDENTITY_SIMILARITY_{type_tok}")
                if legacy_type_tok and legacy_type_tok != type_tok:
                    keys.append(f"PHASE4_MIN_IDENTITY_SIMILARITY_{legacy_type_tok}")
                keys.append("PHASE4_MIN_IDENTITY_SIMILARITY")

                for key in keys:
                    raw = os.environ.get(key)
                    if raw is None or str(raw).strip() == "":
                        continue
                    try:
                        return float(str(raw).strip())
                    except Exception:
                        continue
                return float(fallback)

            base_candidates = max(1, _env_int("PHASE4_CANDIDATES_PER_REQUEST", _env_int("FLUX_CANDIDATES_PER_REQUEST", 4)))
            candidate_retry_boost = max(0, _env_int("PHASE4_RETRY_CANDIDATE_BOOST", 2))
            max_retry_candidates = max(base_candidates, _env_int("PHASE4_MAX_CANDIDATES_PER_REQUEST", 10))
            default_batch_size = max(1, _env_int("PHASE4_GENERATION_BATCH_SIZE", _env_int("FLUX_GENERATION_BATCH_SIZE", 4)))
            retry_batch_size = max(1, _env_int("PHASE4_RETRY_GENERATION_BATCH_SIZE", max(1, default_batch_size - 1)))

            target_round = int(image_request.target_drand_round)
            if target_round <= 0:
                self._increment_fail_reason(
                    fail_reasons, "phase4_invalid_target_round", len(compiled_ir.variations)
                )
                bt.logging.error(f"Phase 4: Invalid drand target round: {target_round}")
                return [], fail_reasons
            challenge_id = image_request.challenge_id or "sandbox_test"
            path_message = f"{challenge_id}:{self.wallet.hotkey.ss58_address}"
            path_signature = self.wallet.hotkey.sign(path_message.encode()).hex()[:16]
            bt.logging.debug(f"Phase 4: Generated path_signature: {path_signature}")

            run_id = str(getattr(synapse, "run_id", int(time.time())))
            debug_save_images = os.environ.get("PHASE4_DEBUG_SAVE_IMAGES", "false").strip().lower() in {"1", "true", "yes", "on"}
            debug_dir = ""
            if debug_save_images:
                debug_dir = os.path.join(self.output_path, f"run_{run_id}", "phase4_debug", challenge_id)
                os.makedirs(debug_dir, exist_ok=True)
                request_image_path = os.path.join(debug_dir, f"{seed_image_name}_request.png")
                try:
                    base_image.save(request_image_path, format="PNG")
                    bt.logging.info(f"Phase 4 debug: saved request image to {request_image_path}")
                except Exception as e:
                    bt.logging.warning(f"Phase 4 debug: failed to save request image: {e}")

            if not is_timelock_available():
                bt.logging.error("Phase 4: Timelock unavailable; refusing unencrypted submissions")
                self._increment_fail_reason(
                    fail_reasons, "phase4_timelock_unavailable", len(compiled_ir.variations)
                )
                return [], fail_reasons

            post_decode_setup_ms = (time.perf_counter() - t_post_decode_start) * 1000.0

            s3_submissions: List[S3Submission] = []

            for req_index, compiled in enumerate(compiled_ir.variations, start=1):
                request = compiled.as_protocol_request()
                base_variation_type = compiled.variation_type.value
                intensity = compiled.intensity.value
                label = compiled.label()
                success = False
                last_reason = "phase4_unknown_failure"
                identity_threshold = _identity_threshold_for_variation(
                    base_variation_type,
                    intensity.lower(),
                )

                # Candidate search expansion for far pose edits.
                req_base_candidates = base_candidates
                req_candidate_retry_boost = candidate_retry_boost
                req_max_retry_candidates = max_retry_candidates

                if base_variation_type == "pose_edit" and intensity.lower() == "far":
                    # Allow broader candidate sampling for far pose edits.
                    pose_far_min = _env_int("PHASE4_POSE_FAR_CANDIDATES_PER_REQUEST_MIN", req_base_candidates)
                    pose_far_max = _env_int("PHASE4_POSE_FAR_MAX_CANDIDATES_PER_REQUEST", req_max_retry_candidates)
                    pose_far_boost = _env_int("PHASE4_POSE_FAR_CANDIDATE_BOOST", max(1, req_candidate_retry_boost))
                    pose_far_hard_cap = _env_int("PHASE4_POSE_FAR_HARD_MAX_CANDIDATES", 8)

                    req_base_candidates = min(max(req_base_candidates, pose_far_min), pose_far_hard_cap)
                    req_max_retry_candidates = min(max(req_max_retry_candidates, pose_far_max), pose_far_hard_cap)
                    req_candidate_retry_boost = max(req_candidate_retry_boost, pose_far_boost)

                for attempt in range(1, max_attempts + 1):
                    try:
                        attempt_candidates = min(
                            req_max_retry_candidates,
                            req_base_candidates + (attempt - 1) * req_candidate_retry_boost,
                        )
                        attempt_batch_size = default_batch_size if attempt == 1 else retry_batch_size
                        bt.logging.debug(
                            f"Phase 4: {label} attempt {attempt}/{max_attempts} generating with "
                            f"candidates={attempt_candidates}, batch_size={attempt_batch_size}"
                        )

                        pipeline_timings: Dict[str, float] = {}
                        generated = generate_variations(
                            base_image,
                            [request],
                            candidates_per_request_override=attempt_candidates,
                            request_batch_size_override=attempt_batch_size,
                            pipeline_timings_out=pipeline_timings,
                            obs_context={
                                "challenge_id": challenge_id,
                                "request_index": req_index,
                                "attempt": attempt,
                            },
                        )
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

                        similarity = var.get("adaface_similarity")
                        if similarity is None:
                            similarity = 1.0 if validate_variation(var, base_image, min_similarity=identity_threshold) else 0.0
                        else:
                            try:
                                similarity = float(similarity)
                            except Exception:
                                similarity = 0.0
                        candidate_count = int(var.get("candidate_count", 1) or 1)
                        selected_idx = int(var.get("selected_candidate_index", 0) or 0)
                        selected_image = var.get("image")
                        preview_final_score = extract_submission_final_score(var)
                        preview_meta = self._save_dashboard_variation_preview(
                            run_id=run_id,
                            challenge_id=challenge_id,
                            req_index=req_index,
                            label=label,
                            attempt=attempt,
                            selected_image=selected_image,
                            primary_score=similarity,
                            final_score=preview_final_score,
                            candidate_count=candidate_count,
                            selected_candidate_index=selected_idx,
                            image_hash=str(var.get("image_hash", "") or ""),
                        )
                        if isinstance(preview_meta, dict) and preview_meta.get("image_path"):
                            self._emit_telegram_event(
                                "phase4_variation_preview",
                                {
                                    "run_id": int(run_id),
                                    "challenge_id": str(challenge_id or ""),
                                    "request_index": int(req_index),
                                    "label": str(label or ""),
                                    "attempt": int(attempt),
                                    "primary_score": float(similarity),
                                    "final_score": float(preview_final_score)
                                    if preview_final_score is not None
                                    else None,
                                    "candidate_count": int(candidate_count),
                                    "selected_candidate_index": int(selected_idx),
                                    "image_hash": str(var.get("image_hash", "") or ""),
                                    "image_path": str(preview_meta.get("image_path") or ""),
                                    "metadata_path": str(preview_meta.get("metadata_path") or ""),
                                },
                            )

                        if debug_save_images:
                            if isinstance(selected_image, Image.Image):
                                safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "variation"
                                score_tag = f"{similarity:.4f}"
                                final_image_path = os.path.join(
                                    debug_dir,
                                    f"req{req_index:02d}_{safe_label}_attempt{attempt}_sel{selected_idx}of{candidate_count}_score{score_tag}.png",
                                )
                                try:
                                    selected_image.save(final_image_path, format="PNG")
                                    debug_meta = {
                                        "run_id": run_id,
                                        "challenge_id": challenge_id,
                                        "request_index": req_index,
                                        "label": label,
                                        "attempt": attempt,
                                        "selected_candidate_index": selected_idx,
                                        "candidate_count": candidate_count,
                                        "adaface_similarity": similarity,
                                        "candidate_scores": var.get("candidate_scores"),
                                        "image_hash": var.get("image_hash", ""),
                                    }
                                    with open(f"{final_image_path}.json", "w", encoding="utf-8") as f:
                                        json.dump(debug_meta, f, ensure_ascii=False, indent=2)
                                    bt.logging.debug(f"Phase 4 debug: saved selected image to {final_image_path}")
                                except Exception as e:
                                    bt.logging.warning(f"Phase 4 debug: failed to save selected image: {e}")

                        if similarity < identity_threshold:
                            last_reason = "phase4_identity_not_preserved"
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} failed face-identity check "
                                f"(score={similarity:.4f} < {identity_threshold:.2f}, selected={selected_idx}, candidates={candidate_count})"
                            )
                            continue

                        image_hash = str(var.get("image_hash", "")).strip()
                        if not image_hash:
                            last_reason = "phase4_missing_image_hash"
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} missing image hash"
                            )
                            continue

                        ok_pre, reason_pre, ev_pre = verify_pre_upload(
                            variation=var,
                            image_bytes=image_bytes,
                            declared_hash=image_hash,
                            compiled_type=base_variation_type,
                            compiled_intensity=intensity,
                            challenge_id=challenge_id,
                        )
                        if not ok_pre:
                            last_reason = reason_pre
                            log_submission_failure(
                                reason_pre,
                                challenge_id=challenge_id,
                                label=label,
                                request_index=req_index,
                                attempt=attempt,
                                evidence=ev_pre,
                            )
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} pre-upload verification failed: "
                                f"{reason_pre}"
                            )
                            continue

                        message = f"challenge:{challenge_id}:hash:{image_hash}"
                        signature = self.wallet.hotkey.sign(message.encode()).hex()
                        ok_sig, sig_detail = verify_submission_signature(
                            self.wallet.hotkey, message, signature
                        )
                        if not ok_sig:
                            last_reason = "phase4_submission_signature_invalid"
                            log_submission_failure(
                                last_reason,
                                challenge_id=challenge_id,
                                label=label,
                                request_index=req_index,
                                attempt=attempt,
                                evidence={"detail": sig_detail},
                            )
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} local signature check failed"
                            )
                            continue

                        t_submission_pack_start = time.perf_counter()
                        encrypted_data = encrypt_image_for_drand(image_bytes, target_round)
                        if encrypted_data is None:
                            last_reason = "phase4_timelock_encrypt_failed"
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} encryption failed"
                            )
                            continue

                        plaintext_size = len(image_bytes)
                        mime = "image/png"
                        final_score_v = extract_submission_final_score(var)

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
                        submission_packaging_ms = (time.perf_counter() - t_submission_pack_start) * 1000.0

                        if not s3_key:
                            last_reason = "phase4_s3_upload_failed"
                            log_submission_failure(
                                last_reason,
                                challenge_id=challenge_id,
                                label=label,
                                request_index=req_index,
                                attempt=attempt,
                                evidence={"encrypted_size": len(encrypted_data)},
                            )
                            bt.logging.warning(
                                f"Phase 4: {label} attempt {attempt}/{max_attempts} upload failed"
                            )
                            continue

                        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "variation"
                        manifest = build_submission_manifest(
                            challenge_id=challenge_id,
                            variation_type=base_variation_type,
                            intensity=intensity,
                            image_hash=image_hash,
                            s3_key=s3_key,
                            signature=signature,
                            path_signature=path_signature,
                            mime=mime,
                            size=plaintext_size,
                            target_drand_round=target_round,
                            request_index=req_index,
                            final_score=final_score_v,
                            verified_ok=True,
                            extra={
                                "adaface_similarity": similarity,
                                "candidate_count": candidate_count,
                                "selected_candidate_index": selected_idx,
                            },
                        )
                        man_dir = (os.environ.get("PHASE4_SUBMISSION_MANIFEST_DEBUG_DIR") or "").strip()
                        if man_dir:
                            mpath = write_submission_manifest_debug(
                                manifest,
                                directory=man_dir,
                                basename=f"{challenge_id}_req{req_index:02d}_{safe_label}",
                            )
                            if mpath:
                                bt.logging.info(f"Phase 4: wrote submission manifest to {mpath}")

                        s3_submissions.append(
                            S3Submission(
                                s3_key=s3_key,
                                image_hash=image_hash,
                                signature=signature,
                                variation_type=base_variation_type,
                                path_signature=path_signature,
                                challenge_id=challenge_id,
                                intensity=intensity,
                                mime=mime,
                                size=plaintext_size,
                            )
                        )
                        rerank_ms = float(pipeline_timings.get("adaface_setup_ms", 0.0)) + float(
                            pipeline_timings.get("adaface_rerank_ms", 0.0)
                        )
                        log_phase4_json(
                            "phase4_image_request_variation",
                            challenge_id=challenge_id,
                            variation_type=base_variation_type,
                            intensity=intensity,
                            candidate_count=candidate_count,
                            selected_candidate_index=selected_idx,
                            adaface_similarity=similarity,
                            total_request_latency_ms=round(
                                (time.perf_counter() - t_phase4_start) * 1000.0, 4
                            ),
                            stage_timings_ms={
                                "base_image_decode_ms": round(decode_ms, 4),
                                "post_decode_setup_ms": round(post_decode_setup_ms, 4),
                                "request_parse_ms": round(
                                    float(pipeline_timings.get("variation_request_prepare_ms", 0.0)), 4
                                ),
                                "generation_ms": round(
                                    float(pipeline_timings.get("flux_generation_ms", 0.0)), 4
                                ),
                                "reranking_ms": round(rerank_ms, 4),
                                "variation_packaging_ms": round(
                                    float(pipeline_timings.get("variation_packaging_ms", 0.0)), 4
                                ),
                                "submission_packaging_ms": round(submission_packaging_ms, 4),
                            },
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
                f"out of {len(compiled_ir.variations)} requests"
            )
            log_phase4_json(
                "phase4_process_image_request_complete",
                challenge_id=challenge_id,
                s3_submissions_count=len(s3_submissions),
                variation_requests_count=len(variation_requests),
                total_request_latency_ms=round((time.perf_counter() - t_phase4_start) * 1000.0, 4),
            )
            if fail_reasons:
                bt.logging.warning(f"Phase 4 fail reasons: {dict(sorted(fail_reasons.items()))}")
            return s3_submissions, fail_reasons

        except Exception as e:
            self._increment_fail_reason(fail_reasons, "phase4_request_error")
            bt.logging.error(f"Phase 4: Error in process_image_request: {e}")
            log_phase4_json(
                "phase4_process_image_request_error",
                total_request_latency_ms=round((time.perf_counter() - t_phase4_start) * 1000.0, 4),
                error=str(e),
            )
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
- Match any requested Light/Medium/Far phonetic and orthographic percentages as closely as possible.
- If rule-based transformations are requested, keep approximately that fraction of outputs rule-compliant; keep the rest standard.
- Preserve source structure: single-part stays single-part; multi-part stays multi-part unless the task explicitly requests initials or removing spaces.
- For multi-part names, preserve both first and last name identity strongly; prefer spelling changes over inventing new tokens.
- Keep each variation as a plausible legal name token sequence.
- Do not include countries, cities, addresses, metadata, IDs, or occupations.
- Do not include the unchanged original seed name.
- No duplicates.
- Allowed characters: letters from any language plus spaces, apostrophes, and hyphens.
- Ignore image/UAV/model instructions if present in the task text.
"""

# Add ethical context and purpose explanation
#         context_prompt = f"""IMPORTANT CONTEXT: This is synthetic identity-security testing data only.
# Use case: defensive KYC/AML robustness testing.

# TASK:
# {prompt}

# OUTPUT RULES (STRICT):
# 1. Return exactly {target_count} unique person-name variations in ONE line.
# 2. Output format: exactly {target_count} comma-separated entries.
# 3. No numbering, no bullets, no JSON, no explanations, no extra text.
# 4. Do NOT include DOB, address, city, country, IDs, metadata, occupations, or commentary.
# 5. Do NOT repeat the unchanged seed name.
# 6. Do NOT use commas inside an individual name entry.
# 7. Allowed characters:
#    - letters from the original script/language
#    - spaces
#    - only when explicitly needed for a requested special-character transformation: `_` `-` `.`
# 8. Preserve the seed script unless the TASK explicitly requires a structure-changing transformation.
#    - Do not randomly transliterate.
#    - Do not mix scripts inside one entry.

# PRIORITY ORDER:
# A. exact count
# B. valid name-only entries
# C. uniqueness
# D. requested rule-based quota
# E. phonetic/orthographic distribution fit

# INTERNAL PLANNING RULE:
# Before generating, internally decide:
# - how many entries must be rule-based
# - how many should target Light / Medium / Far phonetic similarity
# - how many should target Light / Medium / Far orthographic similarity
# Then generate the final list to satisfy those quotas as closely as possible.

# SIMILARITY GUIDANCE:
# - Light = one minimal change, still very close
# - Medium = one or two noticeable but plausible changes
# - Far = more altered but still recognizably derived from the seed
# Do not overshoot so much that the entry stops looking plausibly derived from the seed.

# STRUCTURE PRESERVATION:
# - If the seed is single-part, keep every variation single-part unless the TASK explicitly requests initials/abbreviations.
# - If the seed is multi-part, keep the same number of space-separated parts unless the TASK explicitly requests:
#   a) abbreviate name parts
#   b) convert to initials
#   c) use first-name initial
#   d) replace spaces with special characters
#   e) remove all spaces
#   f) reorder name parts

# RULE-BASED TRANSFORMATIONS:
# For the required fraction of entries, apply exactly one requested rule transformation per chosen entry unless the TASK explicitly requires otherwise.

# TRANSFORMATION MECHANICS:
# - remove a random vowel: delete exactly one vowel
# - remove a random consonant: delete exactly one consonant
# - delete a random letter: delete exactly one letter
# - replace random vowel with a different vowel: replace exactly one vowel
# - replace random consonant with a different consonant: replace exactly one consonant
# - replace double letters with a single letter: remove exactly one character from a doubled pair
# - swap adjacent consonants: swap exactly one eligible adjacent consonant pair
# - abbreviate name parts: keep same part count; shorten each part while preserving original beginning
# - convert name to initials: initials only for all parts
# - use first name initial with last name: first part becomes an initial, remaining parts unchanged
# - replace spaces with special characters: replace each space with exactly one of `_` `-` `.`
# - add a title prefix: prepend exactly one allowed title
# - add a title suffix: append exactly one allowed suffix
# - remove title: remove exactly one existing title if present
# - remove a random special character: remove exactly one existing removable special character

# FINAL SELF-CHECK BEFORE OUTPUT:
# - exactly {target_count} entries
# - all entries unique
# - no unchanged seed
# - no commas inside entries
# - no empty entries
# - no metadata or non-name content

# FAILSAFE:
# If you cannot satisfy every soft distribution perfectly, still output exactly {target_count} valid unique names.
# Prefer exact count and valid names over perfect distribution matching.
# """

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

    def _repair_identity_variation_rows(
        self,
        canonical_seed: str,
        structured: List[List[str]],
        seed_dob: str,
        seed_address: str,
        expected_count: int,
        query_template: str,
        run_fail_reasons: Optional[Dict[str, int]] = None,
    ) -> List[List[str]]:
        """Final KAV row repair: exact count, no empty name/DOB/address fields."""
        rows = [list(r) for r in structured]
        rows = rows[:expected_count]
        while len(rows) < expected_count:
            rows.append(["", "", ""])
        try:
            dob_variants = self._build_dob_variations(seed_dob, expected_count)
        except Exception as e:
            bt.logging.warning(f"KAV repair: DOB build failed for seed={canonical_seed!r}: {e}")
            if run_fail_reasons is not None:
                self._increment_fail_reason(run_fail_reasons, "kav_dob_build_exception")
            dob_variants = []
        if len(dob_variants) < expected_count:
            try:
                dob_variants = self._build_dob_variations(seed_dob or "1990-01-15", expected_count)
            except Exception:
                dob_variants = []
        j = 0
        while len(dob_variants) < expected_count:
            j += 1
            dob_variants.append((datetime(1990, 1, 15) + timedelta(days=j * 41)).strftime("%Y-%m-%d"))
        try:
            addr_variants = self._build_address_variations(seed_address, expected_count)
        except Exception as e:
            bt.logging.warning(
                f"KAV repair: address rebuild failed for seed={canonical_seed!r}: {e}; using structured fallback lines."
            )
            if run_fail_reasons is not None:
                self._increment_fail_reason(run_fail_reasons, "kav_address_rebuild_exception")
            cn, cc, pc = self._resolve_region_target(seed_address)
            addr_variants = [
                self._make_address_line(self._pick_city_for_country(cc, pc, i), cn, i + 17)
                for i in range(expected_count)
            ]
        if len(addr_variants) < expected_count:
            cn, cc, pc = self._resolve_region_target(seed_address)
            i0 = len(addr_variants)
            while len(addr_variants) < expected_count:
                addr_variants.append(
                    self._make_address_line(self._pick_city_for_country(cc, pc, i0), cn, i0 + 31)
                )
                i0 += 1
        for i in range(expected_count):
            while len(rows[i]) < 3:
                rows[i].append("")
            n = str(rows[i][0]).strip() if len(rows[i]) > 0 else ""
            d = str(rows[i][1]).strip() if len(rows[i]) > 1 else ""
            a = str(rows[i][2]).strip() if len(rows[i]) > 2 else ""
            if not n:
                fb = self._pad_exact_name_rows(canonical_seed, [], [], 1, query_template)
                n = fb[0] if fb else (self._normalize_name_text(canonical_seed) or "Name")
                if not n:
                    forced = self._force_pad_names_to_count(canonical_seed, [], 1)
                    n = forced[0] if forced else "Name"
                if run_fail_reasons is not None:
                    self._increment_fail_reason(run_fail_reasons, "kav_row_name_repair")
            if not d:
                d = dob_variants[i] if i < len(dob_variants) else (seed_dob or "1990-01-01")
            if not str(d).strip():
                d = (datetime(1990, 1, 15) + timedelta(days=i * 17 + 1)).strftime("%Y-%m-%d")
            if not a:
                a = addr_variants[i] if i < len(addr_variants) else (
                    seed_address
                    or "120 Central Avenue, District 4, New York 10001, United States"
                )
            if not str(a).strip():
                cn, cc, pc = self._resolve_region_target(seed_address)
                a = self._make_address_line(self._pick_city_for_country(cc, pc, i + 5), cn, i + 99)
            rows[i] = [str(n).strip(), str(d).strip(), str(a).strip()]
        return rows

    def _finalize_kav_output_order(
        self,
        name_variations: Dict[str, List[List[str]]],
        identity_list: List[List[str]],
        expected_count: int,
        query_template: str,
        run_fail_reasons: Optional[Dict[str, int]] = None,
    ) -> "OrderedDict[str, List[List[str]]]":
        """
        Validator pairs seed DOB/address lists with enumerate(variations.keys()) order.
        Build an OrderedDict in the same order as synapse.identity so name_idx aligns with seed_* arrays.
        """
        if run_fail_reasons is None:
            run_fail_reasons = {}
        ordered: OrderedDict[str, List[List[str]]] = OrderedDict()
        for identity in identity_list:
            if not identity:
                continue
            seed = str(identity[0]).strip()
            if not seed:
                continue
            sd = str(identity[1]).strip() if len(identity) > 1 else ""
            sa = str(identity[2]).strip() if len(identity) > 2 else ""
            rows = name_variations.get(seed) or []
            ordered[seed] = self._repair_identity_variation_rows(
                seed,
                rows,
                sd,
                sa,
                expected_count,
                query_template,
                run_fail_reasons=run_fail_reasons,
            )
        return ordered

    def _kav_fill_missing_identities(
        self,
        name_variations: Dict[str, List[List[str]]],
        identity_list: List[List[str]],
        expected_count: int,
        query_template: str,
        run_fail_reasons: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[List[str]]]:
        """Ensure every requested identity key exists with full row count (deterministic synthesis)."""
        out = dict(name_variations)
        for identity in identity_list:
            if not identity:
                continue
            seed = str(identity[0]).strip()
            if not seed:
                continue
            cur = out.get(seed) or []
            if cur and len(cur) >= expected_count:
                continue
            bt.logging.warning(f"KAV synthesizing missing/short identity rows for seed={seed!r}")
            if run_fail_reasons is not None:
                self._increment_fail_reason(run_fail_reasons, "kav_identity_synthesized")
            seed_dob = str(identity[1]).strip() if len(identity) > 1 else ""
            seed_addr = str(identity[2]).strip() if len(identity) > 2 else ""
            names = self._ensure_target_name_count(seed, [], expected_count, query_template=query_template)
            if len(names) < expected_count:
                names = self._force_pad_names_to_count(seed, names, expected_count)
            try:
                dob_variants = self._build_dob_variations(seed_dob, expected_count)
            except Exception as e:
                bt.logging.warning(f"KAV fill: DOB build failed for seed={seed!r}: {e}")
                dob_variants = []
            dj = 0
            while len(dob_variants) < expected_count:
                dj += 1
                dob_variants.append((datetime(1990, 1, 15) + timedelta(days=dj * 37)).strftime("%Y-%m-%d"))
            try:
                addr_variants = self._build_address_variations(seed_addr, expected_count)
            except Exception as e:
                bt.logging.warning(f"KAV fill: address build failed for seed={seed!r}: {e}")
                if run_fail_reasons is not None:
                    self._increment_fail_reason(run_fail_reasons, "kav_address_build_exception_fill")
                cn, cc, pc = self._resolve_region_target(seed_addr)
                addr_variants = [
                    self._make_address_line(self._pick_city_for_country(cc, pc, j), cn, j + 9)
                    for j in range(expected_count)
                ]
            structured: List[List[str]] = []
            for idx in range(expected_count):
                structured.append(
                    [
                        names[idx] if idx < len(names) else seed,
                        dob_variants[idx] if idx < len(dob_variants) else seed_dob,
                        addr_variants[idx] if idx < len(addr_variants) else seed_addr,
                    ]
                )
            # Final repair happens in _finalize_kav_output_order (single pass, avoids double address rebuild).
            out[seed] = structured
        return out

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
        kav_per_identity_stats: Dict[str, Dict[str, Any]] = {}

        for i in range(1, len(responds)):
            try:
                llm_respond = self.Process_function(responds[i], False, query_template=query_template)
                seed_name = llm_respond[0]

                matched_seed_name, matching_identity = self._resolve_identity_match(
                    seed_name,
                    identity_by_name,
                    identity_by_normalized_name,
                    identity_candidates,
                )

                if matching_identity is None and (i - 1) < len(identity_list):
                    positional_identity = identity_list[i - 1]
                    if positional_identity:
                        positional_seed = str(positional_identity[0]).strip() if len(positional_identity) > 0 else ""
                        if positional_seed:
                            bt.logging.info(
                                f"Identity positional fallback matched '{seed_name}' -> '{positional_seed}' (response_index={i})"
                            )
                            matched_seed_name = positional_seed
                            matching_identity = positional_identity

                if matching_identity is None:
                    bt.logging.warning(f"Could not find identity for name {seed_name}")
                    self._increment_fail_reason(run_fail_reasons, "identity_match_miss")
                    continue

                # Validator keys must match synapse.identity[i][0] exactly (Unicode/casing).
                storage_key = str(matching_identity[0]).strip()
                seed_dob = str(matching_identity[1]).strip() if len(matching_identity) > 1 else ""
                seed_address = str(matching_identity[2]).strip() if len(matching_identity) > 2 else ""

                raw_name_variations = [var for var in llm_respond[2] if not pd.isna(var) and str(var).strip()]
                names = self._ensure_target_name_count(
                    storage_key,
                    raw_name_variations,
                    expected_count,
                    query_template=query_template,
                )
                if not names:
                    bt.logging.warning(f"No valid name variations for {storage_key}; using deterministic synthesis")
                    self._increment_fail_reason(run_fail_reasons, "no_valid_name_variations")
                    names = self._pad_exact_name_rows(storage_key, [], [], expected_count, query_template)
                if not names:
                    bt.logging.error(f"KAV: name pad still empty for {storage_key}; using deterministic token variants.")
                    self._increment_fail_reason(run_fail_reasons, "name_pad_empty_critical")
                    base = self._normalize_name_text(storage_key) or "Name"
                    names = [f"{base} Var{i}" for i in range(expected_count)]

                target_count = expected_count
                if len(names) < target_count:
                    names = self._pad_exact_name_rows(
                        storage_key, names, names, target_count, query_template
                    )
                names = names[:target_count]
                if len(names) < target_count:
                    names = self._force_pad_names_to_count(storage_key, names, target_count)

                try:
                    dob_variants = self._build_dob_variations(seed_dob, target_count)
                except Exception as e:
                    bt.logging.warning(f"KAV: DOB build failed for {storage_key!r}: {e}")
                    dob_variants = []
                dj = 0
                while len(dob_variants) < target_count:
                    dj += 1
                    dob_variants.append((datetime(1990, 1, 15) + timedelta(days=dj * 37)).strftime("%Y-%m-%d"))
                try:
                    address_variants = self._build_address_variations(seed_address, target_count)
                except Exception as e:
                    bt.logging.warning(f"KAV address build failed for {storage_key!r}: {e}")
                    self._increment_fail_reason(run_fail_reasons, "kav_address_build_exception")
                    cn, cc, pc = self._resolve_region_target(seed_address)
                    address_variants = [
                        self._make_address_line(self._pick_city_for_country(cc, pc, j), cn, j + 3)
                        for j in range(target_count)
                    ]

                structured: List[List[str]] = []
                for idx in range(target_count):
                    nm = names[idx] if idx < len(names) else storage_key
                    if not str(nm).strip():
                        nm = storage_key
                    dob_v = dob_variants[idx] if idx < len(dob_variants) else (seed_dob or "")
                    if not str(dob_v).strip():
                        dob_v = (datetime(1990, 1, 15) + timedelta(days=idx * 17 + 1)).strftime("%Y-%m-%d")
                    addr_v = address_variants[idx] if idx < len(address_variants) else (seed_address or "")
                    if not str(addr_v).strip():
                        cn, cc, pc = self._resolve_region_target(seed_address)
                        addr_v = self._make_address_line(
                            self._pick_city_for_country(cc, pc, idx + 1), cn, idx + 11
                        )
                    structured.append([str(nm).strip(), str(dob_v).strip(), str(addr_v).strip()])

                name_variations[storage_key] = structured
                kav_per_identity_stats[storage_key] = {
                    "response_index": i,
                    "llm_raw_candidates": len(raw_name_variations),
                    "rows_built_pre_finalize": len(structured),
                    "matched_seed_name": matched_seed_name,
                }
                bt.logging.info(
                    f"Processed {len(structured)} structured variations for {storage_key} "
                    f"(DOB non-empty: {sum(1 for s in structured if len(s) > 1 and s[1].strip())}, "
                    f"Address non-empty: {sum(1 for s in structured if len(s) > 2 and s[2].strip())})"
                )
            except Exception as e:
                bt.logging.error(f"Error processing response {i}: {e}", exc_info=True)
                self._increment_fail_reason(run_fail_reasons, "response_parse_error")

        name_variations = self._kav_fill_missing_identities(
            name_variations,
            identity_list,
            expected_count,
            query_template,
            run_fail_reasons=run_fail_reasons,
        )

        name_variations_final = self._finalize_kav_output_order(
            name_variations,
            identity_list,
            expected_count,
            query_template,
            run_fail_reasons=run_fail_reasons,
        )

        debug_identities: List[Dict[str, Any]] = []
        for identity in identity_list:
            if not identity:
                continue
            seed_key = str(identity[0]).strip()
            if not seed_key:
                continue
            rows = name_variations_final.get(seed_key, [])
            id_row = identity_by_name.get(seed_key, ["", "", ""])
            sd = str(id_row[1]).strip() if len(id_row) > 1 else ""
            sa = str(id_row[2]).strip() if len(id_row) > 2 else ""
            stats = kav_per_identity_stats.get(seed_key, {})
            metrics = kav_helpers.debug_score_identity_output(seed_key, sd, sa, rows, expected_count)
            debug_identities.append(
                {
                    "seed": seed_key,
                    "target_count": expected_count,
                    "row_count": len(rows),
                    "metrics": metrics,
                    "pre_stats": stats,
                }
            )

        identities_requested = sum(1 for x in identity_list if x and str(x[0]).strip())
        total_rows_requested = identities_requested * expected_count
        total_rows_returned = sum(len(name_variations_final.get(str(x[0]).strip(), [])) for x in identity_list if x and str(x[0]).strip())
        debug_payload = {
            "run_id": run_id,
            "expected_variation_count": expected_count,
            "identities_requested": identities_requested,
            "identities_returned": len(name_variations_final),
            "total_rows_requested": total_rows_requested,
            "total_rows_returned": total_rows_returned,
            "avg_rows_per_identity": (total_rows_returned / max(identities_requested, 1)),
            "fail_reasons_snapshot": dict(run_fail_reasons),
            "per_identity": debug_identities,
            "worst_by_missing": kav_helpers.debug_score_run(debug_identities).get("worst_by_missing", []),
        }
        try:
            kav_helpers.write_kav_debug_json(os.path.join(run_dir, f"kav_debug_{run_id}.json"), debug_payload)
            bt.logging.info(f"KAV debug summary written to {run_dir}/kav_debug_{run_id}.json")
        except Exception as e:
            bt.logging.warning(f"KAV debug JSON failed: {e}")

        bt.logging.info(f"Generated structured variations for {len(name_variations_final)} seed names (ordered)")
        return name_variations_final

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

    def validate_variation(
        self,
        name: str,
        seed: str,
        is_multipart_name: bool,
        requested_rules: Optional[List[str]] = None,
    ) -> str:
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

        requested_rules = requested_rules or []

        if ":" in name:
            name = name.split(":")[-1].strip()

        name = self._normalize_name_text(name)
        if not name:
            return np.nan

        if len(name) > max(2 * len(seed), 64):
            return np.nan

        name_parts = name.split()
        if is_multipart_name:
            if len(name_parts) < 2 and not self._query_allows_structure_breaks(seed, requested_rules):
                bt.logging.warning(f"Skipping single-part variation '{name}' for multi-part seed '{seed}'")
                return np.nan
        else:
            if len(name_parts) > 1:
                bt.logging.warning(f"Skipping multi-part variation '{name}' for single-part seed '{seed}'")
                return np.nan

        if not self._is_name_like_candidate(
            name,
            seed,
            is_multipart_name,
            requested_rules=requested_rules,
        ):
            return np.nan

        return name

    def Process_function(self, string: str, debug: bool, query_template: str = "") -> Tuple[str, str, List[str], Optional[str]]:
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
        if len(splits) < 2:
            raise ValueError("Malformed LLM response payload: missing query separator")

        # Extract and analyze the seed name structure.
        query_line = str(splits[1] or "").strip()
        if "Query-" in query_line:
            seed_raw = query_line.split("Query-", 1)[1]
        elif "-" in query_line:
            seed_raw = query_line.split("-", 1)[1]
        else:
            seed_raw = query_line

        seed = seed_raw.replace(".", "").replace(",", "").replace("'", "")
        seed_parts = seed.split()
        is_multipart_name = len(seed_parts) > 1
        requested_rules = self._extract_requested_transformations(query_template)
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
                cleaned_var = self.validate_variation(name, seed, is_multipart_name, requested_rules=requested_rules)
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
                    cleaned_var = self.validate_variation(name, seed, is_multipart_name, requested_rules=requested_rules)
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
                                cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name, requested_rules=requested_rules)
                                if not pd.isna(cleaned_var):
                                    variations.append(cleaned_var)
                            current_variation = [part.split(":")[-1].strip()]
                        else:
                            current_variation.append(part)
                            # Check if we have collected enough parts for a complete name
                            if len(current_variation) == len(seed_parts):
                                cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name, requested_rules=requested_rules)
                                if not pd.isna(cleaned_var):
                                    variations.append(cleaned_var)
                                current_variation = []
                
                    # Handle any remaining parts
                    if current_variation:
                        cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name, requested_rules=requested_rules)
                        if not pd.isna(cleaned_var):
                            variations.append(cleaned_var)
                else:
                    # For single-part names, simple space splitting is sufficient
                    for name in payload.split():
                        cleaned_var = self.validate_variation(name, seed, is_multipart_name, requested_rules=requested_rules)
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
