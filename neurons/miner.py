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
import random
import re
from difflib import SequenceMatcher
import bittensor as bt
import ollama
import pandas as pd
import os
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
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
from MIID.miner.image_generator import decode_base_image, generate_variations, validate_variation
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
        patterns = [
            r"\bexact(?:ly)?\s+(\d{1,2})\s+variations?\b",
            r"\bgenerate\s+(\d{1,2})\s+variations?\b",
            r"\btotal\s+(\d{1,2})\s+variations?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, query_template, flags=re.IGNORECASE)
            if match:
                try:
                    value = int(match.group(1))
                    return max(4, min(20, value))
                except Exception:
                    continue
        return default

    @staticmethod
    def _compact_text(value: str) -> str:
        compact = re.sub(r"[^a-z]", "", value.lower())
        compact = compact.replace("republic", "").replace("kingdom", "").replace("state", "")
        compact = compact.replace("federation", "").replace("democratic", "").replace("islamic", "")
        return compact

    def _clean_name_candidate(self, raw_name: str, seed_name: str, is_multipart: bool) -> Optional[str]:
        """Normalize and validate name candidates before scoring."""
        if not raw_name:
            return None
        candidate = re.sub(r"[^A-Za-zÀ-ÿ\s]", " ", str(raw_name))
        candidate = " ".join(candidate.split()).strip()
        if not candidate:
            return None
        if candidate.lower() == seed_name.lower():
            return None
        parts = candidate.split()
        if is_multipart and len(parts) < 2:
            return None
        if not is_multipart and len(parts) != 1:
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
        part_clean = re.sub(r"[^A-Za-zÀ-ÿ]", "", part)
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
            perturb = part[:-1] + random.choice("aeiouwy")
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
        if preferred_city:
            return preferred_city
        if country_code and country_code in self._city_names_by_country:
            candidates = self._city_names_by_country[country_code]
            if candidates:
                return candidates[idx % len(candidates)]
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
        # Generate a unique run ID using timestamp
        run_id = int(time.time())
        bt.logging.info(f"Starting run {run_id} for {len(synapse.identity)} names")
        
        # Get timeout from synapse (default to 120s if not specified)
        timeout = getattr(synapse, 'timeout', 120.0)
        bt.logging.info(f"Request timeout: {timeout:.1f}s for {len(synapse.identity)} names")
        start_time = time.time()
        
        # Create a run-specific directory
        run_dir = os.path.join(self.output_path, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        
        # This will store all responses from the LLM in a format that can be processed later
        # Format: ["Respond", "---", "Query-{name}", "---", "{LLM response}"]
        Response_list = []
        
        # Track which names we've processed
        processed_names = []
        
        # Process each identity in the request, respecting the timeout
        for i, identity in enumerate(tqdm(synapse.identity, desc="Processing identities")):
            # Check if we're approaching the timeout (reserve 15% for processing)
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            time_buffer = timeout * 0.15  # Reserve 15% of total time for final processing
            
            # If time is running out, skip remaining identities
            if remaining < time_buffer:
                bt.logging.warning(
                    f"Time limit approaching ({elapsed:.1f}/{timeout:.1f}s), "
                    f"processed {len(processed_names)}/{len(synapse.identity)} identities. "
                    f"Skipping remaining identities to ensure timely response."
                )
                break
            
            # Extract name, dob, and address from identity array
            name = identity[0] if len(identity) > 0 else "Unknown"
            dob = identity[1] if len(identity) > 1 else "Unknown"
            address = identity[2] if len(identity) > 2 else "Unknown"
            
            # Format the response list for later processing
            Response_list.append("Respond")
            Response_list.append("---")
            Response_list.append("Query-" + name)
            Response_list.append("---")
            
            # Format the query with the current name, address, and DOB
            formatted_query = synapse.query_template.replace("{name}", name)
            formatted_query = formatted_query.replace("{address}", address)
            formatted_query = formatted_query.replace("{dob}", dob)
            
            # Query the LLM with timeout awareness
            try:
                bt.logging.info(f"Generating variations for name: {name}, remaining time: {remaining:.1f}s")
                # Pass a more limited timeout to the LLM call to ensure we stay within bounds
                name_respond = self.Get_Respond_LLM(formatted_query)
                Response_list.append(name_respond)
                processed_names.append(name)
            except Exception as e:
                bt.logging.error(f"Error querying LLM for name {name}: {str(e)}")
                Response_list.append("Error: " + str(e))
        
        # Check if we've managed to process at least some names
        if not processed_names:
            bt.logging.error("Could not process any names within the timeout period")
            synapse.variations = {}
            return synapse
        
        # Process the responses to extract variations, but be aware of remaining time
        remaining = timeout - (time.time() - start_time)
        bt.logging.info(f"Processing responses with {remaining:.1f}s remaining of {timeout:.1f}s timeout")
        
        # Only proceed with processing if we have enough time
        if remaining > 1.0:  # Ensure at least 1 second for processing
            variations = self.process_variations(
                Response_list,
                run_id,
                run_dir,
                synapse.identity,
                synapse.query_template,
            )
            bt.logging.info(f"======== FINAL VARIATIONS===============================================: {variations}")
            # Set the variations in the synapse for return to the validator
            synapse.variations = variations
        else:
            bt.logging.warning(f"Insufficient time for processing responses, returning empty result")
            synapse.variations = {}
        
        # Log final timing information
        total_time = time.time() - start_time
        bt.logging.info(
            f"Request completed in {total_time:.2f}s of {timeout:.1f}s allowed. "
            f"Processed {len(processed_names)}/{len(synapse.identity)} names."
        )
        
        bt.logging.info(f"======== SYNAPSE VARIATIONS===============================================: {synapse.variations}")
        bt.logging.info(f"==========================Processed variations for {len(synapse.variations)} names in run {run_id}")
        bt.logging.info(f"==========================Synapse: {synapse}")
        bt.logging.info("========================================================================================")

        # ==========================================================================
        # Phase 4: Process Image Request
        # ==========================================================================
        if hasattr(synapse, 'image_request') and synapse.image_request is not None:
            try:
                s3_submissions = self.process_image_request(synapse)
                synapse.s3_submissions = s3_submissions
                bt.logging.info(f"Phase 4: Generated {len(s3_submissions)} S3 submissions")
            except Exception as e:
                bt.logging.error(f"Phase 4: Failed to process image request: {e}")
                synapse.s3_submissions = []
        # ==========================================================================

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

    def process_image_request(self, synapse: IdentitySynapse) -> List[S3Submission]:
        """Process Phase 4 image variation request.

        Generates image variations, encrypts them with drand timelock,
        uploads to S3, and returns S3 submission references.

        Args:
            synapse: IdentitySynapse with image_request

        Returns:
            List of S3Submission objects
        """
        image_request = synapse.image_request
        if not image_request:
            return []

        try:
            # 1. Decode base image
            bt.logging.info(f"Phase 4: Decoding base image: {image_request.image_filename}")
            base_image = decode_base_image(image_request.base_image)

            # Extract seed image name from filename (remove .png extension if present)
            seed_image_name = image_request.image_filename
            if seed_image_name.endswith('.png'):
                seed_image_name = seed_image_name[:-4]  # Remove .png extension
            elif seed_image_name.endswith('.jpg') or seed_image_name.endswith('.jpeg'):
                seed_image_name = seed_image_name.rsplit('.', 1)[0]  # Remove extension

            # 2. Generate variations via FLUX: pass full variation_requests (type + intensity per variation)
            #    so the model gets the correct prompt for each (pose_edit/light, expression_edit/medium, etc.)
            bt.logging.info(
                f"Phase 4: Generating {image_request.requested_variations} variations "
                f"(from validator: {[f'{v.type}({v.intensity})' for v in image_request.variation_requests]})"
            )
            variations = generate_variations(
                base_image,
                image_request.variation_requests
            )

            # 3. Process each variation
            s3_submissions = []
            target_round = image_request.target_drand_round
            challenge_id = image_request.challenge_id or "sandbox_test"

            # Generate path_signature ONCE per challenge for security
            # This prevents other miners from writing to our path
            path_message = f"{challenge_id}:{self.wallet.hotkey.ss58_address}"
            path_signature = self.wallet.hotkey.sign(path_message.encode()).hex()[:16]
            bt.logging.debug(f"Phase 4: Generated path_signature: {path_signature}")

            for var in variations:
                try:
                    # Validate image before processing
                    if not self.is_valid_image_bytes(var["image_bytes"]):
                        bt.logging.warning(
                            f"Phase 4: Skipping invalid/corrupt image for {var['variation_type']}"
                        )
                        continue

                    # Validate face identity preserved (AdaFace, threshold 0.7)
                    if not validate_variation(var, base_image, min_similarity=0.7):
                        bt.logging.warning(
                            f"Phase 4: Skipping {var['variation_type']} - face identity not preserved"
                        )
                        continue
                    
                    # Sign the image hash
                    message = f"challenge:{challenge_id}:hash:{var['image_hash']}"
                    signature = self.wallet.hotkey.sign(message.encode()).hex()

                    # Encrypt with drand timelock
                    if is_timelock_available():
                        encrypted_data = encrypt_image_for_drand(
                            var["image_bytes"],
                            target_round
                        )
                        if encrypted_data is None:
                            bt.logging.warning(f"Phase 4: Encryption failed for {var['variation_type']}")
                            continue
                    else:
                        # SANDBOX: Use raw bytes if timelock not available
                        bt.logging.warning("Phase 4: Timelock not available, using raw bytes (SANDBOX ONLY)")
                        encrypted_data = var["image_bytes"]

                    # Upload to S3 (SANDBOX: mock upload)
                    s3_key = upload_to_s3(
                        encrypted_data=encrypted_data,
                        miner_hotkey=self.wallet.hotkey.ss58_address,
                        signature=signature,
                        image_hash=var["image_hash"],
                        target_round=target_round,
                        challenge_id=challenge_id,
                        variation_type=var["variation_type"],
                        path_signature=path_signature,
                        seed_image_name=seed_image_name
                    )

                    if s3_key:
                        s3_submissions.append(S3Submission(
                            s3_key=s3_key,
                            image_hash=var["image_hash"],
                            signature=signature,
                            variation_type=var["variation_type"],
                            path_signature=path_signature
                        ))
                        bt.logging.debug(f"Phase 4: Created submission for {var['variation_type']}")

                except Exception as e:
                    bt.logging.error(f"Phase 4: Error processing variation {var['variation_type']}: {e}")
                    continue

            bt.logging.info(f"Phase 4: Successfully created {len(s3_submissions)} S3 submissions")
            return s3_submissions

        except Exception as e:
            bt.logging.error(f"Phase 4: Error in process_image_request: {e}")
            return []

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
- Return only name variations.
- Return exactly {target_count} name variations.
- Output plain comma-separated values.
- No numbering, no bullets, no JSON, no explanation text.
- Keep each variation as a realistic person name.
- Preserve the original name-part structure (single-part stays single-part, multi-part stays multi-part).
- Provide diverse, non-duplicate variants.
- Ignore image, UAV, and metadata sections if present in the task text.
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
                    "temperature": 0.7,
                    "top_p": 0.9
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
            
        Returns:
            Dictionary mapping each name to its list of [name, dob, address] variations
        """
        bt.logging.info(f"Processing {len(Response_list)} responses")
        responds = "".join(Response_list).split("Respond")
        expected_count = self._extract_expected_variation_count(query_template, default=10)
        bt.logging.info(f"Parsed expected variation count from query: {expected_count}")

        identity_by_name = {}
        identity_by_normalized_name = {}
        for identity in identity_list:
            if not identity:
                continue
            seed_name = str(identity[0]).strip()
            if seed_name:
                identity_by_name[seed_name] = identity
                norm_seed = re.sub(r"\s+", " ", re.sub(r"[^A-Za-zÀ-ÿ\s]", " ", seed_name)).strip().lower()
                if norm_seed:
                    identity_by_normalized_name[norm_seed] = identity

        name_variations: Dict[str, List[List[str]]] = {}

        for i in range(1, len(responds)):
            try:
                llm_respond = self.Process_function(responds[i], False)
                seed_name = llm_respond[0]

                matching_identity = identity_by_name.get(seed_name)
                if matching_identity is None:
                    norm_name = re.sub(r"\s+", " ", re.sub(r"[^A-Za-zÀ-ÿ\s]", " ", seed_name)).strip().lower()
                    matching_identity = identity_by_normalized_name.get(norm_name)
                if matching_identity is None:
                    bt.logging.warning(f"Could not find identity for name {seed_name}")
                    continue

                seed_dob = str(matching_identity[1]).strip() if len(matching_identity) > 1 else ""
                seed_address = str(matching_identity[2]).strip() if len(matching_identity) > 2 else ""

                raw_name_variations = [var for var in llm_respond[2] if not pd.isna(var) and str(var).strip()]
                names = self._ensure_target_name_count(seed_name, raw_name_variations, expected_count)
                if not names:
                    bt.logging.warning(f"No valid name variations for {seed_name}; skipping")
                    name_variations[seed_name] = []
                    continue

                target_count = min(expected_count, len(names))
                names = names[:target_count]
                dob_variants = self._build_dob_variations(seed_dob, target_count)
                address_variants = self._build_address_variations(seed_address, target_count)

                structured = []
                for idx in range(target_count):
                    structured.append([
                        names[idx],
                        dob_variants[idx] if idx < len(dob_variants) else (seed_dob or ""),
                        address_variants[idx] if idx < len(address_variants) else (seed_address or ""),
                    ])

                name_variations[seed_name] = structured
                bt.logging.info(
                    f"Processed {len(structured)} structured variations for {seed_name} "
                    f"(DOB non-empty: {sum(1 for s in structured if len(s) > 1 and s[1].strip())}, "
                    f"Address non-empty: {sum(1 for s in structured if len(s) > 2 and s[2].strip())})"
                )
            except Exception as e:
                bt.logging.error(f"Error processing response {i}: {e}")

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
        name = name.strip()
        if not name or name.isspace():
            return np.nan
        
        # Handle cases with colons (e.g., "Here are variations: Name")
        if ":" in name:
            name = name.split(":")[-1].strip()
        
        # Check length reasonability (variation shouldn't be more than 2x the seed length)
        if len(name) > 2 * len(seed):
            return np.nan
        
        # Check structure consistency with seed name
        name_parts = name.split()
        if is_multipart_name:
            # For multi-part seed names (e.g., "John Smith"), variations must also have multiple parts
            if len(name_parts) < 2:
                bt.logging.warning(f"Skipping single-part variation '{name}' for multi-part seed '{seed}'")
                return np.nan
        else:
            # For single-part seed names (e.g., "John"), variations must be single part
            if len(name_parts) > 1:
                bt.logging.warning(f"Skipping multi-part variation '{name}' for single-part seed '{seed}'")
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
