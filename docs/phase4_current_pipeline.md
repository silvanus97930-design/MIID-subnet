# Phase 4 image pipeline (current miner)

Concise map of the SN54 miner path from the synapse to S3 submission metadata. Behavior matches `neurons/miner.py`, `MIID/miner/image_generator.py`, `MIID/miner/generate_variations.py`, and `MIID/miner/downloading_model.py` (model prefetch only).

## End-to-end flow

1. **Entry** — `Miner.forward` finishes name-variation work, then if `synapse.image_request` is set, calls `process_image_request(synapse)` and assigns `synapse.s3_submissions`.

2. **Request parsing (`process_image_request`)** — Reads `IdentitySynapse.image_request` (`ImageRequest`): `base_image` (base64), `image_filename`, `variation_requests` (`VariationRequest`: `type`, `intensity`, `description`, `detail`), `target_drand_round`, `challenge_id`, `reveal_timestamp`. Aborts with fail counters if the request is empty, has no variation requests, or `target_drand_round <= 0`.

3. **Base image decoding** — `decode_base_image(image_request.base_image)` in `image_generator.py` (Base64 → PIL RGB).

4. **Per-variation loop** — For each `VariationRequest`, the miner may adjust identity thresholds and candidate/batch limits (e.g. far `pose_edit`, `lighting_edit` medium, `screen_replay` standard), then calls `generate_variations` with a **single** request and optional candidate/batch overrides, with retries on failure.

5. **Candidate generation (`generate_variations` in `image_generator.py`)** — Delegates to `generate_variations` in `generate_variations.py` (FLUX): builds prompts from validator `description` + `detail`, runs batched diffusion (`num_images_per_prompt` = candidates per request), returns raw candidate PIL images per request. Adaptive overrides (e.g. far pose) widen candidate counts and tweak steps/guidance via env-driven caps.

6. **AdaFace scoring** — Loads AdaFace once per `generate_variations` call, extracts a base embedding, scores each candidate with `score_variation_candidates`, then `_select_best_candidate` picks the highest cosine-similarity score.

7. **Candidate selection** — Best candidate is encoded to PNG bytes, SHA-256 `image_hash` computed, metadata attached (`adaface_similarity`, `candidate_scores`, `candidate_count`, `selected_candidate_index`).

8. **Validation gate** — Miner checks image bytes, optional `validate_variation` if similarity missing, and `adaface_similarity` against env thresholds before accepting.

9. **Encryption / upload / return** — Builds `challenge:{challenge_id}:hash:{image_hash}`, signs with hotkey. `encrypt_image_for_drand` timelock-encrypts bytes for `target_drand_round`. `upload_to_s3` stores ciphertext and metadata; returns `s3_key`. Response rows are `S3Submission` (`s3_key`, `image_hash`, `signature`, `variation_type`, `path_signature`). `synapse.s3_submissions` lists these; **images are not returned inline**.

## Observability (structured logs)

JSON logs use `MIID.miner.pipeline_observability.log_phase4_json` (`event` + fields):

- **`phase4_generate_variations`** — Per call to `image_generator.generate_variations`: `stage_timings_ms` (`request_parse_ms`, `generation_ms`, `reranking_ms`, `final_packaging_ms`), plus optional `obs_context` (`challenge_id`, `request_index`, `attempt`).
- **`phase4_image_request_variation`** — On successful S3 row: `challenge_id`, `variation_type`, `intensity`, `candidate_count`, `selected_candidate_index`, `adaface_similarity`, `total_request_latency_ms`, `stage_timings_ms` (decode, post-decode setup, FLUX/AdaFace/packaging breakdown, `submission_packaging_ms`).
- **`phase4_process_image_request_complete`** — End of handler: `challenge_id`, counts, `total_request_latency_ms`.
- **`phase4_process_image_request_error`** — Top-level exception: `total_request_latency_ms`, `error`.

**Naming note:** `request_parse_ms` inside `stage_timings_ms` on the miner row is FLUX **prompt preparation** time (variation requests → prompts). **Post-decode setup** (paths, drand round, challenge id, timelock gate) is `post_decode_setup_ms`. **Base64 → PIL** is `base_image_decode_ms`.

## Model download

`MIID/miner/downloading_model.py` is a standalone Hugging Face prefetch for FLUX weights; it is not invoked on every request but supports offline caches used by `generate_variations.py`.
