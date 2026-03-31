# Phase 4 image pipeline (current miner)

Current Phase 4 generation is now designed around an external ComfyUI server, with the miner acting as a client/router and keeping the existing request compiler, identity scoring, reranking, encryption, and S3 submission path.

## End-to-end flow

1. **Entry** — `neurons/miner.py` calls `process_image_request(synapse)` when `synapse.image_request` is present.
2. **Request parsing** — `MIID/miner/request_spec.py` compiles each validator `VariationRequest` into a normalized request object before generation.
3. **Base image decode** — `MIID/miner/image_generator.py` converts the Base64 seed image into a PIL RGB image.
4. **Per-variation loop** — The miner still processes one variation at a time, preserving retry logic, identity thresholds, candidate-count tuning, and submission packaging.
5. **Request routing** — `MIID/miner/request_routing.py` resolves each incoming validator variation by `type` into a backend route and workflow name. This is per variation, not “all five tasks at once”, so mixed validator requests can be routed correctly.
6. **ComfyUI backend dispatch** — `MIID/miner/generator_backends/comfyui.py` uploads the seed image to ComfyUI, resolves the task family workflow in `MIID/miner/comfyui_workflows.py`, submits the graph over the ComfyUI HTTP API, waits for completion, and downloads the generated images.
7. **Task routing** — The backend builds task-specific ComfyUI graphs for:
   - `pose_edit`: SDXL img2img plus FaceID conditioning, with ControlNet when available
   - `expression_edit`: SDXL img2img plus IP-Adapter FaceID
   - `background_edit`: SDXL img2img/outpaint plus IP-Adapter FaceID
   - `lighting_edit`: ComfyUI lighting-edit path with SDXL fallback graph
   - `screen_replay`: ComfyUI screen-replay generation path with replay-specific prompt metadata
8. **Post-generation scoring** — `image_generator.generate_variations()` still runs face preprocessing, AdaFace/ArcFace identity scoring, adherence scoring, ensemble reranking, and candidate selection exactly as before.
9. **Packaging / submission** — The winning image is encoded, hashed, timelock-encrypted, uploaded to S3, and returned as `S3Submission`.

## Runtime requirements

- `SN54_IMAGE_GENERATION_BACKEND=comfyui`
- `COMFYUI_BASE_URL` must point at a running ComfyUI server
- The startup script now performs a ComfyUI `/system_stats` preflight before launching the miner when the Comfy backend is selected
- `MIID/miner/downloading_model.py` now verifies ComfyUI reachability when the backend is `comfyui`

## ComfyUI notes

- The current implementation queries the live ComfyUI `/object_info` API and adapts to the installed nodes.
- If preferred nodes for a task family are missing, the backend logs capability notes and uses the best available ComfyUI graph for that task.
- The current server inventory observed during migration included `sd_xl_base_1.0.safetensors`, `ip-adapter-faceid-plusv2_sdxl.bin`, and one ControlNet file, but not dedicated InstantID or IC-Light nodes. Those can be added later without changing the miner integration surface.

## Observability

Structured logs still use `MIID.miner.pipeline_observability.log_phase4_json`.

- `phase4_generate_variations` now includes the selected backend name.
- Per-variation metadata carries `backend`, `route_backend`, `route_workflow`, `workflow_name`, optional `comfyui_prompt_id`, and any capability notes alongside the existing scoring data.

## Key modules

- `MIID/miner/comfyui_client.py`
- `MIID/miner/comfyui_workflows.py`
- `MIID/miner/generator_backends/comfyui.py`
- `MIID/miner/image_generator.py`
- `scripts/miner/load_env.sh`
- `scripts/miner/start_sn54_miner.sh`
