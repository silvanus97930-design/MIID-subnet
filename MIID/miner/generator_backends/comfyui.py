from __future__ import annotations

import os
import threading
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence

import bittensor as bt
from PIL import Image

from MIID.miner.comfyui_client import ComfyUIClient
from MIID.miner.comfyui_workflows import build_workflow_plan
from MIID.miner.generator_backends.base import GenerationConfig, ImageGeneratorBackend
from MIID.miner.request_routing import resolve_request_route


class ComfyUIGeneratorBackend(ImageGeneratorBackend):
    """External ComfyUI-backed image generator for all Phase 4 task families."""

    def __init__(self) -> None:
        self._client = ComfyUIClient()
        self._lock = threading.Lock()
        self._serialize = (os.environ.get("COMFYUI_SERIALIZE_REQUESTS", "true").strip().lower() in {"1", "true", "yes", "on"})

    def _generation_lock_context(self):
        return self._lock if self._serialize else nullcontext()

    def load(self) -> None:
        self._client.prewarm()

    def prewarm(self) -> None:
        self.load()

    def generate_candidates(
        self,
        base_image: Any,
        compiled_request: Sequence[Any],
        generation_config: GenerationConfig,
        timings_out: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        if not compiled_request:
            return []
        if not isinstance(base_image, Image.Image):
            raise TypeError("ComfyUIGeneratorBackend expects a PIL.Image base_image")

        with self._generation_lock_context():
            t_prep0 = time.perf_counter()
            object_info = self._client.object_info()
            available_nodes = {str(key) for key in object_info.keys()}
            upload_name = self._client.upload_image(base_image.convert("RGB"), filename_prefix="miid_seed")
            routes = [resolve_request_route(req) for req in compiled_request]
            plans = [
                build_workflow_plan(
                    req,
                    uploaded_image=upload_name,
                    source_size=base_image.size,
                    generation_config=generation_config,
                    available_nodes=available_nodes,
                    request_index=index,
                    workflow_name_override=route.workflow_name,
                )
                for index, (req, route) in enumerate(zip(compiled_request, routes))
            ]
            prepare_ms = (time.perf_counter() - t_prep0) * 1000.0

            results: List[Dict[str, Any]] = []
            generation_ms_total = 0.0
            for plan, route in zip(plans, routes):
                missing = [node for node in plan.required_nodes if node not in available_nodes]
                if missing:
                    raise RuntimeError(
                        f"ComfyUI workflow '{plan.workflow_name}' cannot run; missing nodes: {', '.join(missing)}"
                    )
                if plan.capability_notes:
                    bt.logging.warning(
                        f"ComfyUI workflow {plan.workflow_name} capability notes: "
                        f"{'; '.join(plan.capability_notes)}"
                    )
                t_gen0 = time.perf_counter()
                prompt_id = self._client.queue_prompt(plan.prompt)
                history_entry = self._client.wait_for_completion(prompt_id)
                images = self._client.collect_output_images(history_entry, node_ids=plan.save_node_ids)
                generation_ms_total += (time.perf_counter() - t_gen0) * 1000.0
                if not images:
                    raise RuntimeError(f"ComfyUI workflow '{plan.workflow_name}' completed without output images")
                row = dict(plan.metadata)
                row["candidates"] = images
                row["backend"] = "comfyui"
                row["comfyui_prompt_id"] = prompt_id
                row["route_backend"] = route.backend_name
                row["route_workflow"] = route.workflow_name
                results.append(row)

            if timings_out is not None:
                timings_out["variation_request_prepare_ms"] = prepare_ms
                timings_out["flux_generation_ms"] = generation_ms_total
            return results
