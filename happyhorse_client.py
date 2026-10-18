#!/usr/bin/env python3
"""
happyhorse_client.py — Pure HappyHorse-1.0 AI Video Generator Client
(Zero 2D drawing / zero fake procedural shapes).

Connects ONLY to real HappyHorse-1.0 video generation endpoints or imports
real HappyHorse-1.0 `.mp4` clips:
1. Official Alibaba DashScope / QwenCloud (`DASHSCOPE_API_KEY`):
   `POST /api/v1/services/aigc/video-generation/video-synthesis` (`happyhorse-1.0` / `happyhorse-1.1-t2v`)
2. Official Fal.ai (`FAL_KEY`):
   `POST https://queue.fal.run/alibaba/happy-horse/text-to-video`
3. APIMart / Runware / Self-Hosted HappyHorse-1.0 Server (`HAPPYHORSE_API_URL` + `HAPPYHORSE_API_KEY`):
   `POST /v1/videos/generations`
4. Local Folder of Real HappyHorse `.mp4` Clips (`happyhorse_clips/`):
   Ingests real HappyHorse-1.0 `.mp4` clips for automatic 30-minute episode
   and 3-hour movie stitching.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import urllib.request
from typing import Any, Dict, List, Optional

HAPPYHORSE_CAMERA_DIRECTIVES = {
    "dolly_zoom_in": "slow dolly in, shallow depth of field, 35mm anamorphic lens",
    "pan_left_to_right": "smooth cinematic pan left to right, physically coherent motion",
    "pan_right_to_left": "smooth cinematic pan right to left, physically coherent motion",
    "crane_up": "aerial crane shot rising upward, volumetric lighting",
    "slow_pull_back": "slow pull back wide shot, dramatic lighting",
    "tracking_orbit": "cinematic handheld tracking shot following subject movement",
}


@dataclass
class HappyHorseTaskRequest:
    """Official HappyHorse-1.0 T2V / I2V / R2V API request payload."""
    model: str = "happyhorse-1.0"
    prompt: str = ""
    resolution: str = "1080P"        # "720P" or "1080P"
    ratio: str = "16:9"              # "16:9", "9:16", "1:1", "4:3", "3:4"
    duration: int = 10               # integer 3..15 seconds
    seed: int = 42
    first_frame_image: Optional[str] = None
    image_urls: Optional[List[str]] = None

    def to_dashscope_payload(self) -> Dict[str, Any]:
        inp: Dict[str, Any] = {"prompt": self.prompt[:2500]}
        if self.first_frame_image:
            inp["first_frame_image"] = self.first_frame_image
        elif self.image_urls:
            inp["image_urls"] = self.image_urls
        return {
            "model": self.model,
            "input": inp,
            "parameters": {
                "resolution": self.resolution,
                "ratio": self.ratio,
                "duration": max(3, min(15, int(self.duration))),
                "seed": self.seed,
            },
        }

    def to_unified_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": self.prompt[:2500],
            "resolution": self.resolution,
            "size": self.ratio,
            "duration": max(3, min(15, int(self.duration))),
            "seed": self.seed,
        }
        if self.first_frame_image:
            payload["first_frame_image"] = self.first_frame_image
        elif self.image_urls:
            payload["image_urls"] = self.image_urls
        return payload

    def to_fal_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": self.prompt[:2500],
            "aspect_ratio": self.ratio,
            "resolution": self.resolution.lower(),
            "duration": max(3, min(15, int(self.duration))),
            "seed": self.seed,
        }
        if self.first_frame_image:
            payload["image_url"] = self.first_frame_image
        return payload


class HappyHorseVideoGenerator:
    """
    Pure HappyHorse-1.0 Video Generator client.
    No 2D drawing or fake procedural canvas rendering.
    """

    def __init__(
        self,
        model: str = "happyhorse-1.0",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: str = "auto",
    ):
        self.model = model
        self.provider = provider
        self.api_url = (api_url or os.environ.get("HAPPYHORSE_API_URL") or "").rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("HAPPYHORSE_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("FAL_KEY")
            or ""
        )

    def is_configured(self) -> bool:
        return bool(self.api_key or self.api_url)

    @staticmethod
    def build_happyhorse_prompt(
        visual_prompt: str,
        camera_motion: str,
        speaker: str,
        dialogue: str,
        style: str,
    ) -> str:
        cam_desc = HAPPYHORSE_CAMERA_DIRECTIVES.get(
            camera_motion, "smooth cinematic camera movement, shallow depth of field"
        )
        return (
            f"{visual_prompt}. Camera: {cam_desc}. "
            f"Dialogue ({speaker}): \"{dialogue}\" with synchronized native audio and ambient foley."
        )

    def generate_shot_mp4(
        self,
        req: HappyHorseTaskRequest,
        out_mp4: str,
        timeout: float = 300.0,
    ) -> str:
        """
        Submit a real HappyHorse-1.0 video generation job and download the MP4.
        Raises RuntimeError with clear instructions if no HappyHorse API endpoint
        or key is configured.
        """
        if not self.is_configured():
            raise RuntimeError(
                "HappyHorse-1.0 requires a live HappyHorse endpoint or API key "
                "(set HAPPYHORSE_API_KEY, DASHSCOPE_API_KEY, FAL_KEY, or HAPPYHORSE_API_URL, "
                "or place downloaded HappyHorse .mp4 clips in 'happyhorse_clips/'). "
                "2D fallback rendering has been completely removed."
            )

        os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)

        if self.provider == "fal" or (os.environ.get("FAL_KEY") and not self.api_url):
            endpoint = self.api_url or "https://fal.run/alibaba/happy-horse/text-to-video"
            body = req.to_fal_payload()
            headers = {
                "Authorization": f"Key {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "HappyHorse-MovieStudio/1.0",
            }
        elif "dashscope" in self.api_url or os.environ.get("DASHSCOPE_API_KEY"):
            endpoint = (
                self.api_url
                or "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
            )
            body = req.to_dashscope_payload()
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "X-DashScope-Async": "enable",
                "Content-Type": "application/json",
                "User-Agent": "HappyHorse-MovieStudio/1.0",
            }
        else:
            endpoint = self.api_url or "https://api.apimart.ai/v1/videos/generations"
            body = req.to_unified_payload()
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "HappyHorse-MovieStudio/1.0",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

        http_req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(http_req, timeout=30) as r:
            res = json.loads(r.read().decode("utf-8"))

        video_url = (
            res.get("video_url")
            or (res.get("video") or {}).get("url")
            or (res.get("output") or {}).get("video_url")
        )
        task_id = res.get("task_id") or (res.get("output") or {}).get("task_id")

        if not video_url and task_id:
            base_host = endpoint.split("/api/v1")[0] if "/api/v1" in endpoint else self.api_url
            poll_url = f"{base_host}/api/v1/tasks/{task_id}"
            deadline = time.time() + timeout
            while time.time() < deadline:
                time.sleep(3.0)
                p_req = urllib.request.Request(poll_url, headers=headers)
                with urllib.request.urlopen(p_req, timeout=20) as pr:
                    p_res = json.loads(pr.read().decode("utf-8"))
                out_block = p_res.get("output") or p_res
                status = out_block.get("task_status") or out_block.get("status")
                if status in ("SUCCEEDED", "completed", "succeeded"):
                    video_url = out_block.get("video_url")
                    break
                if status in ("FAILED", "CANCELED", "failed"):
                    raise RuntimeError(f"HappyHorse-1.0 task failed: {p_res}")

        if not video_url:
            raise RuntimeError(f"HappyHorse-1.0 response did not contain video_url: {res}")

        with urllib.request.urlopen(video_url, timeout=60) as vr:
            data = vr.read()
        with open(out_mp4, "wb") as f:
            f.write(data)
        return out_mp4
