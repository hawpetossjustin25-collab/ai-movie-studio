#!/usr/bin/env python3
"""
happyhorse_client.py — 100% Free Keyless AI Video Generator Client
(MiniMax-H3 ZeroGPU Cluster + Optional HappyHorse-1.0 API, Zero 2D Fallback).

How it generates real AI video with native stereo audio for $0 (no API key,
no signup, no local GPU required):
1. Primary Free Keyless Engine — Live Hugging Face ZeroGPU MiniMax-H3 Cluster:
   - `https://wank3r-minimax-h3.hf.space` (`MiniMax-H3` 8-step Turbo, 960x544
     24fps H.264 + 32kHz stereo AAC native audio, 5-15s per clip)
   - `https://pepe104-minimax-h3-turbo-lora-uncensored.hf.space` (6-step Turbo
     automatic failover)
   - Supports uploading the previous shot's final frame to `/gradio_api/upload`
     and passing it as `image_path` (`first_frame_image`) so every shot chains
     seamlessly into the next!
2. Optional Paid/Custom API Override:
   - Also supports HappyHorse-1.0 (`DASHSCOPE_API_KEY`, `FAL_KEY`, or custom
     `HAPPYHORSE_API_URL`) if the user ever supplies a key.
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

FREE_MINIMAX_H3_HOSTS = [
    ("https://wank3r-minimax-h3.hf.space", "wank3r"),
    ("https://pepe104-minimax-h3-turbo-lora-uncensored.hf.space", "pepe104"),
]


@dataclass
class HappyHorseTaskRequest:
    """Unified request payload for Free MiniMax-H3 ZeroGPU & HappyHorse-1.0."""
    model: str = "minimax-h3-free-zerogpu"
    prompt: str = ""
    resolution: str = "960x544 · 16:9 fast"
    ratio: str = "16:9"
    duration: int = 5                # 4..15 seconds
    steps: int = 8
    seed: int = 42
    first_frame_image: Optional[str] = None   # local file path or data URI
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
                "resolution": "1080P" if "1080" in self.resolution else "720P",
                "ratio": self.ratio,
                "duration": max(3, min(15, int(self.duration))),
                "seed": self.seed,
            },
        }


class HappyHorseVideoGenerator:
    """
    Real AI Video Generator Client (Zero 2D drawing):
    - Defaults to the 100% Free Keyless MiniMax-H3 ZeroGPU Cluster (generates
      real 960x544 24fps H.264 + 32kHz stereo native audio MP4 clips for free).
    - Also supports HappyHorse-1.0 API if an API key is provided.
    """

    def __init__(
        self,
        model: str = "minimax-h3-free-zerogpu",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: str = "free_minimax_h3",
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
        # Free MiniMax-H3 ZeroGPU mode works out of the box with no API key!
        return True

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
            f"{speaker} speaking: \"{dialogue}\", cinematic lighting, realistic motion, native stereo audio."
        )

    @staticmethod
    def _upload_first_frame_to_gradio(host: str, image_file_path: str) -> Optional[Dict[str, Any]]:
        """Upload a local JPEG frame to Gradio 5 `/gradio_api/upload` for I2V continuity."""
        if not image_file_path or not os.path.exists(image_file_path):
            return None
        boundary = "----AIMovieStudioBoundary" + str(int(time.time() * 1000))
        filename = os.path.basename(image_file_path)
        file_bytes = open(image_file_path, "rb").read()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            f"{host}/gradio_api/upload",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "Mozilla/5.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                paths = json.loads(r.read().decode("utf-8"))
            if isinstance(paths, list) and paths:
                return {
                    "path": paths[0],
                    "url": f"{host}/gradio_api/file={paths[0]}",
                    "size": len(file_bytes),
                    "orig_name": filename,
                    "mime_type": "image/jpeg",
                    "is_stream": False,
                    "meta": {"_type": "gradio.FileData"},
                }
        except Exception:
            return None
        return None

    def _generate_via_free_minimax_h3(
        self,
        req: HappyHorseTaskRequest,
        out_mp4: str,
        first_frame_path: Optional[str] = None,
        timeout: float = 240.0,
    ) -> str:
        """
        Call the live Free Hugging Face ZeroGPU MiniMax-H3 cluster (`wank3r` -> `pepe104`
        failover) and download the generated 960x544 24fps H.264 + 32kHz stereo MP4.
        """
        last_err = None
        dur = max(4, min(15, int(req.duration)))

        for host, variant in FREE_MINIMAX_H3_HOSTS:
            try:
                img_filedata = self._upload_first_frame_to_gradio(host, first_frame_path) if first_frame_path else None
                if variant == "wank3r":
                    data_list = [
                        req.prompt[:2000],
                        img_filedata,
                        None,
                        "960x544 \u00b7 16:9 fast",
                        dur,
                        max(7, int(req.steps)),
                        int(req.seed),
                        False,
                        True,   # turbo=True
                        [],
                        None,
                        1.0,
                        None,
                    ]
                else:
                    data_list = [
                        req.prompt[:2000],
                        img_filedata,
                        None,
                        "960x544 \u00b7 16:9 fast",
                        dur,
                        6,
                        int(req.seed),
                        False,
                        True,
                        "",
                    ]

                call_url = f"{host}/gradio_api/call/generate"
                post_req = urllib.request.Request(
                    call_url,
                    data=json.dumps({"data": data_list}).encode("utf-8"),
                    headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                    method="POST",
                )
                with urllib.request.urlopen(post_req, timeout=20) as r:
                    ev_id = json.loads(r.read().decode("utf-8"))["event_id"]

                stream_url = f"{call_url}/{ev_id}"
                video_download_url = None
                cur_event = ""
                with urllib.request.urlopen(stream_url, timeout=timeout) as sr:
                    for raw_line in sr:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if line.startswith("event:"):
                            cur_event = line.split(":", 1)[1].strip()
                        elif line.startswith("data:"):
                            payload_str = line.split(":", 1)[1].strip()
                            if cur_event == "error":
                                raise RuntimeError(f"{host} error: {payload_str}")
                            if cur_event == "complete":
                                arr = json.loads(payload_str)
                                if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                                    video_download_url = arr[0].get("url") or (
                                        f"{host}/gradio_api/file={arr[0]['path']}" if arr[0].get("path") else None
                                    )
                                break

                if video_download_url:
                    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
                    urllib.request.urlretrieve(video_download_url, out_mp4)
                    if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 10000:
                        return out_mp4
            except Exception as exc:
                last_err = exc
                continue

        raise RuntimeError(f"Free MiniMax-H3 ZeroGPU cluster failed: {last_err}")

    def generate_shot_mp4(
        self,
        req: HappyHorseTaskRequest,
        out_mp4: str,
        first_frame_path: Optional[str] = None,
        timeout: float = 240.0,
    ) -> str:
        """
        Generate a real AI video clip:
        - Uses Free Keyless MiniMax-H3 ZeroGPU by default (no API key needed!).
        - Or uses custom/paid HappyHorse-1.0 endpoint if `self.api_key` or `self.api_url` is set.
        """
        os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
        if not self.api_key and not self.api_url:
            return self._generate_via_free_minimax_h3(req, out_mp4, first_frame_path=first_frame_path, timeout=timeout)

        endpoint = (
            self.api_url
            or "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        )
        body = req.to_dashscope_payload()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-Async": "enable",
            "Content-Type": "application/json",
            "User-Agent": "AIMovieStudio/1.0",
        }
        http_req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(http_req, timeout=30) as r:
            res = json.loads(r.read().decode("utf-8"))
        video_url = res.get("video_url") or (res.get("output") or {}).get("video_url")
        if not video_url:
            raise RuntimeError(f"API response did not contain video_url: {res}")
        urllib.request.urlretrieve(video_url, out_mp4)
        return out_mp4
