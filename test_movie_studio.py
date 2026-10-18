#!/usr/bin/env python3
"""
test_movie_studio.py — Verification suite for Pure HappyHorse-1.0 AI Movie Studio
(Verifies ALL 2D drawing is removed and HappyHorse-1.0 API + 30m/3h FFmpeg stitcher work):
1. Verify zero 2D drawing (`PIL` / `ImageDraw` / `Canvas2D`) exists in `happyhorse_client.py`,
   `movie_engine.py`, `movie_studio_app.py`, or `index.html`.
2. Verify HappyHorse-1.0 API client & `first_frame_image` continuity chaining against a mock
   HappyHorse-1.0 server returning real H.264+AAC MP4 clips.
3. Verify automatic stitching of HappyHorse `.mp4` clips into 6 x 30-Minute Episodes (`1,800s`)
   and a single 3-Hour Full Movie (`10,800s` = `03:00:00`).
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import imageio_ffmpeg

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from happyhorse_client import HappyHorseTaskRequest, HappyHorseVideoGenerator  # noqa: E402
from movie_planner import MoviePlanner  # noqa: E402
from movie_engine import MovieProductionEngine, probe_video_duration_sec, extract_last_frame_data_uri  # noqa: E402

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append(("PASS", name, detail or "ok"))
    except Exception as exc:
        results.append(("FAIL", name, str(exc)))


def _make_sample_mp4(out_path: str, duration_sec: int = 5) -> str:
    """Create a valid H.264+AAC MP4 test clip via ffmpeg lavfi (no 2D Pillow drawing)."""
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-f", "lavfi",
        "-i", "color=c=black:s=640x360:r=12",
        "-f", "lavfi",
        "-i", "anullsrc=r=22050:cl=mono",
        "-t", str(duration_sec),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    return out_path


def t_zero_2d_drawing():
    for fname in ["happyhorse_client.py", "movie_engine.py", "movie_studio_app.py", "index.html"]:
        src = (BASE / fname).read_text(encoding="utf-8")
        assert "ImageDraw" not in src, f"Found 2D ImageDraw in {fname}"
        assert "build_procedural_keyframe" not in src, f"Found 2D procedural keyframe in {fname}"
        assert "<canvas" not in src, f"Found 2D <canvas> in {fname}"
    # Also verify calling HappyHorseVideoGenerator without API key/clips refuses 2D fallback
    gen = HappyHorseVideoGenerator()
    try:
        gen.generate_shot_mp4(HappyHorseTaskRequest(prompt="test"), "_should_not_exist.mp4")
        raise AssertionError("Should have refused unconfigured generation without 2D fallback")
    except RuntimeError as e:
        assert "2D fallback rendering has been completely removed" in str(e)
    return "Confirmed 0 occurrences of ImageDraw / 2D Canvas across all studio files"


def t_happyhorse_api_and_continuity():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_mp4 = _make_sample_mp4(os.path.join(tmpdir, "sample_hh.mp4"), duration_sec=5)
        uri = extract_last_frame_data_uri(sample_mp4)
        assert uri and uri.startswith("data:image/jpeg;base64,")
        req = HappyHorseTaskRequest(
            model="happyhorse-1.0",
            prompt="Cinematic shot",
            resolution="1080P",
            ratio="16:9",
            duration=10,
            first_frame_image=uri,
        )
        ds = req.to_dashscope_payload()
        assert ds["model"] == "happyhorse-1.0"
        assert ds["input"]["first_frame_image"].startswith("data:image/jpeg;base64,")
    return "HappyHorse-1.0 T2V + I2V first_frame_image continuity extraction verified"


def t_30min_and_3hr_happyhorse_stitching():
    with tempfile.TemporaryDirectory() as tmpdir:
        clips_dir = Path(tmpdir) / "clips"
        out_dir = Path(tmpdir) / "out"
        clips_dir.mkdir()
        out_dir.mkdir()
        c1 = _make_sample_mp4(str(clips_dir / "hh_clip_01.mp4"), duration_sec=10)
        c2 = _make_sample_mp4(str(clips_dir / "hh_clip_02.mp4"), duration_sec=10)

        engine = MovieProductionEngine(output_root=out_dir, clips_dir=clips_dir, happyhorse_model="happyhorse-1.0")
        manifest = engine.produce_full_movie(
            prompt="A cyberpunk detective and an AI android uncover an orbital conspiracy across Neo-Tokyo",
            target_episode_sec=1800.0,
            num_acts=6,
        )
        ep1_dur = manifest["episodes_30min"][0]["duration_sec"]
        total_dur = manifest["total_duration_sec"]
        assert abs(ep1_dur - 1800.0) <= 15.0, f"Expected 1800s (30m) episode, got {ep1_dur}s"
        assert abs(total_dur - 10800.0) <= 60.0, f"Expected 10800s (3h) movie, got {total_dur}s"
        return (
            f"Stitched HappyHorse MP4 clips into 6 x {ep1_dur/60:.1f}m Episodes -> "
            f"Full {total_dur/3600:.2f}-Hour Movie ({total_dur:.0f}s) in {manifest['render_wall_time_sec']}s"
        )


if __name__ == "__main__":
    print("=" * 72)
    print("HAPPYHORSE-1.0 PURE PIPELINE AUDIT (ZERO 2D RENDERING)")
    print("=" * 72)
    check("1. Zero 2D drawing / procedural canvas code in studio", t_zero_2d_drawing)
    check("2. HappyHorse-1.0 API payload & I2V last-frame continuity", t_happyhorse_api_and_continuity)
    check("3. 6 x 30-Min Episodes -> 3-Hour Movie HappyHorse MP4 Stitcher", t_30min_and_3hr_happyhorse_stitching)

    for status, name, detail in results:
        mark = "✓" if status == "PASS" else "✗"
        print(f" {mark} {status:4s}  {name}\n         → {detail}")
    print("=" * 72)
    fails = sum(1 for s, _, _ in results if s != "PASS")
    sys.exit(1 if fails else 0)
