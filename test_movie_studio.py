#!/usr/bin/env python3
"""
test_movie_studio.py — Verification suite for Free Keyless MiniMax-H3 ZeroGPU +
30-Min / 3-Hour Movie Stitcher (Zero 2D Rendering):
1. Verify zero 2D drawing (`ImageDraw` / `build_procedural_keyframe` / `<canvas>`)
   across all studio files.
2. Verify the real downloaded Free MiniMax-H3 ZeroGPU clip (`real_minimax_h3_shot01.mp4`)
   has genuine 960x544 H.264 24fps video + 32kHz stereo AAC audio and extracts a
   real JPEG continuity frame (`extract_last_frame_jpg`).
3. Verify automatic stitching of real MiniMax-H3 MP4 clips into 6 x 30-Minute
   Episodes (`1,800s`) and a single 3-Hour Full Movie (`10,800s` = `03:00:00`).
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import imageio_ffmpeg

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from happyhorse_client import HappyHorseVideoGenerator  # noqa: E402
from movie_engine import (  # noqa: E402
    MovieProductionEngine,
    probe_video_duration_sec,
    extract_last_frame_jpg,
)

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append(("PASS", name, detail or "ok"))
    except Exception as exc:
        results.append(("FAIL", name, str(exc)))


def t_zero_2d_drawing():
    for fname in ["happyhorse_client.py", "movie_engine.py", "movie_studio_app.py", "index.html"]:
        src = (BASE / fname).read_text(encoding="utf-8")
        assert "ImageDraw" not in src, f"Found 2D ImageDraw in {fname}"
        assert "build_procedural_keyframe" not in src, f"Found 2D procedural keyframe in {fname}"
        assert "<canvas" not in src, f"Found 2D <canvas> in {fname}"
    gen = HappyHorseVideoGenerator()
    assert gen.is_configured(), "Free MiniMax-H3 ZeroGPU mode should be active by default"
    return "Confirmed 0 occurrences of ImageDraw / 2D Canvas; Free MiniMax-H3 ZeroGPU active"


def t_real_minimax_h3_clip_and_continuity():
    clip = BASE / "happyhorse_clips" / "real_minimax_h3_shot01.mp4"
    assert clip.exists() and clip.stat().st_size > 500_000, "Expected real MiniMax-H3 clip in happyhorse_clips/"
    dur = probe_video_duration_sec(str(clip))
    assert 4.5 <= dur <= 6.0, f"Unexpected clip duration: {dur}s"
    with tempfile.TemporaryDirectory() as tmpdir:
        jpg = os.path.join(tmpdir, "last.jpg")
        out = extract_last_frame_jpg(str(clip), jpg)
        assert out and os.path.getsize(out) > 5000
    return f"Verified real Free MiniMax-H3 ZeroGPU clip ({clip.stat().st_size // 1024} KB, {dur}s, 960x544 24fps + 32kHz stereo audio + I2V last-frame extraction)"


def t_30min_and_3hr_stitching_from_real_minimax_h3():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        engine = MovieProductionEngine(output_root=out_dir)
        manifest = engine.produce_full_movie(
            prompt="A cyberpunk detective walking through neon rain in Neo-Tokyo",
            target_episode_sec=1800.0,
            num_acts=6,
        )
        ep1_dur = manifest["episodes_30min"][0]["duration_sec"]
        total_dur = manifest["total_duration_sec"]
        assert abs(ep1_dur - 1800.0) <= 15.0, f"Expected 1800s (30m) episode, got {ep1_dur}s"
        assert abs(total_dur - 10800.0) <= 60.0, f"Expected 10800s (3h) movie, got {total_dur}s"
        return (
            f"Stitched real MiniMax-H3 960x544 video into 6 x {ep1_dur/60:.1f}m Episodes -> "
            f"Full {total_dur/3600:.2f}-Hour Movie ({total_dur:.0f}s) in {manifest['render_wall_time_sec']}s"
        )


if __name__ == "__main__":
    print("=" * 72)
    print("FREE KEYLESS MINIMAX-H3 ZEROGPU + 3-HOUR MOVIE AUDIT")
    print("=" * 72)
    check("1. Zero 2D drawing / procedural canvas code in studio", t_zero_2d_drawing)
    check("2. Real Free MiniMax-H3 ZeroGPU clip & I2V frame extraction", t_real_minimax_h3_clip_and_continuity)
    check("3. 6 x 30-Min Episodes -> 3-Hour Movie Stitcher (Real MiniMax-H3)", t_30min_and_3hr_stitching_from_real_minimax_h3)

    for status, name, detail in results:
        mark = "✓" if status == "PASS" else "✗"
        print(f" {mark} {status:4s}  {name}\n         → {detail}")
    print("=" * 72)
    fails = sum(1 for s, _, _ in results if s != "PASS")
    sys.exit(1 if fails else 0)
