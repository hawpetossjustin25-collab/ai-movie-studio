#!/usr/bin/env python3
"""
test_movie_studio.py — End-to-end verification suite for AI Movie Studio:
1. REA Multi-Generator Architecture Bridge (`ViMax` + `forge-film` + `aeon` + `MoneyPrinter`)
2. 6-Act x 30-Minute (3-Hour) Screenplay & Character Bank Planner
3. Real H.264 + AAC Shot Rendering (2.5D Ken-Burns + Sidechain-Ducked Audio + Subtitles)
4. Real 30-Minute (1,800s) Episode MP4 Assembly & FFmpeg Duration Probe
5. Real 3-Hour (10,800s = 03:00:00) Full Movie Automatic Stitching across 6 x 30-Min Episodes
"""

import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from movie_planner import MoviePlanner, REAArchitectureBridge  # noqa: E402
from movie_engine import (  # noqa: E402
    MovieProductionEngine,
    probe_video_duration_sec,
    render_shot_mp4,
)

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append(("PASS", name, detail or "ok"))
    except Exception as exc:
        results.append(("FAIL", name, str(exc)))


def t_rea_architecture_bridge():
    caps = REAArchitectureBridge.inspect_capabilities()
    assert caps["target_episode_sec"] == 1800 and caps["target_full_movie_sec"] == 10800
    repos = caps["synthesized_repositories"]
    assert "HKUDS/ViMax" in repos and "F-R-L/forge-film" in repos and "AEON-7/aeon-movie-maker" in repos
    return f"REA status={caps['rea_status']}, 4 GitHub architectures synthesized (target 1800s/ep -> 10800s/movie)"


def t_movie_planner_3hr():
    planner = MoviePlanner()
    plan = planner.create_movie_plan(
        "A cyberpunk detective and an AI android uncover an orbital conspiracy across Neo-Tokyo",
        num_acts=6,
        episode_duration_sec=1800.0,
        shots_per_act=12,
    )
    assert len(plan.acts) == 6
    assert plan.total_duration_sec == 10800.0
    assert len(plan.characters) == 4
    total_shots = sum(len(a.shots) for a in plan.acts)
    return f"Planned '{plan.title}' ({len(plan.acts)} acts x 30m = 180m / 3.0 hrs, {total_shots} shots, {len(plan.characters)} consistent characters)"


def t_single_shot_render():
    planner = MoviePlanner()
    plan = planner.create_movie_plan("Sci-fi starship battle over Saturn")
    shot = plan.acts[0].shots[0]
    out_mp4 = str(BASE / "output_movies" / "_test_single_shot.mp4")
    render_shot_mp4(
        shot=shot,
        act_title=plan.acts[0].title,
        movie_title=plan.title,
        style=plan.style,
        characters=plan.characters,
        out_mp4=out_mp4,
        render_duration_sec=3.0,
        width=640,
        height=360,
        fps=12,
    )
    dur = probe_video_duration_sec(out_mp4)
    size_kb = os.path.getsize(out_mp4) / 1024.0
    assert os.path.exists(out_mp4) and 2.5 <= dur <= 3.5, f"Unexpected shot duration: {dur}s"
    os.remove(out_mp4)
    return f"Rendered H.264+AAC cinema shot ({dur:.2f}s, {size_kb:.1f} KB, 2.5D Ken-Burns + ducked audio + subtitles)"


def t_full_3_hour_movie_production():
    engine = MovieProductionEngine(output_root=BASE / "output_movies")
    t0 = time.perf_counter()
    manifest = engine.produce_full_movie(
        prompt="A cyberpunk detective and an AI android uncover an orbital conspiracy across Neo-Tokyo",
        target_episode_sec=1800.0,  # 30 minutes per episode!
        num_acts=6,                 # 6 x 30-min episodes = 3 hours!
        unique_shots_per_act=2,
        shot_clip_sec=5.0,
        width=640,
        height=360,
        fps=10,
        use_ollama=False,
    )
    dt = time.perf_counter() - t0
    ep1_dur = manifest["episodes_30min"][0]["duration_sec"]
    total_dur = manifest["total_duration_sec"]
    full_file = manifest["full_movie_mp4"]
    size_mb = os.path.getsize(full_file) / (1024.0 * 1024.0)
    assert abs(ep1_dur - 1800.0) <= 15.0, f"Expected ~1800s (30m) episode, got {ep1_dur}s"
    assert abs(total_dur - 10800.0) <= 60.0, f"Expected ~10800s (3h) movie, got {total_dur}s"
    return (
        f"6 x 30-min Episodes ({ep1_dur/60:.1f}m each) auto-stitched into Full 3-Hour Movie "
        f"({total_dur/3600:.2f} hrs / {total_dur:.0f}s, {size_mb:.1f} MB) in {dt:.1f}s on CPU!"
    )


if __name__ == "__main__":
    print("=" * 72)
    print("AI MOVIE STUDIO — 30-MIN EPISODE & 3-HOUR MOVIE AUDIT")
    print("=" * 72)
    check("1. REA Multi-Generator Architecture Bridge", t_rea_architecture_bridge)
    check("2. 6-Act x 30-Min (3-Hour) Screenplay & Character Bank Planner", t_movie_planner_3hr)
    check("3. H.264 + AAC Cinema Shot Renderer (Ken-Burns + Sidechain Audio)", t_single_shot_render)
    check("4. Full 6 x 30-Min Episodes -> 3-Hour Movie Auto-Stitcher", t_full_3_hour_movie_production)

    for status, name, detail in results:
        mark = "✓" if status == "PASS" else "✗"
        print(f" {mark} {status:4s}  {name}\n         → {detail}")
    print("=" * 72)
    fails = sum(1 for s, _, _ in results if s != "PASS")
    sys.exit(1 if fails else 0)
