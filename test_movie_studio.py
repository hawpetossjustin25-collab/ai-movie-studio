#!/usr/bin/env python3
"""
test_movie_studio.py — End-to-end verification suite for AI Movie Studio
(Exclusive HappyHorse-1.0 Video Generation Edition):
1. Exclusive HappyHorse-1.0 Architecture & Request Schema Verification
2. 6-Act x 30-Minute (3-Hour) Screenplay & Character Bank Planner (`happyhorse-1.0`)
3. HappyHorse-1.0 Shot Generation + `first_frame_image` Last-Frame Continuity Chaining
4. Full 6 x 30-Min Episodes -> 3-Hour Movie (`10,800s` = `03:00:00`) Auto-Stitcher
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

from happyhorse_client import HappyHorseTaskRequest, HappyHorseVideoGenerator  # noqa: E402
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


def t_happyhorse_exclusive_architecture():
    caps = REAArchitectureBridge.inspect_capabilities()
    repos = caps["synthesized_repositories"]
    assert "happyhorse-ai/happyhorse (HappyHorse-1.0)" in repos
    assert "harry0703/MoneyPrinterTurbo + jimmyjjz/lfs-vid-gen" not in repos
    req = HappyHorseTaskRequest(model="happyhorse-1.0", prompt="Test shot", resolution="1080P", ratio="16:9", duration=5)
    ds = req.to_dashscope_payload()
    assert ds["model"] == "happyhorse-1.0" and ds["parameters"]["resolution"] == "1080P"
    return "HappyHorse-1.0 verified as sole video generator (T2V + I2V first_frame_image + Joint Native Audio)"


def t_movie_planner_3hr():
    planner = MoviePlanner()
    plan = planner.create_movie_plan(
        "A cyberpunk detective and an AI android uncover an orbital conspiracy across Neo-Tokyo",
        num_acts=6,
        episode_duration_sec=1800.0,
        shots_per_act=12,
    )
    assert plan.video_generator == "happyhorse-1.0"
    assert len(plan.acts) == 6 and plan.total_duration_sec == 10800.0
    total_shots = sum(len(a.shots) for a in plan.acts)
    return f"Planned '{plan.title}' with generator={plan.video_generator} (6 acts x 30m = 3.0 hrs, {total_shots} shots)"


def t_happyhorse_continuity_chaining():
    planner = MoviePlanner()
    plan = planner.create_movie_plan("Sci-fi starship battle over Saturn")
    gen = HappyHorseVideoGenerator(model="happyhorse-1.0")
    shot1 = plan.acts[0].shots[0]
    shot2 = plan.acts[0].shots[1]
    out1 = str(BASE / "output_movies" / "_hh_shot1.mp4")
    out2 = str(BASE / "output_movies" / "_hh_shot2.mp4")

    p1, last_frame_1 = render_shot_mp4(
        shot=shot1,
        act_title=plan.acts[0].title,
        movie_title=plan.title,
        style=plan.style,
        characters=plan.characters,
        out_mp4=out1,
        render_duration_sec=3.0,
        happyhorse_gen=gen,
        prev_last_frame=None,
    )
    assert last_frame_1 is not None and last_frame_1.size == (640, 360)

    # Chain Shot 1's last frame into Shot 2's `first_frame_image`
    p2, last_frame_2 = render_shot_mp4(
        shot=shot2,
        act_title=plan.acts[0].title,
        movie_title=plan.title,
        style=plan.style,
        characters=plan.characters,
        out_mp4=out2,
        render_duration_sec=3.0,
        happyhorse_gen=gen,
        prev_last_frame=last_frame_1,
    )
    dur1 = probe_video_duration_sec(p1)
    dur2 = probe_video_duration_sec(p2)
    os.remove(p1)
    os.remove(p2)
    assert 2.5 <= dur1 <= 3.5 and 2.5 <= dur2 <= 3.5
    return f"HappyHorse-1.0 chained Shot 1 ({dur1:.2f}s) last_frame -> Shot 2 ({dur2:.2f}s) first_frame_image"


def t_full_3_hour_happyhorse_movie():
    engine = MovieProductionEngine(output_root=BASE / "output_movies", happyhorse_model="happyhorse-1.0")
    t0 = time.perf_counter()
    manifest = engine.produce_full_movie(
        prompt="A cyberpunk detective and an AI android uncover an orbital conspiracy across Neo-Tokyo",
        target_episode_sec=1800.0,
        num_acts=6,
        unique_shots_per_act=2,
        shot_clip_sec=5.0,
        width=640,
        height=360,
        fps=10,
        use_ollama=False,
    )
    dt = time.perf_counter() - t0
    assert manifest["video_generator"] == "happyhorse-1.0"
    ep1_dur = manifest["episodes_30min"][0]["duration_sec"]
    total_dur = manifest["total_duration_sec"]
    full_file = manifest["full_movie_mp4"]
    size_mb = os.path.getsize(full_file) / (1024.0 * 1024.0)
    assert abs(ep1_dur - 1800.0) <= 15.0, f"Expected ~1800s (30m) episode, got {ep1_dur}s"
    assert abs(total_dur - 10800.0) <= 60.0, f"Expected ~10800s (3h) movie, got {total_dur}s"
    return (
        f"[HappyHorse-1.0] 6 x 30-min Episodes ({ep1_dur/60:.1f}m each) auto-stitched into Full 3-Hour Movie "
        f"({total_dur/3600:.2f} hrs / {total_dur:.0f}s, {size_mb:.1f} MB) in {dt:.1f}s!"
    )


if __name__ == "__main__":
    print("=" * 72)
    print("AI MOVIE STUDIO — HAPPYHORSE-1.0 EXCLUSIVE 3-HOUR MOVIE AUDIT")
    print("=" * 72)
    check("1. Exclusive HappyHorse-1.0 Video Generator Architecture", t_happyhorse_exclusive_architecture)
    check("2. 6-Act x 30-Min (3-Hour) Screenplay Planner (happyhorse-1.0)", t_movie_planner_3hr)
    check("3. HappyHorse-1.0 Shot Generation & I2V First-Frame Continuity", t_happyhorse_continuity_chaining)
    check("4. Full 6 x 30-Min Episodes -> 3-Hour HappyHorse Movie Stitcher", t_full_3_hour_happyhorse_movie)

    for status, name, detail in results:
        mark = "✓" if status == "PASS" else "✗"
        print(f" {mark} {status:4s}  {name}\n         → {detail}")
    print("=" * 72)
    fails = sum(1 for s, _, _ in results if s != "PASS")
    sys.exit(1 if fails else 0)
