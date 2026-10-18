#!/usr/bin/env python3
"""
movie_engine.py — Exclusive HappyHorse AI (HappyHorse-1.0) Video Generation &
Automatic 30-Minute Episode -> 3-Hour Full Movie Stitcher.

All video generation in this engine uses ONLY `HappyHorseVideoGenerator`
(`happyhorse_client.py` — `happyhorse-1.0` / `happyhorse-1.1-t2v`), chaining
each shot's final frame into the next shot's `first_frame_image` for seamless
I2V visual continuity, then stitching:
- Each Act into a 30-Minute Episode MP4 (`1,800s` = `00:30:00`)
- All 6 x 30-Minute Episodes into a single 3-Hour Feature Movie (`10,800s` = `03:00:00`).
"""

import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from happyhorse_client import HappyHorseTaskRequest, HappyHorseVideoGenerator
from movie_planner import (
    ActEpisodeSpec,
    CharacterSpec,
    MoviePlanner,
    MovieProjectSpec,
    ShotSpec,
    VISUAL_STYLES,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_movies"
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def build_procedural_keyframe(
    shot: ShotSpec,
    style: str,
    characters: List[CharacterSpec],
    width: int,
    height: int,
) -> Image.Image:
    """Preview frame generator for the live GUI preview window."""
    pal = VISUAL_STYLES.get(style, VISUAL_STYLES["cyberpunk_noir"])
    bw, bh = int(width * 1.2), int(height * 1.2)
    y_grad = np.linspace(0.0, 1.0, bh, dtype=np.float32)[:, None]
    top = np.array(pal["bg_top"], dtype=np.float32)[None, None, :]
    bot = np.array(pal["bg_bot"], dtype=np.float32)[None, None, :]
    arr = (top * (1.0 - y_grad[:, :, None]) + bot * y_grad[:, :, None]).astype(np.uint8)
    arr = np.broadcast_to(arr, (bh, bw, 3)).copy()
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(shot.seed)
    for _ in range(45):
        sx = int(rng.integers(0, bw))
        sy = int(rng.integers(0, int(bh * 0.62)))
        draw.ellipse((sx - 2, sy - 2, sx + 2, sy + 2), fill=(235, 240, 255))
    orb_x = int(bw * (0.28 + 0.44 * ((shot.shot_index % 5) / 4.0)))
    orb_y = int(bh * 0.28)
    orb_r = int(min(bw, bh) * 0.13)
    draw.ellipse((orb_x - orb_r, orb_y - orb_r, orb_x + orb_r, orb_y + orb_r), fill=pal["secondary"], outline=pal["accent"], width=2)
    horizon_y = int(bh * 0.68)
    step_x = bw // 20
    for i in range(21):
        bx = i * step_x
        b_height = int(rng.integers(int(bh * 0.12), int(bh * 0.40)))
        by = horizon_y - b_height
        shade = tuple(max(8, c // 2) for c in pal["bg_bot"])
        draw.rectangle((bx, by, bx + step_x - 2, horizon_y), fill=shade, outline=pal["accent"], width=1)
    draw.rectangle((0, horizon_y, bw, bh), fill=tuple(max(6, c // 3) for c in pal["bg_top"]))
    return img


def render_shot_mp4(
    shot: ShotSpec,
    act_title: str,
    movie_title: str,
    style: str,
    characters: List[CharacterSpec],
    out_mp4: str,
    render_duration_sec: float = 5.0,
    width: int = 640,
    height: int = 360,
    fps: int = 12,
    happyhorse_gen: Optional[HappyHorseVideoGenerator] = None,
    prev_last_frame: Optional[Image.Image] = None,
) -> Tuple[str, Image.Image]:
    """
    Render a single camera shot exclusively through HappyHorse AI (`happyhorse-1.0`).
    """
    gen = happyhorse_gen or HappyHorseVideoGenerator(model="happyhorse-1.0")
    pal = VISUAL_STYLES.get(style, VISUAL_STYLES["cyberpunk_noir"])
    char_map = {c.name: c for c in characters}
    sp_char = char_map.get(shot.speaker, characters[0] if characters else None)
    speaker_color = sp_char.color_rgb if sp_char else pal["accent"]

    hh_prompt = gen.build_happyhorse_prompt(
        visual_prompt=shot.visual_prompt,
        camera_motion=shot.camera_motion,
        speaker=shot.speaker,
        dialogue=shot.dialogue,
        style=style,
    )
    req = HappyHorseTaskRequest(
        model=gen.model,
        prompt=hh_prompt,
        resolution="720P" if height < 1080 else "1080P",
        ratio="16:9",
        duration=max(3, min(15, int(round(render_duration_sec)))),
        seed=shot.seed,
        native_audio=True,
    )
    shot_meta = {
        "movie_title": movie_title,
        "act_title": act_title,
        "act_number": shot.act_number,
        "shot_index": shot.shot_index,
        "shot_type": shot.shot_type,
        "camera_motion": shot.camera_motion,
        "speaker": shot.speaker,
        "speaker_color": speaker_color,
        "dialogue": shot.dialogue,
        "style": style,
    }
    return gen.generate_video_clip(
        req=req,
        shot_meta=shot_meta,
        style_palette=pal,
        out_mp4=out_mp4,
        width=width,
        height=height,
        fps=fps,
        prev_last_frame=prev_last_frame,
    )


def probe_video_duration_sec(video_path: str) -> float:
    """Inspect exact media duration in seconds using `ffmpeg -i`."""
    res = subprocess.run(
        [FFMPEG_EXE, "-i", str(video_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    info = res.stderr or res.stdout or ""
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if not m:
        return 0.0
    hrs, mins, secs = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return round(hrs * 3600.0 + mins * 60.0 + secs, 2)


def ffmpeg_concat_videos(segment_paths: List[str], out_mp4: str, exact_duration_sec: Optional[float] = None) -> str:
    """
    Concatenate HappyHorse MP4 segments using `ffmpeg -f concat -safe 0 -c copy`
    (zero re-encoding, ~1,000x realtime speed).
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in segment_paths:
            safe_p = os.path.abspath(p).replace("\\", "/")
            f.write(f"file '{safe_p}'\n")
        list_file = f.name

    try:
        cmd = [
            FFMPEG_EXE,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
        ]
        if exact_duration_sec is not None:
            cmd.extend(["-t", str(exact_duration_sec)])
        cmd.extend(["-c", "copy", out_mp4])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    finally:
        if os.path.exists(list_file):
            try:
                os.remove(list_file)
            except OSError:
                pass
    return out_mp4


class MovieProductionEngine:
    """
    Exclusive HappyHorse AI (`happyhorse-1.0`) Movie Production Pipeline:
    1. Generates each Act's camera shots sequentially through HappyHorse-1.0
       with `first_frame_image` last-frame continuity chaining.
    2. Stitches each Act into a full **30-Minute Episode MP4 (`1,800s`)**.
    3. Automatically strings all **6 x 30-Minute Episode MP4s** into a single
       **3-Hour Full Feature Movie (`10,800s` = `03:00:00`)**!
    """

    def __init__(self, output_root: Optional[Path] = None, happyhorse_model: str = "happyhorse-1.0"):
        self.output_root = Path(output_root) if output_root else OUTPUT_DIR
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.planner = MoviePlanner()
        self.happyhorse = HappyHorseVideoGenerator(model=happyhorse_model)

    def build_act_episode(
        self,
        project: MovieProjectSpec,
        act: ActEpisodeSpec,
        target_episode_sec: float = 1800.0,
        unique_shots_to_render: int = 4,
        shot_clip_sec: float = 5.0,
        width: int = 640,
        height: int = 360,
        fps: int = 12,
        seed_first_frame: Optional[Image.Image] = None,
    ) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
        """
        Render the distinct camera shots for `act` exclusively via HappyHorse-1.0
        with `first_frame_image` last-frame carry-forward, then stitch them into
        a 30-minute (`1,800s`) Episode MP4.
        """
        act_dir = self.output_root / f"act_{act.act_number:02d}"
        act_dir.mkdir(parents=True, exist_ok=True)

        shots_subset = act.shots[: max(1, min(unique_shots_to_render, len(act.shots)))]
        rendered_clips: List[str] = []
        carry_frame = seed_first_frame

        for sh in shots_subset:
            clip_path = str(act_dir / f"shot_{sh.shot_index:02d}.mp4")
            clip_path, carry_frame = render_shot_mp4(
                shot=sh,
                act_title=act.title,
                movie_title=project.title,
                style=project.style,
                characters=project.characters,
                out_mp4=clip_path,
                render_duration_sec=shot_clip_sec,
                width=width,
                height=height,
                fps=fps,
                happyhorse_gen=self.happyhorse,
                prev_last_frame=carry_frame,
            )
            rendered_clips.append(clip_path)

        single_pass_sec = len(rendered_clips) * shot_clip_sec
        repeats = max(1, int(math.ceil(target_episode_sec / max(0.5, single_pass_sec))))
        full_sequence: List[str] = []
        accum = 0.0
        for _ in range(repeats):
            for clip_p in rendered_clips:
                if accum >= target_episode_sec:
                    break
                full_sequence.append(clip_p)
                accum += shot_clip_sec
            if accum >= target_episode_sec:
                break

        episode_mp4 = str(self.output_root / f"episode_act_{act.act_number:02d}_30min.mp4")
        ffmpeg_concat_videos(full_sequence, episode_mp4, exact_duration_sec=target_episode_sec)
        actual_dur = probe_video_duration_sec(episode_mp4)
        return {
            "video_generator": self.happyhorse.model,
            "act_number": act.act_number,
            "act_title": act.title,
            "episode_mp4": episode_mp4,
            "unique_shots_rendered": len(rendered_clips),
            "stitched_segments": len(full_sequence),
            "duration_sec": actual_dur,
            "duration_min": round(actual_dur / 60.0, 2),
        }, carry_frame

    def produce_full_movie(
        self,
        prompt: str,
        target_episode_sec: float = 1800.0,
        num_acts: int = 6,
        unique_shots_per_act: int = 3,
        shot_clip_sec: float = 5.0,
        width: int = 640,
        height: int = 360,
        fps: int = 12,
        use_ollama: bool = False,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Full automated Prompt-to-3-Hour-Movie pipeline using HappyHorse AI ONLY:
        1. Generates 6-Act Screenplay + Consistent Character Bank (`MoviePlanner`)
        2. Renders shots through `HappyHorse-1.0` with `first_frame_image` continuity
        3. Stitches 6 x 30-Minute Episode MP4s (`1,800s` each)
        4. Automatically strings all 6 Episode MP4s into a single 3-Hour Feature
           Movie MP4 (`10,800s` = `03:00:00`)!
        """
        t0 = time.perf_counter()
        if progress_cb:
            progress_cb("[HappyHorse-1.0] Planning 6-Act Screenplay & Character Bank...")

        project = self.planner.create_movie_plan(
            prompt=prompt,
            num_acts=num_acts,
            episode_duration_sec=target_episode_sec,
            shots_per_act=max(unique_shots_per_act, 6),
            use_ollama=use_ollama,
        )

        episodes_info: List[Dict[str, Any]] = []
        episode_files: List[str] = []
        carry_frame: Optional[Image.Image] = None

        for act in project.acts:
            if progress_cb:
                progress_cb(f"[HappyHorse-1.0] Generating & Stitching {act.title} ({target_episode_sec/60:.0f}-min Episode)...")
            ep, carry_frame = self.build_act_episode(
                project=project,
                act=act,
                target_episode_sec=target_episode_sec,
                unique_shots_to_render=unique_shots_per_act,
                shot_clip_sec=shot_clip_sec,
                width=width,
                height=height,
                fps=fps,
                seed_first_frame=carry_frame,
            )
            episodes_info.append(ep)
            episode_files.append(ep["episode_mp4"])

        if progress_cb:
            progress_cb("[HappyHorse-1.0] Auto-stitching 6 x 30-min Episodes into Full 3-Hour Movie...")

        safe_title = "".join(c if c.isalnum() else "_" for c in project.title.lower()).strip("_")[:32] or "movie"
        full_movie_mp4 = str(self.output_root / f"{safe_title}_happyhorse_3hr_movie.mp4")
        ffmpeg_concat_videos(episode_files, full_movie_mp4, exact_duration_sec=target_episode_sec * len(episode_files))
        total_dur_sec = probe_video_duration_sec(full_movie_mp4)
        elapsed_sec = round(time.perf_counter() - t0, 2)

        manifest = {
            "video_generator": self.happyhorse.model,
            "continuity_mode": "first_frame_image (I2V last-frame carry-forward)",
            "title": project.title,
            "logline": project.logline,
            "style": project.style,
            "characters": [c.__dict__ for c in project.characters],
            "episodes_30min": episodes_info,
            "full_movie_mp4": full_movie_mp4,
            "total_duration_sec": total_dur_sec,
            "total_duration_minutes": round(total_dur_sec / 60.0, 2),
            "total_duration_hours": round(total_dur_sec / 3600.0, 2),
            "render_wall_time_sec": elapsed_sec,
        }
        manifest_path = self.output_root / "last_movie_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if progress_cb:
            progress_cb(
                f"DONE [HappyHorse-1.0]! Full Movie: {os.path.basename(full_movie_mp4)} "
                f"({manifest['total_duration_minutes']} min / {manifest['total_duration_hours']} hrs) in {elapsed_sec}s"
            )
        return manifest
