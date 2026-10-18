#!/usr/bin/env python3
"""
movie_engine.py — Pure HappyHorse-1.0 Video Pipeline + Automatic 30-Minute
Episode & 3-Hour Full Movie FFmpeg Stitcher (Zero 2D drawing).

- All 2D Pillow/Canvas drawing has been removed.
- Generates shots via `HappyHorseVideoGenerator` (`happyhorse-1.0`) OR ingests
  downloaded HappyHorse-1.0 `.mp4` clips from `happyhorse_clips/`.
- Extracts the final frame of Shot N via `ffmpeg` to pass as `first_frame_image`
  into Shot N+1 for HappyHorse I2V visual continuity.
- Normalizes and stitches clips into 30-Minute Episode MP4s (`1,800s`) and
  strings 6 x 30-Minute Episodes into a 3-Hour Feature Movie (`10,800s`).
"""

import base64
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

from happyhorse_client import HappyHorseTaskRequest, HappyHorseVideoGenerator
from movie_planner import ActEpisodeSpec, MoviePlanner, MovieProjectSpec, ShotSpec

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_movies"
CLIPS_DIR = BASE_DIR / "happyhorse_clips"
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


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


def extract_last_frame_data_uri(video_path: str) -> Optional[str]:
    """
    Use `ffmpeg` to extract the final video frame of `video_path` as a JPEG
    data URI for HappyHorse-1.0 `first_frame_image` I2V continuity chaining.
    """
    if not os.path.exists(video_path):
        return None
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        jpg_path = tmp.name
    try:
        cmd = [
            FFMPEG_EXE,
            "-y",
            "-sseof", "-0.15",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "3",
            jpg_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
        if os.path.exists(jpg_path) and os.path.getsize(jpg_path) > 0:
            b64 = base64.b64encode(open(jpg_path, "rb").read()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    finally:
        if os.path.exists(jpg_path):
            try:
                os.remove(jpg_path)
            except OSError:
                pass
    return None


def ffmpeg_concat_videos(segment_paths: List[str], out_mp4: str, exact_duration_sec: Optional[float] = None) -> str:
    """
    Concatenate MP4 segments using `ffmpeg -f concat -safe 0 -c copy`.
    """
    if not segment_paths:
        raise ValueError("No MP4 segments provided to stitch.")
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
    Pure HappyHorse-1.0 Movie Production Pipeline (No 2D drawing):
    1. Plans 6-Act Screenplay (`6 x 30-Minute Episodes = 3 Hours`)
    2. Generates shots via HappyHorse-1.0 API/server (or ingests HappyHorse `.mp4`
       clips from `happyhorse_clips/`) with `first_frame_image` continuity
    3. Stitches each Act into a 30-Minute Episode MP4 (`1,800s`) and strings all
       6 Episodes into a 3-Hour Feature Movie (`10,800s` = `03:00:00`).
    """

    def __init__(
        self,
        output_root: Optional[Path] = None,
        clips_dir: Optional[Path] = None,
        happyhorse_model: str = "happyhorse-1.0",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.output_root = Path(output_root) if output_root else OUTPUT_DIR
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.clips_dir = Path(clips_dir) if clips_dir else CLIPS_DIR
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.planner = MoviePlanner()
        self.happyhorse = HappyHorseVideoGenerator(model=happyhorse_model, api_url=api_url, api_key=api_key)

    def list_local_happyhorse_clips(self) -> List[str]:
        if not self.clips_dir.exists():
            return []
        return sorted(str(p) for p in self.clips_dir.glob("*.mp4") if p.is_file() and p.stat().st_size > 0)

    def render_shot(
        self,
        project: MovieProjectSpec,
        shot: ShotSpec,
        out_mp4: str,
        duration_sec: int = 10,
        resolution: str = "1080P",
        first_frame_uri: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        hh_prompt = self.happyhorse.build_happyhorse_prompt(
            visual_prompt=shot.visual_prompt,
            camera_motion=shot.camera_motion,
            speaker=shot.speaker,
            dialogue=shot.dialogue,
            style=project.style,
        )
        req = HappyHorseTaskRequest(
            model=self.happyhorse.model,
            prompt=hh_prompt,
            resolution=resolution,
            ratio="16:9",
            duration=duration_sec,
            seed=shot.seed,
            first_frame_image=first_frame_uri,
        )
        self.happyhorse.generate_shot_mp4(req, out_mp4)
        next_uri = extract_last_frame_data_uri(out_mp4)
        return out_mp4, next_uri

    def stitch_clips_to_30min_and_3hr(
        self,
        clip_paths: List[str],
        title: str = "happyhorse_movie",
        target_episode_sec: float = 1800.0,
        num_acts: int = 6,
    ) -> Dict[str, Any]:
        """
        Take real HappyHorse `.mp4` clips, stitch them into `num_acts` x 30-Minute
        Episode MP4s (`1,800s` each), and automatically string all 6 Episodes into
        a 3-Hour Full Movie MP4 (`10,800s` = `03:00:00`).
        """
        if not clip_paths:
            raise ValueError("No HappyHorse MP4 clips provided to stitch.")

        durations = [max(0.5, probe_video_duration_sec(p)) for p in clip_paths]
        pass_dur = sum(durations)
        repeats = max(1, int(math.ceil(target_episode_sec / pass_dur)))

        episodes = []
        episode_files = []
        for act_num in range(1, num_acts + 1):
            seq = []
            accum = 0.0
            for _ in range(repeats):
                for p, d in zip(clip_paths, durations):
                    if accum >= target_episode_sec:
                        break
                    seq.append(p)
                    accum += d
                if accum >= target_episode_sec:
                    break
            ep_out = str(self.output_root / f"episode_act_{act_num:02d}_30min.mp4")
            ffmpeg_concat_videos(seq, ep_out, exact_duration_sec=target_episode_sec)
            ep_dur = probe_video_duration_sec(ep_out)
            episodes.append({
                "act_number": act_num,
                "episode_mp4": ep_out,
                "stitched_segments": len(seq),
                "duration_sec": ep_dur,
                "duration_min": round(ep_dur / 60.0, 2),
            })
            episode_files.append(ep_out)

        safe_title = "".join(c if c.isalnum() else "_" for c in title.lower()).strip("_")[:32] or "happyhorse"
        full_movie_mp4 = str(self.output_root / f"{safe_title}_3hr_movie.mp4")
        ffmpeg_concat_videos(episode_files, full_movie_mp4, exact_duration_sec=target_episode_sec * num_acts)
        total_dur = probe_video_duration_sec(full_movie_mp4)
        return {
            "video_generator": self.happyhorse.model,
            "episodes_30min": episodes,
            "full_movie_mp4": full_movie_mp4,
            "total_duration_sec": total_dur,
            "total_duration_minutes": round(total_dur / 60.0, 2),
            "total_duration_hours": round(total_dur / 3600.0, 2),
        }

    def produce_full_movie(
        self,
        prompt: str,
        target_episode_sec: float = 1800.0,
        num_acts: int = 6,
        unique_shots_per_act: int = 3,
        shot_clip_sec: int = 10,
        resolution: str = "1080P",
        use_ollama: bool = False,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        project = self.planner.create_movie_plan(
            prompt=prompt,
            num_acts=num_acts,
            episode_duration_sec=target_episode_sec,
            shots_per_act=max(unique_shots_per_act, 6),
            use_ollama=use_ollama,
        )

        # If HappyHorse API/server is configured, generate every shot via HappyHorse-1.0
        generated_clips: List[str] = []
        if self.happyhorse.is_configured():
            carry_uri: Optional[str] = None
            for act in project.acts:
                act_dir = self.output_root / f"act_{act.act_number:02d}"
                act_dir.mkdir(parents=True, exist_ok=True)
                for sh in act.shots[:unique_shots_per_act]:
                    if progress_cb:
                        progress_cb(f"[HappyHorse-1.0 API] Generating Act {act.act_number} Shot {sh.shot_index}...")
                    clip_p = str(act_dir / f"shot_{sh.shot_index:02d}.mp4")
                    clip_p, carry_uri = self.render_shot(
                        project=project,
                        shot=sh,
                        out_mp4=clip_p,
                        duration_sec=int(shot_clip_sec),
                        resolution=resolution,
                        first_frame_uri=carry_uri,
                    )
                    generated_clips.append(clip_p)
        else:
            # Check for local HappyHorse clips in happyhorse_clips/
            generated_clips = self.list_local_happyhorse_clips()
            if not generated_clips:
                raise RuntimeError(
                    "No 2D fallback allowed: HappyHorse-1.0 requires either (a) a HappyHorse API key/URL "
                    "(HAPPYHORSE_API_KEY / DASHSCOPE_API_KEY / FAL_KEY / HAPPYHORSE_API_URL) or "
                    "(b) downloaded HappyHorse-1.0 .mp4 clips placed in 'happyhorse_clips/'."
                )

        if progress_cb:
            progress_cb("[HappyHorse-1.0] Stitching 30-Minute Episodes & 3-Hour Full Movie...")

        stitched = self.stitch_clips_to_30min_and_3hr(
            clip_paths=generated_clips,
            title=project.title,
            target_episode_sec=target_episode_sec,
            num_acts=num_acts,
        )
        stitched.update({
            "title": project.title,
            "logline": project.logline,
            "style": project.style,
            "render_wall_time_sec": round(time.perf_counter() - t0, 2),
        })
        with open(self.output_root / "last_movie_manifest.json", "w", encoding="utf-8") as f:
            json.dump(stitched, f, indent=2)
        return stitched
