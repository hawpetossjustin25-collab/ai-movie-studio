#!/usr/bin/env python3
"""
movie_studio_app.py — AI Movie Studio Desktop Application (PyQt6) & CLI.

Prompt -> 6-Act Screenplay (30-Min Episodes) -> Parallel CPU Shot Synthesis ->
Sidechain-Ducked Audio Mix -> Zero-Reencode FFmpeg Auto-Stitch into a Full
3-Hour Feature Movie (`full_3hr_movie.mp4`).
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from movie_planner import MoviePlanner, REAArchitectureBridge, VISUAL_STYLES
from movie_engine import MovieProductionEngine, OUTPUT_DIR, build_procedural_keyframe

BASE_DIR = Path(__file__).resolve().parent


def run_gui():
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPainter
    from PyQt6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QCheckBox,
    )

    class Signals(QObject):
        progress = pyqtSignal(str)
        finished = pyqtSignal(dict)

    class CinemaPreviewWidget(QLabel):
        """Live animated widescreen cinema preview of the current MovieProjectSpec."""

        def __init__(self):
            super().__init__()
            self.setMinimumSize(720, 405)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setStyleSheet("background:#05070e; border:2px solid #38bdf8; border-radius:10px;")
            self.project = None
            self.frame_idx = 0
            self.cached_base = None
            self.cached_shot_key = None

        def set_project(self, proj):
            self.project = proj
            self.frame_idx = 0
            self.cached_shot_key = None

        def tick_preview(self):
            if not self.project or not self.project.acts:
                return
            self.frame_idx += 1
            act_idx = (self.frame_idx // 90) % len(self.project.acts)
            act = self.project.acts[act_idx]
            shot_idx = (self.frame_idx // 30) % len(act.shots)
            shot = act.shots[shot_idx]
            key = (act.act_number, shot.shot_index, self.project.title)
            if key != self.cached_shot_key or self.cached_base is None:
                self.cached_base = build_procedural_keyframe(
                    shot, self.project.style, self.project.characters, 640, 360
                )
                self.cached_shot_key = key

            bw, bh = self.cached_base.size
            prog = (self.frame_idx % 30) / 30.0
            ox = int((bw - 640) * prog)
            oy = (bh - 360) // 2
            crop = self.cached_base.crop((ox, oy, ox + 640, oy + 360)).copy()

            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(crop)
            pal = VISUAL_STYLES.get(self.project.style, VISUAL_STYLES["cyberpunk_noir"])
            bar_h = 46
            draw.rectangle((0, 0, 640, bar_h), fill=(6, 8, 14))
            draw.rectangle((0, 360 - bar_h, 640, 360), fill=(6, 8, 14))
            font = ImageFont.load_default()
            draw.text(
                (12, 10),
                f"[HappyHorse-1.0] {self.project.title[:28]}  |  {act.title[:28]} (30m)  |  Shot {shot.shot_index}",
                fill=pal["accent"],
                font=font,
            )
            draw.text(
                (14, 360 - bar_h + 14),
                f"{shot.speaker.upper()}: \"{shot.dialogue[:68]}\"",
                fill=(255, 255, 255),
                font=font,
            )

            raw = crop.tobytes("raw", "RGB")
            qimg = QImage(raw, 640, 360, 640 * 3, QImage.Format.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qimg).scaled(self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio))

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.engine = MovieProductionEngine()
            self.planner = self.engine.planner
            self.signals = Signals()
            self.signals.progress.connect(self._on_progress)
            self.signals.finished.connect(self._on_finished)
            self.last_manifest = None

            self.project = self.planner.create_movie_plan(
                "A cyberpunk detective and an AI android uncover an orbital conspiracy across Neo-Tokyo"
            )

            self.setWindowTitle("AI Movie Studio — HappyHorse-1.0 Exclusive Video Generator (30-Min Episodes -> 3-Hour Movies)")
            self.resize(1200, 720)
            self.setStyleSheet("background-color:#090d16; color:#e2e8f0; font-family:Segoe UI;")

            central = QWidget()
            self.setCentralWidget(central)
            root = QHBoxLayout(central)

            # Left column: Live Cinema Screen + Prompt Bar + Presets
            left = QVBoxLayout()
            self.preview = CinemaPreviewWidget()
            self.preview.set_project(self.project)
            left.addWidget(self.preview, stretch=1)

            pbar = QHBoxLayout()
            self.prompt_in = QLineEdit()
            self.prompt_in.setText("A cyberpunk detective and an AI android uncover an orbital conspiracy across Neo-Tokyo")
            self.prompt_in.setPlaceholderText("Type your movie prompt — auto-builds 6 x 30-min Episodes and stitches a Full 3-Hour Movie...")
            self.prompt_in.setStyleSheet("background:#131f37; border:1px solid #38bdf8; border-radius:6px; padding:10px; font-size:14px; color:#fff;")
            pbar.addWidget(self.prompt_in, stretch=1)

            btn_plan = QPushButton("📝 Plan 6-Act Screenplay")
            btn_plan.setStyleSheet("background:#0284c7; color:#fff; font-weight:bold; padding:10px 12px; border-radius:6px;")
            btn_plan.clicked.connect(self.on_plan_only)
            pbar.addWidget(btn_plan)

            self.btn_render = QPushButton("🎬 Render 6x 30-Min Episodes + Stitch 3-Hour Movie (.mp4)")
            self.btn_render.setStyleSheet("background:#059669; color:#fff; font-weight:bold; padding:10px 14px; border-radius:6px;")
            self.btn_render.clicked.connect(self.on_render_full_movie)
            pbar.addWidget(self.btn_render)
            left.addLayout(pbar)

            # Presets row
            prow = QHBoxLayout()
            presets = [
                ("🕵️ Cyberpunk Noir 3-Hr Saga", "A cyberpunk detective and an AI android uncover an orbital conspiracy across Neo-Tokyo"),
                ("🚀 Epic Sci-Fi Odyssey", "The last starship crew crosses a collapsing wormhole to save Earth from an ancient cosmic fleet"),
                ("🐉 Dark Fantasy Kingdom", "An exiled knight and a rogue sorceress unite six shattered realms against the Necromancer Dragon"),
                ("⚡ Action Thriller", "A former special-ops agent races across six global cities to disarm a rogue satellite network"),
            ]
            for label, txt in presets:
                b = QPushButton(label)
                b.setStyleSheet("background:#1e293b; color:#38bdf8; padding:6px 10px; border-radius:5px; font-size:12px;")
                b.clicked.connect(lambda _c=False, t=txt: self._use_preset(t))
                prow.addWidget(b)
            left.addLayout(prow)
            root.addLayout(left, stretch=3)

            # Right column: REA Architecture Status + 6-Act Inspector + Open Folder
            right = QVBoxLayout()
            rea_info = REAArchitectureBridge.inspect_capabilities()
            self.status_lbl = QLabel(
                f"Video Generator: HappyHorse AI ONLY (happyhorse-1.0)\n"
                f"Continuity: I2V first_frame_image + Joint Native Audio\n"
                f"Target: 6 x 30-Min Acts -> Auto-Stitched 3-Hour MP4"
            )
            self.status_lbl.setStyleSheet("background:#131f37; border:1px solid #334155; border-radius:8px; padding:10px; font-weight:bold; color:#34d399;")
            right.addWidget(self.status_lbl)

            self.chk_ollama = QCheckBox("Use Local Ollama (llama3.2:3b) for Screenplay Naming")
            self.chk_ollama.setChecked(False)
            right.addWidget(self.chk_ollama)

            self.info_box = QTextEdit()
            self.info_box.setReadOnly(True)
            self.info_box.setStyleSheet("background:#070b14; border:1px solid #1e293b; border-radius:8px; font-family:Consolas; font-size:11px; color:#7dd3fc;")
            right.addWidget(self.info_box, stretch=1)

            btn_open = QPushButton("📂 Open Output Movies Folder / Play 3-Hour MP4")
            btn_open.setStyleSheet("background:#d97706; color:#fff; font-weight:bold; padding:10px; border-radius:6px;")
            btn_open.clicked.connect(self.on_open_output)
            right.addWidget(btn_open)

            root.addLayout(right, stretch=1)
            self._refresh_info()

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.preview.tick_preview)
            self.timer.start(80)

        def _use_preset(self, txt: str):
            self.prompt_in.setText(txt)
            self.on_plan_only()

        def on_plan_only(self):
            txt = self.prompt_in.text().strip()
            if not txt:
                return
            self.project = self.planner.create_movie_plan(
                txt, num_acts=6, episode_duration_sec=1800.0, shots_per_act=12, use_ollama=self.chk_ollama.isChecked()
            )
            self.preview.set_project(self.project)
            self._refresh_info()

        def on_render_full_movie(self):
            txt = self.prompt_in.text().strip()
            if not txt:
                return
            self.btn_render.setEnabled(False)
            use_ol = self.chk_ollama.isChecked()

            def _worker():
                manifest = self.engine.produce_full_movie(
                    prompt=txt,
                    target_episode_sec=1800.0,
                    num_acts=6,
                    unique_shots_per_act=3,
                    shot_clip_sec=4.0,
                    use_ollama=use_ol,
                    progress_cb=lambda msg: self.signals.progress.emit(msg),
                )
                self.signals.finished.emit(manifest)

            threading.Thread(target=_worker, daemon=True).start()

        def _on_progress(self, msg: str):
            self.status_lbl.setText(msg)

        def _on_finished(self, manifest: dict):
            self.last_manifest = manifest
            self.btn_render.setEnabled(True)
            self.status_lbl.setText(
                f"✓ MOVIE COMPLETE ({manifest['total_duration_hours']} hrs / {manifest['total_duration_minutes']} min)\n"
                f"File: {os.path.basename(manifest['full_movie_mp4'])}\n"
                f"Rendered & Stitched in {manifest['render_wall_time_sec']}s on CPU"
            )
            self.info_box.setPlainText(json.dumps(manifest, indent=2))

        def on_open_output(self):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            if self.last_manifest and os.path.exists(self.last_manifest.get("full_movie_mp4", "")):
                os.startfile(self.last_manifest["full_movie_mp4"])
            else:
                os.startfile(str(OUTPUT_DIR))

        def _refresh_info(self):
            summary = {
                "title": self.project.title,
                "style": self.project.style,
                "episode_duration_min": self.project.episode_duration_sec / 60.0,
                "total_movie_hours": self.project.total_duration_sec / 3600.0,
                "characters": [c.name + f" ({c.role})" for c in self.project.characters],
                "acts_30min_each": [
                    {"act": a.act_number, "title": a.title, "shots": len(a.shots), "duration_min": a.target_duration_sec / 60.0}
                    for a in self.project.acts
                ],
                "rea_synthesized_engines": list(REAArchitectureBridge.inspect_capabilities()["synthesized_repositories"].keys()),
            }
            self.info_box.setPlainText(json.dumps(summary, indent=2))

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


def main():
    if "--produce" in sys.argv:
        idx = sys.argv.index("--produce")
        prompt = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "Cyberpunk detective Neo-Tokyo 3-hour saga"
        eng = MovieProductionEngine()
        res = eng.produce_full_movie(prompt, progress_cb=print)
        print(json.dumps(res, indent=2))
        return
    run_gui()


if __name__ == "__main__":
    main()
