#!/usr/bin/env python3
"""
movie_studio_app.py — HappyHorse-1.0 AI Movie Studio (Zero 2D drawing).

1. Plans a 6-Act x 30-Minute (3-Hour) Screenplay with HappyHorse-1.0 native
   camera & joint-audio prompts (`happyhorse-1.0`).
2. Generates video clips via HappyHorse-1.0 API (`DASHSCOPE_API_KEY` / `FAL_KEY`
   / `HAPPYHORSE_API_URL`) OR ingests real HappyHorse-1.0 `.mp4` clips placed
   in `happyhorse_clips/`.
3. Automatically stitches clips into 6 x 30-Minute Episode MP4s (`1,800s`) and
   strings all 6 Episodes into a Full 3-Hour Movie (`10,800s` = `03:00:00`).
"""

import json
import os
from pathlib import Path
import shutil
import sys
import threading
import webbrowser

from happyhorse_client import HappyHorseVideoGenerator
from movie_planner import MoviePlanner, REAArchitectureBridge
from movie_engine import MovieProductionEngine, OUTPUT_DIR, CLIPS_DIR

BASE_DIR = Path(__file__).resolve().parent


def run_gui():
    from PyQt6.QtCore import pyqtSignal, QObject
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    class Signals(QObject):
        progress = pyqtSignal(str)
        finished = pyqtSignal(dict)
        failed = pyqtSignal(str)

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.engine = MovieProductionEngine()
            self.planner = self.engine.planner
            self.signals = Signals()
            self.signals.progress.connect(self._on_progress)
            self.signals.finished.connect(self._on_finished)
            self.signals.failed.connect(self._on_failed)
            self.last_manifest = None

            self.project = self.planner.create_movie_plan(
                "A cyberpunk detective and an AI android uncover an orbital conspiracy across Neo-Tokyo"
            )

            self.setWindowTitle("AI Movie Studio — Pure HappyHorse-1.0 Video Generator & 30m->3h Stitcher (Zero 2D)")
            self.resize(1120, 700)
            self.setStyleSheet("background-color:#090d16; color:#e2e8f0; font-family:Segoe UI;")

            central = QWidget()
            self.setCentralWidget(central)
            root = QVBoxLayout(central)

            # Top status banner
            self.status_lbl = QLabel(
                "Video Generator: HappyHorse-1.0 ONLY (2D rendering completely removed)  |  "
                f"Local HappyHorse MP4 clips in happyhorse_clips/: {len(self.engine.list_local_happyhorse_clips())}"
            )
            self.status_lbl.setStyleSheet("background:#131f37; border:1px solid #38bdf8; border-radius:8px; padding:10px; font-weight:bold; color:#34d399;")
            root.addWidget(self.status_lbl)

            # HappyHorse API Key / URL configuration row
            cfg_row = QHBoxLayout()
            self.api_key_in = QLineEdit(self.engine.happyhorse.api_key)
            self.api_key_in.setPlaceholderText("Optional: HappyHorse API Key (DASHSCOPE_API_KEY / FAL_KEY / HAPPYHORSE_API_KEY)...")
            self.api_key_in.setEchoMode(QLineEdit.EchoMode.Password)
            self.api_key_in.setStyleSheet("background:#131f37; border:1px solid #334155; border-radius:6px; padding:8px; color:#fff;")
            cfg_row.addWidget(self.api_key_in, stretch=2)

            self.api_url_in = QLineEdit(self.engine.happyhorse.api_url)
            self.api_url_in.setPlaceholderText("Optional: Custom/Self-Hosted HappyHorse Endpoint URL...")
            self.api_url_in.setStyleSheet("background:#131f37; border:1px solid #334155; border-radius:6px; padding:8px; color:#fff;")
            cfg_row.addWidget(self.api_url_in, stretch=2)

            btn_web = QPushButton("🌐 Open HappyHorse Online (happyhorses.io)")
            btn_web.setStyleSheet("background:#7c3aed; color:#fff; font-weight:bold; padding:8px 12px; border-radius:6px;")
            btn_web.clicked.connect(lambda: webbrowser.open("https://happyhorses.io"))
            cfg_row.addWidget(btn_web)

            btn_import = QPushButton("📥 Import HappyHorse .mp4 Clips")
            btn_import.setStyleSheet("background:#0d9488; color:#fff; font-weight:bold; padding:8px 12px; border-radius:6px;")
            btn_import.clicked.connect(self.on_import_clips)
            cfg_row.addWidget(btn_import)
            root.addLayout(cfg_row)

            # Movie Prompt Row
            pbar = QHBoxLayout()
            self.prompt_in = QLineEdit("A cyberpunk detective and an AI android uncover an orbital conspiracy across Neo-Tokyo")
            self.prompt_in.setStyleSheet("background:#131f37; border:1px solid #38bdf8; border-radius:6px; padding:10px; font-size:14px; color:#fff;")
            pbar.addWidget(self.prompt_in, stretch=1)

            self.chk_ollama = QCheckBox("Use Ollama (llama3.2:3b)")
            pbar.addWidget(self.chk_ollama)

            btn_plan = QPushButton("📝 Build 6-Act HappyHorse Prompts")
            btn_plan.setStyleSheet("background:#0284c7; color:#fff; font-weight:bold; padding:10px 12px; border-radius:6px;")
            btn_plan.clicked.connect(self.on_plan)
            pbar.addWidget(btn_plan)

            self.btn_produce = QPushButton("🎬 Generate / Stitch 6x 30-Min Episodes -> 3-Hour Movie (.mp4)")
            self.btn_produce.setStyleSheet("background:#059669; color:#fff; font-weight:bold; padding:10px 14px; border-radius:6px;")
            self.btn_produce.clicked.connect(self.on_produce)
            pbar.addWidget(self.btn_produce)
            root.addLayout(pbar)

            # Main Inspector & Shot Prompt View (No 2D canvas)
            self.info_box = QTextEdit()
            self.info_box.setReadOnly(True)
            self.info_box.setStyleSheet("background:#070b14; border:1px solid #1e293b; border-radius:8px; font-family:Consolas; font-size:12px; color:#7dd3fc;")
            root.addWidget(self.info_box, stretch=1)

            bot_row = QHBoxLayout()
            btn_clips_dir = QPushButton("📁 Open happyhorse_clips/ Folder")
            btn_clips_dir.setStyleSheet("background:#1e293b; color:#38bdf8; font-weight:bold; padding:10px; border-radius:6px;")
            btn_clips_dir.clicked.connect(lambda: os.startfile(str(CLIPS_DIR)))
            bot_row.addWidget(btn_clips_dir)

            btn_out_dir = QPushButton("📂 Open output_movies/ Folder & Play 3-Hour MP4")
            btn_out_dir.setStyleSheet("background:#d97706; color:#fff; font-weight:bold; padding:10px; border-radius:6px;")
            btn_out_dir.clicked.connect(self.on_open_output)
            bot_row.addWidget(btn_out_dir)
            root.addLayout(bot_row)

            self._refresh_view()

        def on_import_clips(self):
            files, _ = QFileDialog.getOpenFileNames(self, "Select HappyHorse .mp4 Clips", "", "MP4 Video (*.mp4)")
            for f in files:
                shutil.copy2(f, CLIPS_DIR / os.path.basename(f))
            self._refresh_view()

        def on_plan(self):
            txt = self.prompt_in.text().strip()
            if not txt:
                return
            self.project = self.planner.create_movie_plan(
                txt, num_acts=6, episode_duration_sec=1800.0, shots_per_act=12, use_ollama=self.chk_ollama.isChecked()
            )
            self._refresh_view()

        def on_produce(self):
            txt = self.prompt_in.text().strip()
            if not txt:
                return
            self.engine.happyhorse.api_key = self.api_key_in.text().strip() or self.engine.happyhorse.api_key
            self.engine.happyhorse.api_url = self.api_url_in.text().strip() or self.engine.happyhorse.api_url
            self.btn_produce.setEnabled(False)

            def _worker():
                try:
                    manifest = self.engine.produce_full_movie(
                        prompt=txt,
                        target_episode_sec=1800.0,
                        num_acts=6,
                        unique_shots_per_act=3,
                        shot_clip_sec=10,
                        use_ollama=self.chk_ollama.isChecked(),
                        progress_cb=lambda m: self.signals.progress.emit(m),
                    )
                    self.signals.finished.emit(manifest)
                except Exception as exc:
                    self.signals.failed.emit(str(exc))

            threading.Thread(target=_worker, daemon=True).start()

        def _on_progress(self, msg: str):
            self.status_lbl.setText(msg)

        def _on_finished(self, manifest: dict):
            self.last_manifest = manifest
            self.btn_produce.setEnabled(True)
            self.status_lbl.setText(
                f"✓ HAPPYHORSE-1.0 3-HOUR MOVIE READY ({manifest['total_duration_hours']} hrs / {manifest['total_duration_minutes']} min) -> {os.path.basename(manifest['full_movie_mp4'])}"
            )
            self.info_box.setPlainText(json.dumps(manifest, indent=2))

        def _on_failed(self, err: str):
            self.btn_produce.setEnabled(True)
            self.status_lbl.setText(f"HappyHorse-1.0 Notice: {err}")

        def on_open_output(self):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            if self.last_manifest and os.path.exists(self.last_manifest.get("full_movie_mp4", "")):
                os.startfile(self.last_manifest["full_movie_mp4"])
            else:
                os.startfile(str(OUTPUT_DIR))

        def _refresh_view(self):
            clips = self.engine.list_local_happyhorse_clips()
            self.status_lbl.setText(
                "Video Generator: HappyHorse-1.0 ONLY (Zero 2D rendering)  |  "
                f"Imported HappyHorse .mp4 clips in happyhorse_clips/: {len(clips)}"
            )
            hh_prompts = []
            for a in self.project.acts:
                s0 = a.shots[0]
                hh_prompts.append({
                    "act": f"Act {a.act_number} (30-Min Episode) — {a.title}",
                    "happyhorse_1_0_prompt": HappyHorseVideoGenerator.build_happyhorse_prompt(
                        s0.visual_prompt, s0.camera_motion, s0.speaker, s0.dialogue, self.project.style
                    ),
                    "shots_in_act": len(a.shots),
                })
            view = {
                "video_generator": "happyhorse-1.0 (EXCLUSIVE — 2D rendering removed)",
                "title": self.project.title,
                "episode_duration": "30.0 minutes (1,800s) per Act",
                "full_movie_duration": "3.0 hours (10,800s) across 6 Acts",
                "local_happyhorse_clips_ready": [os.path.basename(c) for c in clips],
                "acts_and_happyhorse_prompts": hh_prompts,
            }
            self.info_box.setPlainText(json.dumps(view, indent=2))

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
