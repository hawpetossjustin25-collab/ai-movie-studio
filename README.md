# 🎬 AI Movie Studio — 30-Minute Episode & 3-Hour Full Movie Generator (100% Free Forever, CPU-First)

Type your movie idea in plain English — **AI Movie Studio** plans a **6-Act Screenplay** (`6 × 30-Minute Episodes = 3 Hours`) with a persistent Character Bank, synthesizes widescreen 2.5D Ken-Burns shots, multi-character dialogue, sidechain-ducked cinematic soundtrack, and burned-in subtitles on CPU, and **automatically stitches all 6 × 30-minute episodes into a single 3-Hour (`03:00:00`) `.mp4` feature movie**!

100% free forever. Runs on CPU-only machines (no GPU required) and in any phone or desktop browser.

---

## 📱 Phone & Web App (Watch & Create 3-Hour Movies Instantly — No Install Needed)

👉 **[Launch Phone / Web App: https://hawpetossjustin25-collab.github.io/ai-movie-studio/](https://hawpetossjustin25-collab.github.io/ai-movie-studio/)**

Open that link on your phone or computer to generate a 6-Act (`6 × 30-Min = 3-Hour`) movie from any text prompt, watch the live widescreen cinema player with synthesized music and spoken dialogue, jump between Acts 1–6, and record `.webm` video clips directly in your browser.

**To install it on your phone's home screen like a native app:**
- **Android (Chrome):** Open the link → tap Menu (**⋮**) → **Add to Home screen**
- **iPhone (Safari):** Open the link → tap **Share** → **Add to Home Screen**

---

## 💻 Installation (All Methods)

### Option 1 — `pip install` directly from GitHub
```bash
pip install git+https://github.com/hawpetossjustin25-collab/ai-movie-studio.git
ai-movie-studio
```

### Option 2 — Clone & Install with `requirements.txt`
```bash
git clone https://github.com/hawpetossjustin25-collab/ai-movie-studio.git
cd ai-movie-studio
pip install -r requirements.txt
python movie_studio_app.py
```

### Option 3 — Command-Line Headless 3-Hour Movie Producer
```bash
python movie_studio_app.py --produce "A cyberpunk detective and an AI android uncover an orbital conspiracy across Neo-Tokyo"
```

### Option 4 — One-Click Windows Download (`.zip` + `.bat` Launcher)
1. **[Click here to download the ZIP](https://github.com/hawpetossjustin25-collab/ai-movie-studio/archive/refs/heads/main.zip)** and extract it.
2. Double-click **`start_movie_studio.bat`** (or the **AI Movie Studio** shortcut on your Desktop).

---

## 🔗 Quick Links

- **Live Phone / Web App:** [https://hawpetossjustin25-collab.github.io/ai-movie-studio/](https://hawpetossjustin25-collab.github.io/ai-movie-studio/)
- **GitHub Repository:** [https://github.com/hawpetossjustin25-collab/ai-movie-studio](https://github.com/hawpetossjustin25-collab/ai-movie-studio)
- **Direct ZIP Download:** [https://github.com/hawpetossjustin25-collab/ai-movie-studio/archive/refs/heads/main.zip](https://github.com/hawpetossjustin25-collab/ai-movie-studio/archive/refs/heads/main.zip)

---

## 🧠 REA Multi-Generator Architecture (`movie_planner.py` & `movie_engine.py`)

Synthesizes the best open-source GitHub AI movie-making architectures via **REA (`rea-agents`)**:
1. **`HKUDS/ViMax`:** Hierarchical Director + Screenwriter + Persistent Character Bank (`6 Acts × 30 min = 3 Hours`).
2. **`F-R-L/forge-film`:** DAG Parallel Scene Scheduling across CPU worker threads + single-pass `ffmpeg -f concat -safe 0 -c copy` zero-reencode stream assembly.
3. **`AEON-7/aeon-movie-maker`:** Deterministic per-character seed consistency + sidechain-ducked audio bus (`-12 dB` music ducking under spoken character dialogue).
4. **`harry0703/MoneyPrinterTurbo` & `jimmyjjz/lfs-vid-gen`:** Keyless AI keyframes (Pollinations Flux) + 100% offline procedural widescreen 2.5D Ken-Burns compositor + burned-in cinema letterbox subtitles.

---

## ✅ Run the Verification Suite
```bash
python test_movie_studio.py
```
