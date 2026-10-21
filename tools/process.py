"""Build web media from media.json.

Runs in GitHub Actions (which can reach the Higgsfield CDN). Outputs into media/:
  film_1600.mp4 / film_1280.mp4  - all clips joined, short-GOP H.264 for smooth scroll scrubbing
  poster.jpg, end.jpg             - first and last frame of the film
  <name>.jpg                      - each still, auto-aligned to the "before" still
  manifest.json                   - clip boundaries (seconds) for the chapter timeline
  qa_sheet.jpg                    - contact sheet of the film for review
"""
import json, os, subprocess, urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "media")
TMP = "/tmp/media"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)
cfg = json.load(open(os.path.join(ROOT, "media.json")))


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


# ---------- film ----------
clips = [fetch(u, f"{TMP}/clip{i}.mp4") for i, u in enumerate(cfg["clips"])]
fps = cfg.get("fps", 24)
norm = []
for i, c in enumerate(clips):
    p = f"{TMP}/norm{i}.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-i", c, "-an", "-vf", f"fps={fps},scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1",
         "-c:v", "libx264", "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p", p])
    norm.append(p)
bounds, t = [0.0], 0.0
for p in norm:
    t += duration(p)
    bounds.append(round(t, 3))
with open(f"{TMP}/list.txt", "w") as f:
    f.writelines(f"file '{p}'\n" for p in norm)
joined = f"{TMP}/joined.mp4"
run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", f"{TMP}/list.txt", "-c", "copy", joined])

for w, crf in ((1600, 23), (1280, 25)):
    run(["ffmpeg", "-y", "-v", "error", "-i", joined, "-an", "-vf", f"scale={w}:-2:flags=lanczos",
         "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-g", "4", "-keyint_min", "4", "-sc_threshold", "0",
         "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", os.path.join(OUT, f"film_{w}.mp4")])

run(["ffmpeg", "-y", "-v", "error", "-i", joined, "-frames:v", "1", "-vf", "scale=1600:-2", "-q:v", "3", os.path.join(OUT, "poster.jpg")])
run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.1", "-i", joined, "-frames:v", "1", "-vf", "scale=1600:-2", "-q:v", "3", os.path.join(OUT, "end.jpg")])
run(["ffmpeg", "-y", "-v", "error", "-i", joined, "-vf", f"fps={max(1, round(16 / bounds[-1] * 1.0, 3))},scale=400:-2,tile=4x4", "-frames:v", "1", os.path.join(OUT, "qa_sheet.jpg")])

# ---------- stills, aligned to "before" ----------
stills = cfg.get("stills", {})
base_name = cfg.get("align_to", "before")
imgs = {}
for name, url in stills.items():
    raw = fetch(url, f"{TMP}/{name}")
    im = cv2.imread(raw, cv2.IMREAD_COLOR)
    imgs[name] = cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_AREA)

report = {}
if base_name in imgs:
    ref = cv2.cvtColor(imgs[base_name], cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = np.zeros(ref.shape, np.uint8)
    mask[: int(ref.shape[0] * 0.62), :] = 255  # house, roof, sky: the parts that never change
    for name, im in imgs.items():
        if name == base_name:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        warp = np.eye(3, dtype=np.float32)
        try:
            # coarse-to-fine: align at half size, then full
            small = lambda a: cv2.resize(a, None, fx=0.5, fy=0.5)
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
            _, w2 = cv2.findTransformECC(small(ref), small(g), np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY, crit, small(mask), 5)
            s = np.diag([2, 2, 1]).astype(np.float32)
            warp = s @ w2 @ np.linalg.inv(s)
            _, warp = cv2.findTransformECC(ref, g, warp.astype(np.float32), cv2.MOTION_HOMOGRAPHY, crit, mask, 5)
            imgs[name] = cv2.warpPerspective(im, warp, (1920, 1080), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                             borderMode=cv2.BORDER_REPLICATE)
            report[name] = "aligned"
        except cv2.error as e:
            report[name] = f"not aligned: {e}"
for name, im in imgs.items():
    cv2.imwrite(os.path.join(OUT, f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 86, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])

json.dump({"duration": bounds[-1], "bounds": bounds, "stills": report}, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
print(json.dumps(report, indent=2))
