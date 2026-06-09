"""
fix_videos_for_web.py
---------------------
Run this ONCE to remux all existing hand_movements.mp4 files
so they are web-playable in the browser (moov atom at front).

Usage:
    python fix_videos_for_web.py

It will scan the 'results' folder and fix every *_hand_movements.mp4 it finds.
"""

import os
import subprocess
import shutil

RESULTS_FOLDER = "results"  # adjust if your results folder is elsewhere


def remux_for_web(input_path):
    tmp_path = input_path + "_faststart.mp4"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-c", "copy", "-movflags", "+faststart", tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            shutil.move(tmp_path, input_path)
            print(f"  ✅ Fixed: {os.path.basename(input_path)}")
        else:
            print(f"  ❌ Failed: {os.path.basename(input_path)}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"  ❌ Error on {os.path.basename(input_path)}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    fixed = 0
    for root, dirs, files in os.walk(RESULTS_FOLDER):
        for f in files:
            if f.endswith("_hand_movements.mp4"):
                full_path = os.path.join(root, f)
                print(f"Processing: {full_path}")
                remux_for_web(full_path)
                fixed += 1

    if fixed == 0:
        print("No *_hand_movements.mp4 files found.")
    else:
        print(f"\nDone! Fixed {fixed} file(s). Restart your Flask app and refresh the page.")
