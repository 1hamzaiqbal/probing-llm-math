import os
import shutil

# Define the root directory and the archive directory
ROOT_DIR = "/Users/hamzaiqbal/grad/llm/probing-llm-math"
ARCHIVE_DIR = os.path.join(ROOT_DIR, "archive")

# List of files to keep (whitelist) - based on build_it.md and current needs
# We keep the main notebooks if they look useful, but the user said "duplicates or useless".
# I will be conservative and archive things that look like obvious duplicates or old versions.
FILES_TO_ARCHIVE = [
    "FIX_Load_Problems_Cell.py",
    "FIX_Run_Performance_Test_Cell.py",
    "eval_pipeline_heatmap.ipynb",
    "output (1).png",
    "output.png",
    "performance_test.ipynb",
    "performance_test_current.ipynb",
    "performance_test_current_heatmap.ipynb",
    "performance_test_filled.ipynb",
    "performance_test_filled.py",
    "performance_test_old.ipynb",
]

def cleanup():
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        print(f"Created archive directory: {ARCHIVE_DIR}")

    for filename in FILES_TO_ARCHIVE:
        src_path = os.path.join(ROOT_DIR, filename)
        dst_path = os.path.join(ARCHIVE_DIR, filename)

        if os.path.exists(src_path):
            try:
                shutil.move(src_path, dst_path)
                print(f"Moved {filename} to archive.")
            except Exception as e:
                print(f"Error moving {filename}: {e}")
        else:
            print(f"File not found (skipped): {filename}")

if __name__ == "__main__":
    cleanup()
