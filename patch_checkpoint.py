"""
Patches a multiagent checkpoint to be loadable by a single-agent trainer.
Fixes the filter key mismatch by resetting filters to empty.

Usage:
  python patch_checkpoint.py <checkpoint_path>
  
Output: <checkpoint_path>_patched
"""
import sys
import pickle
import os

src = sys.argv[1]
dst = src + "_patched"

with open(src, "rb") as f:
    data = pickle.load(f)

worker = pickle.loads(data["worker"])

# The filter assertion checks: all(k in new_filters for k in checkpoint_filters)
# Single-agent worker has filter key 0 only; multiagent has 0,1,2,3
# Keep only key 0 so the assertion passes
filters = worker.get("filters", {})
if filters:
    # Keep only agent 0's filter (or empty if not present)
    worker["filters"] = {0: filters.get(0, filters.get(list(filters.keys())[0]))}

data["worker"] = pickle.dumps(worker)

with open(dst, "wb") as f:
    pickle.dump(data, f)

# Copy the .tune_metadata file too
meta_src = src + ".tune_metadata"
meta_dst = dst + ".tune_metadata"
if os.path.exists(meta_src):
    import shutil
    shutil.copy(meta_src, meta_dst)

print(f"Patched checkpoint saved to: {dst}")
