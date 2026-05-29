from pmlb import classification_dataset_names, regression_dataset_names, fetch_data
import os

cache_dir = "./.pmlb_cache"
os.makedirs(cache_dir, exist_ok=True)

# 下载 regression benchmark（symbolic regression 用）
for ds in regression_dataset_names:
    try:
        print("Downloading:", ds)
        fetch_data(ds, return_X_y=False, local_cache_dir=cache_dir)
    except Exception as e:
        print("FAILED:", ds, e)