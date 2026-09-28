#!/usr/bin/env python3
"""
Preprocessing script: probe the frame count of every video and write the cache JSON used by InsertionDatasetExpand.

Usage:
    python probe_video_lengths.py \
        --ann_path /path/to/info_folder_or_file.json \
        --output /path/to/frame_cache.json \
        --num_workers 64

Notes:
    - ann_path may be a JSON file or a directory of JSON files, in the same format as InsertionDataset
    - Incremental: if the output file already exists, only samples missing from the cache are probed
    - Output format: {"input_video_path||gt_video_path": min(src_frames, gt_frames), ...}
"""

import argparse
import glob
import json
import os
import gc
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from decord import VideoReader


def get_video_length(video_path):
    vr = VideoReader(video_path, num_threads=1)
    n = len(vr)
    del vr
    gc.collect()
    return n


def probe_one(args):
    """Probe the minimum frame count of a video pair. Returns (key, common_len, error)."""
    key, src_path, tar_path = args
    try:
        src_len = get_video_length(src_path)
        tar_len = get_video_length(tar_path)
        return key, min(src_len, tar_len), None
    except Exception as e:
        return key, -1, str(e)


def load_single_source(path):
    """Load JSON annotations, mirroring InsertionDataset._load_single_source."""
    dataset = []

    if os.path.isdir(path):
        data_root = os.path.dirname(os.path.abspath(path))
        json_files = sorted(glob.glob(os.path.join(path, '*.json')))
        print(f"Found {len(json_files)} JSON files in: {path}")
        print(f"Auto data_root: {data_root}")
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        item['_data_root'] = data_root
                    dataset.extend(data)
                else:
                    data['_data_root'] = data_root
                    dataset.append(data)
            except Exception as e:
                print(f"  [WARNING] Failed to load {jf}: {e}")
    elif path.endswith('.json'):
        data_root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    item['_data_root'] = data_root
                dataset.extend(data)
            else:
                data['_data_root'] = data_root
                dataset.append(data)
            print(f"Loaded {len(dataset)} samples from {path}")
            print(f"Auto data_root: {data_root}")
        except Exception as e:
            print(f"  [WARNING] Failed to load {path}: {e}")
    else:
        raise ValueError(f"Invalid path: {path}")

    return dataset


def resolve_path(rel_path, data_info):
    if os.path.isabs(rel_path):
        return rel_path
    item_root = data_info.get('_data_root')
    if item_root is not None:
        return os.path.join(item_root, rel_path)
    return rel_path


def main():
    parser = argparse.ArgumentParser(description="Probe video frame counts and write the cache JSON")
    parser.add_argument("--ann_path", type=str, nargs='+', required=True,
                        help="Annotation path(s): JSON file(s) or directory(ies)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of the output cache JSON file")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Number of worker threads (default 64)")
    args = parser.parse_args()

    # Load all annotations
    all_data = []
    for p in args.ann_path:
        loaded = load_single_source(p)
        all_data.extend(loaded)
    print(f"\nTotal samples loaded: {len(all_data)}")

    # Load the existing cache (incremental update)
    cache_dict = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                cache_dict = json.load(f)
            print(f"Loaded existing cache: {len(cache_dict)} entries from {args.output}")
        except Exception as e:
            print(f"Failed to load existing cache ({e}), starting fresh")

    # Find the samples that still need probing
    to_probe = []
    already_cached = 0
    for info in all_data:
        key = f"{info['input_video']}||{info['gt_video']}"
        if key in cache_dict:
            already_cached += 1
        else:
            src_path = resolve_path(info['input_video'], info)
            tar_path = resolve_path(info['gt_video'], info)
            to_probe.append((key, src_path, tar_path))

    print(f"Already cached: {already_cached}, need to probe: {len(to_probe)}")

    if not to_probe:
        print("Nothing to probe. Done!")
        return

    # Multi-threaded probing, submitted in batches to avoid creating too many futures at once
    print(f"Probing with {args.num_workers} threads ...", flush=True)
    ok_count = 0
    fail_count = 0
    batch_size = args.num_workers * 4

    pbar = tqdm(total=len(to_probe), desc="Probing", mininterval=0.5)
    for batch_start in range(0, len(to_probe), batch_size):
        batch = to_probe[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(executor.map(probe_one, batch))
        for key, common_len, err in results:
            if err is not None:
                fail_count += 1
                if fail_count <= 20:
                    tqdm.write(f"  [FAIL] {key.split('||')[0]}: {err}")
            else:
                cache_dict[key] = common_len
                ok_count += 1
            pbar.update(1)
    pbar.close()

    print(f"\nDone! Probed {ok_count} ok, {fail_count} failed")
    print(f"Total cache entries: {len(cache_dict)}")

    # Save
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    tmp_path = args.output + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(cache_dict, f, ensure_ascii=False)
    os.replace(tmp_path, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
