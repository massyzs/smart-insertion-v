"""Stage-2 insertion dataset with single-frame (image) items and a step-grouped, resumable sampler.

`InsertionDatasetFast` is a self-contained copy of `InsertionDatasetExpand` (dataset_image_video.py).
Video items are loaded exactly as there: contiguous frames of one segment, every frame stretched to the
target size with cv2 INTER_LINEAR, values normalised to [-1, 1], the T5 caption dropped with probability
`text_drop_ratio`. What is added:

  * image entries (index `type == "image"`, or an image file extension on `input_video`): decord cannot open a
    JPEG, so such an entry used to yield no item at all. Here it yields one single-frame item: the two images
    are decoded with PIL, stretched with the same cv2 call and returned with T = 1 (no padding to the clip
    length), i.e. the sample trains the first frame only;
  * entries whose `ref_img` is empty are dropped when the index is loaded (the original substituted a random
    other sample at run time);
  * a load failure is replaced by another item of the SAME kind, so a step keeps its shape;
  * worker hygiene for multi-day runs: decord readers are released right after decoding and the glibc heap
    is trimmed periodically (persistent workers otherwise keep freed decode buffers forever);
  * `StepGroupedBatchSampler`: a deterministic sampler that knows its rank. One global step is
    `num_processes * batch_size` items of one kind (all video or all image), so ranks holding a cheap
    single-frame sample never wait for ranks holding a 33-frame clip. The position is a pure function of
    (seed, epoch, step), which makes resuming exact.
"""
import ctypes
import glob
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Sampler, get_worker_info
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

# One OpenCV thread per process. The resize result does not depend on the thread count, and a process that has
# started OpenCV's thread pool (one thread per core) cannot safely fork dataloader workers afterwards: the
# workers deadlock inside cv2.resize.
cv2.setNumThreads(1)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MALLOC_TRIM_EVERY = 64   # items per worker between two malloc_trim(0) calls; 0 disables

_libc = None


def _malloc_trim():
    """Hand freed heap pages back to the OS (glibc keeps freed decode buffers in its arenas otherwise)."""
    global _libc
    try:
        if _libc is None:
            _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except Exception:  # noqa: BLE001  (non-glibc platform: nothing to trim)
        pass


def read_still_rgb(path):
    """Decode a still image to an RGB uint8 array [H, W, 3].

    Regular image files go through PIL (the decoder used for the reference images). A small part of the
    `image` subset stores its stills as a one-frame H.264 elementary stream under a .jpeg name; PIL cannot
    identify those, so they are decoded with OpenCV (FFmpeg backend) and converted to RGB.
    """
    try:
        return np.array(Image.open(path).convert('RGB'))
    except UnidentifiedImageError as pil_error:
        # not an image file at all (the H.264 stills); a damaged real image raises a different error and is
        # left to __getitem__, which replaces the sample
        cap = cv2.VideoCapture(path)
        try:
            ok, frame_bgr = cap.read()
        finally:
            cap.release()
        if not ok or frame_bgr is None:
            raise pil_error
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def insertion_worker_init_fn(worker_id):
    """DataLoader worker setup: one cv2 thread per worker (cv2 would otherwise start a pool with one thread
    per core in every worker; the resize result does not depend on the thread count). RNG seeding is left
    to PyTorch's default so the caption-drop stream behaves as before."""
    cv2.setNumThreads(1)


class InsertionDatasetFast(Dataset):
    """Expanded insertion dataset (see module docstring).

    Based on a pre-generated frame-count cache JSON (expand_cache_json), splits long videos into
    multiple contiguous segments of length video_sample_n_frames, each used as an independent training sample.
    Image entries become one single-frame item each and need no cache entry.

    Cache format: {"input_video_path||gt_video_path": frame_count, ...}

    Example: video_sample_n_frames=33, video has 81 frames
      -> segment_0: frames 0-32
      -> segment_1: frames 33-65
      -> remaining frames 66-80 are only 15 frames, fewer than 33, dropped by default
    """

    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size_h=480,
        video_sample_size_w=832,
        video_sample_stride=4,
        video_sample_n_frames=81,
        text_drop_ratio=0.1,
        brief_txt_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0,
        video_length_drop_end=1.0,
        check=False,
        check_output_json=None,
        allow_short_last=False,
        min_frames_for_short=None,
        overlap_frames=0,
        expand_cache_json=None,
    ):
        """
        Args:
            allow_short_last: Whether to keep the last segment that is shorter than video_sample_n_frames (padded with the last frame).
            min_frames_for_short: Minimum number of frames required for the last segment when allow_short_last=True.
                                  Defaults to video_sample_n_frames // 2.
            overlap_frames: Number of overlapping frames between adjacent segments, default 0.
            expand_cache_json: Path to the frame-count cache JSON file.
        """
        if isinstance(ann_path, str):
            ann_paths = [ann_path]
        elif isinstance(ann_path, list):
            ann_paths = ann_path
        else:
            raise ValueError(f"ann_path must be str or list of str, got {type(ann_path)}")

        print(f"loading annotations from {ann_path} ...")

        self.dataset = []
        loaded_data_len = 0

        for path in ann_paths:
            loaded_data = self._load_single_source(path)
            self.dataset.extend(loaded_data)
            loaded_data_len += len(loaded_data)
            print(f"  Loaded {len(loaded_data)} samples from {path}")
        print(f"  -------------------- Total loaded samples: {loaded_data_len} ----------------------")

        # Entries without a (stylized) reference image are not training samples.
        n_before = len(self.dataset)
        self.dataset = [d for d in self.dataset if str(d.get('ref_img') or '').strip()]
        if len(self.dataset) != n_before:
            print(f"  [INFO] Dropped {n_before - len(self.dataset)} entries with an empty ref_img")

        self.data_root = data_root

        if check_output_json is not None and os.path.exists(check_output_json) and check:
            print(f"  [INFO] Found cached filtered JSON at {check_output_json}, loading directly...")
            with open(check_output_json, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            print(f"  [INFO] Loaded {len(self.dataset)} samples from cached JSON")
        elif check:
            num_workers = 80
            print(f"  [INFO] Checking video file existence with {num_workers} threads...")

            def check_sample_exists(item):
                src_video = self._resolve_path(item['input_video'], item)
                tar_video = self._resolve_path(item['gt_video'], item)
                ref_img = self._resolve_path(item['ref_img'], item)
                tar_img = self._resolve_path(item['gt_ref_img'], item)
                return item, all(os.path.isfile(p) for p in [src_video, tar_video, ref_img, tar_img])

            valid_dataset = []
            missing_count = 0

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for item, exists in tqdm(executor.map(check_sample_exists, self.dataset),
                                         total=len(self.dataset), desc="Checking videos"):
                    if exists:
                        valid_dataset.append(item)
                    else:
                        missing_count += 1

            if missing_count > 0:
                print(f"  [INFO] Removed {missing_count} samples with missing videos, {len(valid_dataset)} samples remaining")
            self.dataset = valid_dataset

            if check_output_json is not None:
                output_dir = os.path.dirname(check_output_json)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                with open(check_output_json, 'w', encoding='utf-8') as f:
                    json.dump(self.dataset, f, ensure_ascii=False, indent=2)
                print(f"  [INFO] Saved filtered dataset ({len(self.dataset)} samples) to {check_output_json}")

        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.brief_txt_ratio = brief_txt_ratio
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = (video_sample_size_h, video_sample_size_w)
        self.video_sample_size_w = video_sample_size_w
        self.video_sample_size_h = video_sample_size_h

        self.allow_short_last = allow_short_last
        self.min_frames_for_short = min_frames_for_short if min_frames_for_short is not None else video_sample_n_frames // 2
        self.overlap_frames = overlap_frames
        self.expand_cache_json = expand_cache_json

        self.video_transforms = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    inplace=True,
                ),
            ]
        )
        self.image_transforms = transforms.Compose([
            transforms.Resize((video_sample_size_h, video_sample_size_w)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
        ])

        # Keep the original data list
        self._raw_dataset = self.dataset

        self._items_since_trim = 0

        # Load the frame-count cache and expand
        self._expand_dataset()

    # ------------------------------------------------------------------
    #  Expansion logic
    # ------------------------------------------------------------------

    @staticmethod
    def _is_image_entry(info):
        """An entry is an image sample when the index says so or when its `input_video` is an image file."""
        if str(info.get('type', '')).lower() == 'image':
            return True
        return str(info.get('input_video', '')).lower().endswith(IMAGE_EXTENSIONS)

    def _load_frame_cache(self):
        """Load the frame-count cache from expand_cache_json; returns {dataset_idx: common_len} for video entries."""
        if self.expand_cache_json is None or not os.path.exists(self.expand_cache_json):
            raise FileNotFoundError(
                f"Frame-length cache not found: {self.expand_cache_json}\n"
                f"Please build it first (scripts/wan2.1/probe_video_lengths.py)."
            )

        with open(self.expand_cache_json, 'r', encoding='utf-8') as f:
            cache_dict = json.load(f)
        print(f"  [Expand] Loaded frame-length cache: {len(cache_dict)} entries "
              f"from {self.expand_cache_json}")

        video_lengths = {}
        missing = 0
        n_image = 0
        for idx, info in enumerate(self._raw_dataset):
            if self._is_image_entry(info):
                n_image += 1
                continue
            key = f"{info['input_video']}||{info['gt_video']}"
            if key in cache_dict:
                video_lengths[idx] = cache_dict[key]
            else:
                missing += 1

        print(f"  [Expand] Cache matched: {len(video_lengths)}, missing: {missing}, image entries (no cache needed): {n_image}")
        return video_lengths

    def _expand_dataset(self):
        """Split each video into contiguous segments of video_sample_n_frames frames; one item per image entry."""
        target = self.video_sample_n_frames
        step = target - self.overlap_frames
        if step <= 0:
            raise ValueError(f"overlap_frames ({self.overlap_frames}) must be < video_sample_n_frames ({target})")

        video_lengths = self._load_frame_cache()

        self.expanded_items = []
        skipped = 0

        for dataset_idx in range(len(self._raw_dataset)):
            if self._is_image_entry(self._raw_dataset[dataset_idx]):
                # single-frame item: (entry, first frame, 1 frame)
                self.expanded_items.append((dataset_idx, 0, 1))
                continue

            common_len = video_lengths.get(dataset_idx, -1)
            if common_len <= 0:
                skipped += 1
                continue

            start_frame = int(common_len * self.video_length_drop_start)
            end_frame = int(common_len * self.video_length_drop_end)
            available = end_frame - start_frame

            if available <= 0:
                start_frame = 0
                end_frame = common_len
                available = common_len

            if available < self.min_frames_for_short:
                skipped += 1
                continue

            seg_start = start_frame
            while seg_start + target <= end_frame:
                self.expanded_items.append((dataset_idx, seg_start, target))
                seg_start += step

            remaining = end_frame - seg_start
            if remaining > 0:
                if remaining >= target:
                    self.expanded_items.append((dataset_idx, seg_start, target))
                elif self.allow_short_last and remaining >= self.min_frames_for_short:
                    self.expanded_items.append((dataset_idx, seg_start, remaining))

            if seg_start == start_frame and available < target:
                if available >= self.min_frames_for_short:
                    self.expanded_items.append((dataset_idx, start_frame, available))

        self.length = len(self.expanded_items)

        # Kind of every expanded item, for the sampler and for same-kind replacement after a load failure.
        self.item_is_image = np.array(
            [self._is_image_entry(self._raw_dataset[di]) for di, _, _ in self.expanded_items], dtype=bool)
        self.image_item_ids = np.nonzero(self.item_is_image)[0]
        self.video_item_ids = np.nonzero(~self.item_is_image)[0]

        print(f"  [Expand] Original samples: {len(self._raw_dataset)}, "
              f"Expanded samples: {len(self.expanded_items)} "
              f"(video segments: {len(self.video_item_ids)}, single-frame images: {len(self.image_item_ids)}), "
              f"Skipped: {skipped}")
        print(f"  [Expand] Expansion ratio: {len(self.expanded_items) / max(len(self._raw_dataset), 1):.2f}x")

        # Build the expanded dataset list so samplers can access it via .dataset[idx]
        self.dataset = [self._raw_dataset[di] for di, _, _ in self.expanded_items]

        # Cache frame counts for use in get_batch
        self._video_lengths = video_lengths

    # ------------------------------------------------------------------
    #  Utility methods
    # ------------------------------------------------------------------

    def rand_another(self, idx=None):
        """A random other item of the same kind as `idx` (any kind when idx is None)."""
        if idx is None:
            return random.randint(0, self.length - 1)
        pool = self.image_item_ids if self.item_is_image[idx % self.length] else self.video_item_ids
        new_idx = int(pool[random.randint(0, len(pool) - 1)])
        if new_idx == idx and len(pool) > 1:
            new_idx = int(pool[(int(np.searchsorted(pool, new_idx)) + 1) % len(pool)])
        return new_idx

    def _load_single_source(self, path):
        dataset = []

        if os.path.isdir(path):
            data_root_for_items = os.path.dirname(os.path.abspath(path))
            json_files = sorted(glob.glob(os.path.join(path, '*.json')))
            print(f"[InsertionDatasetFast] Found {len(json_files)} JSON files in directory: {path}")
            print(f"[InsertionDatasetFast] Auto data_root: {data_root_for_items}")
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                item['_data_root'] = data_root_for_items
                            dataset.extend(data)
                            print(f"    Loaded {len(data)} samples from {os.path.basename(json_file)}")
                        else:
                            data['_data_root'] = data_root_for_items
                            dataset.append(data)
                            print(f"    Loaded 1 sample from {os.path.basename(json_file)}")
                except json.JSONDecodeError as e:
                    print(f"    [WARNING] Invalid JSON in {json_file}: {e}")
                except Exception as e:
                    print(f"    [WARNING] Failed to load {json_file}: {e}")
        elif path.endswith('.json'):
            data_root_for_items = os.path.dirname(os.path.dirname(os.path.abspath(path)))
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            item['_data_root'] = data_root_for_items
                        dataset.extend(data)
                        print(f"  [InsertionDatasetFast] Loaded {len(data)} samples from {os.path.basename(path)}")
                    else:
                        data['_data_root'] = data_root_for_items
                        dataset.append(data)
                        print(f"  [InsertionDatasetFast] Loaded 1 sample from {os.path.basename(path)}")
                print(f"  [InsertionDatasetFast] Auto data_root: {data_root_for_items}")
            except json.JSONDecodeError as e:
                print(f"  [WARNING] Invalid JSON in {path}: {e}")
            except Exception as e:
                print(f"  [WARNING] Failed to load {path}: {e}")
        else:
            raise ValueError(f"Invalid path: {path}. Must be a JSON file or directory.")

        return dataset

    def _resolve_path(self, rel_path, data_info):
        if os.path.isabs(rel_path):
            return rel_path
        item_root = data_info.get('_data_root')
        if item_root is not None:
            return os.path.join(item_root, rel_path)
        if self.data_root is not None:
            return os.path.join(self.data_root, rel_path)
        return rel_path

    # ------------------------------------------------------------------
    #  Data loading
    # ------------------------------------------------------------------

    def get_batch(self, expanded_idx):
        dataset_idx, seg_start, seg_n_frames = self.expanded_items[expanded_idx]
        data_info = self._raw_dataset[dataset_idx]
        is_image = bool(self.item_is_image[expanded_idx])

        src_video_path = self._resolve_path(data_info['input_video'], data_info)
        tar_video_path = self._resolve_path(data_info['gt_video'], data_info)
        ref_image_path = self._resolve_path(data_info['ref_img'], data_info)
        tar_image_path = self._resolve_path(data_info['gt_ref_img'], data_info)
        instruction = data_info['prompt']
        text = data_info['description']

        target_frames = self.video_sample_n_frames

        # Load images
        tar_image = Image.open(tar_image_path).convert('RGB')
        tar_image_np = np.array(tar_image)
        tar_image_tensor = torch.from_numpy(tar_image_np.copy()).permute(2, 0, 1).contiguous().float() / 255.0
        tar_image_tensor = self.image_transforms(tar_image_tensor).unsqueeze(0)

        ref_image = Image.open(ref_image_path).convert('RGB')
        ref_image_np = np.array(ref_image)
        ref_image_np_resized = cv2.resize(
            ref_image_np,
            (self.video_sample_size_w, self.video_sample_size_h),
            interpolation=cv2.INTER_LINEAR
        )
        ref_image_raw = torch.from_numpy(ref_image_np_resized.copy()).to(torch.uint8)

        ref_image_tensor = torch.from_numpy(ref_image_np.copy()).permute(2, 0, 1).contiguous().float() / 255.0
        ref_image_tensor = self.image_transforms(ref_image_tensor).unsqueeze(0)

        if is_image:
            # Single-frame sample: the "videos" are still images. Same RGB decode as the reference images,
            # same cv2 stretch as video frames, and no padding to the clip length.
            pixel_values = [read_still_rgb(tar_video_path)]
            src_pixel_values = [read_still_rgb(src_video_path)]
        else:
            # Take frames sequentially
            batch_index = np.arange(seg_start, seg_start + seg_n_frames, dtype=int)

            # Clamp with the cached frame count to prevent out-of-range indices
            cached_len = self._video_lengths.get(dataset_idx)
            if cached_len is not None:
                batch_index = np.clip(batch_index, 0, cached_len - 1)

            # Load video frames; each reader is released as soon as its frames are decoded
            video_reader = VideoReader(tar_video_path, num_threads=2)
            pixel_values = video_reader.get_batch(batch_index).asnumpy()
            del video_reader
            video_reader = VideoReader(src_video_path, num_threads=2)
            src_pixel_values = video_reader.get_batch(batch_index).asnumpy()
            del video_reader

        # Resize
        resized_pixel_values = []
        for frame in pixel_values:
            resized_f = cv2.resize(frame, (self.video_sample_size_w, self.video_sample_size_h), interpolation=cv2.INTER_LINEAR)
            resized_pixel_values.append(resized_f)
        pixel_values = np.array(resized_pixel_values)

        resized_pixel_values = []
        for frame in src_pixel_values:
            resized_f = cv2.resize(frame, (self.video_sample_size_w, self.video_sample_size_h), interpolation=cv2.INTER_LINEAR)
            resized_pixel_values.append(resized_f)
        src_pixel_values = np.array(resized_pixel_values)

        src_first_frame_raw = torch.from_numpy(src_pixel_values[0]).to(torch.uint8)

        if not is_image:
            if len(pixel_values) < target_frames:
                pad_count = target_frames - len(pixel_values)
                last_frame = pixel_values[-1:, ...]
                pixel_values = np.concatenate(
                    [pixel_values, np.tile(last_frame, (pad_count, 1, 1, 1))], axis=0
                )
            if len(src_pixel_values) < target_frames:
                pad_count = target_frames - len(src_pixel_values)
                last_frame = src_pixel_values[-1:, ...]
                src_pixel_values = np.concatenate(
                    [src_pixel_values, np.tile(last_frame, (pad_count, 1, 1, 1))], axis=0
                )

        tar_pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
        tar_pixel_values = tar_pixel_values / 255.
        tar_pixel_values = self.video_transforms(tar_pixel_values)

        src_pixel_values = torch.from_numpy(src_pixel_values).permute(0, 3, 1, 2).contiguous()
        src_pixel_values = src_pixel_values / 255.
        src_pixel_values = self.video_transforms(src_pixel_values)

        if random.random() < self.text_drop_ratio:
            text = ''

        return tar_pixel_values, src_pixel_values, tar_image_tensor, ref_image_tensor, instruction, text, ref_image_raw, src_first_frame_raw

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        requested_idx = int(idx)
        try_count = 0
        while try_count < 20:
            sample = {}
            try:
                pixel_values, src_pixel_values, tar_image_tensor, ref_image_tensor, instruction, text, ref_image_raw, src_first_frame_raw = self.get_batch(idx)
                sample["tar_pixel_values"] = pixel_values
                sample["src_pixel_values"] = src_pixel_values
                sample["tar_image_tensor"] = tar_image_tensor
                sample["ref_image_tensor"] = ref_image_tensor
                sample["instruction"] = instruction
                sample["text"] = text
                sample["ref_image_raw"] = ref_image_raw
                sample["src_first_frame_raw"] = src_first_frame_raw
                # bookkeeping for the trainer's data-position self check and logs
                sample["index"] = requested_idx          # the index the sampler asked for
                sample["used_index"] = int(idx)          # differs only after a load failure
                sample["is_image"] = bool(self.item_is_image[int(idx)])

                if len(sample) > 0:
                    break
            except Exception as e:
                dataset_idx = self.expanded_items[idx % self.length][0]
                print(e, self._raw_dataset[dataset_idx], flush=True)
                idx = self.rand_another(requested_idx)
                try_count += 1
        if len(sample) == 0:
            raise RuntimeError(f"Failed to fetch a valid sample after 20 retries, last idx={idx}")

        if MALLOC_TRIM_EVERY > 0 and get_worker_info() is not None:
            self._items_since_trim += 1
            if self._items_since_trim >= MALLOC_TRIM_EVERY:
                self._items_since_trim = 0
                _malloc_trim()

        return sample


class StepGroupedBatchSampler(Sampler):
    """Deterministic, rank-aware batch sampler for synchronous multi-GPU training.

    An epoch is a list of *global steps*. A global step holds `num_processes * batch_size` item indices of one
    kind (all video segments, or all single-frame images when `group_by_type` is on) and rank r takes slice
    [r * batch_size, (r + 1) * batch_size). The plan of an epoch depends only on (seed, epoch): every rank
    builds the same plan, and a resumed run continues at exactly the step where the checkpoint was written.

    Every item appears once per epoch; the last step of each kind is completed by wrapping around to the
    start of that kind's permutation (at most num_processes * batch_size - 1 repeats per kind).

    The position is persistent state set with `set_position` (not consumed by iteration), because a
    DataLoader may create the sampler iterator more than once.
    """

    def __init__(self, item_is_image, batch_size, num_processes, process_index, seed, group_by_type=True):
        self.item_is_image = np.asarray(item_is_image, dtype=bool)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.process_index = int(process_index)
        self.seed = int(seed)
        self.group_by_type = bool(group_by_type)
        self.group_size = self.batch_size * self.num_processes
        if not 0 <= self.process_index < self.num_processes:
            raise ValueError(f"process_index {self.process_index} out of range for {self.num_processes} processes")

        all_ids = np.arange(len(self.item_is_image))
        if self.group_by_type:
            self.pools = [all_ids[~self.item_is_image], all_ids[self.item_is_image]]   # [video, image]
        else:
            self.pools = [all_ids]
        self.pools = [p for p in self.pools if len(p) > 0]
        if not self.pools:
            raise ValueError("empty dataset")
        self.steps_per_kind = [int(np.ceil(len(p) / self.group_size)) for p in self.pools]
        self._steps_per_epoch = int(sum(self.steps_per_kind))

        self.epoch = 0
        self.start_step = 0
        self._plan_epoch = None
        self._plan = None

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch

    def plan(self, epoch):
        """[steps_per_epoch, group_size] int64 array of item indices for `epoch` (cached for one epoch)."""
        if self._plan_epoch != epoch:
            rng = np.random.default_rng([self.seed, int(epoch)])
            groups = []
            for pool, n_steps in zip(self.pools, self.steps_per_kind):
                perm = rng.permutation(pool)
                groups.append(np.resize(perm, n_steps * self.group_size).reshape(n_steps, self.group_size))
            plan = np.concatenate(groups, axis=0)
            plan = plan[rng.permutation(len(plan))]
            self._plan, self._plan_epoch = plan, epoch
        return self._plan

    def set_position(self, epoch, start_step=0):
        if not 0 <= start_step <= self._steps_per_epoch:
            raise ValueError(f"start_step {start_step} out of range (steps per epoch {self._steps_per_epoch})")
        self.epoch = int(epoch)
        self.start_step = int(start_step)

    def rank_batch(self, epoch, step, process_index=None):
        """Item indices that rank `process_index` (default: this rank) must receive at `step` of `epoch`."""
        r = self.process_index if process_index is None else int(process_index)
        return self.plan(epoch)[step, r * self.batch_size:(r + 1) * self.batch_size].tolist()

    def step_is_image(self, epoch, step):
        return bool(self.item_is_image[self.plan(epoch)[step, 0]])

    def __iter__(self):
        epoch, start = self.epoch, self.start_step
        plan = self.plan(epoch)
        lo = self.process_index * self.batch_size
        for step in range(start, len(plan)):
            yield plan[step, lo:lo + self.batch_size].tolist()

    def __len__(self):
        return self._steps_per_epoch
