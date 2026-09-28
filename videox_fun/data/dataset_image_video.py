import csv
import gc
import io
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from random import shuffle
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset
import glob
from tqdm import tqdm
from .utils import (VIDEO_READER_TIMEOUT, Camera, VideoReader_contextmanager,
                    custom_meshgrid, get_random_mask, get_relative_pose,
                    get_video_reader_batch, padding_image, process_pose_file,
                    process_pose_params, ray_condition, resize_frame,
                    resize_image_with_target_area)


class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        # self.bucket = {'image':[], 'video':[]}
        self.bucket = {'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'video')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            # elif len(self.bucket['image']) == self.batch_size:
            #     bucket = self.bucket['image']
            #     yield bucket[:]
            #     del bucket[:]


class ImageVideoDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=False,
        return_file_name=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('type', 'image')=='video':
            # video_id, text = data_info['file_path'], data_info['text']
            try:
                video_id, text = data_info['file_path'], data_info['dense_text']
            except:
                video_id, text = data_info['file_path'], data_info['text']

            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
            return pixel_values, text, 'video', video_dir
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image', image_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class InsertionDataset(Dataset):
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
    ):
        if isinstance(ann_path, str):
            ann_paths = [ann_path]
        elif isinstance(ann_path, list):
            ann_paths = ann_path
        else:
            raise ValueError(f"ann_path must be str or list of str, got {type(ann_path)}")
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")

        # Load all JSON files from multiple sources
        self.dataset = []
        loaded_data_len=0
        
        for path in ann_paths:
            loaded_data = self._load_single_source(path)
            self.dataset.extend(loaded_data)
            loaded_data_len += len(loaded_data)
            print(f"  Loaded {len(loaded_data)} samples from {path}")
        print(f"  -------------------- Total loaded samples: {loaded_data_len} ----------------------")

        self.data_root = data_root
        
        # If check_output_json already exists, load it directly and skip the check
        if check_output_json is not None and os.path.exists(check_output_json) and check:
            print(f"  [INFO] Found cached filtered JSON at {check_output_json}, loading directly...")
            with open(check_output_json, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            print(f"  [INFO] Loaded {len(self.dataset)} samples from cached JSON")
        elif check:
            # Only check video file existence when check=True and no cache exists
            num_workers=80
            print(f"  [INFO] Checking video file existence with {num_workers} threads...")
            
            def check_sample_exists(item):
                src_video = self._resolve_path(item['input_video'], item)
                tar_video = self._resolve_path(item['gt_video'], item)
                ref_img = self._resolve_path(item['ref_img'], item)
                tar_img = self._resolve_path(item['gt_ref_img'], item)
                return item, all(os.path.exists(p) for p in [src_video, tar_video, ref_img, tar_img])
            
            valid_dataset = []
            missing_count = 0
            
            # Use map for streaming processing to avoid creating all futures at once and exhausting memory
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
            
            # If an output path is specified, save the filtered merged JSON
            if check_output_json is not None:
                output_dir = os.path.dirname(check_output_json)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                with open(check_output_json, 'w', encoding='utf-8') as f:
                    json.dump(self.dataset, f, ensure_ascii=False, indent=2)
                print(f"  [INFO] Saved filtered dataset ({len(self.dataset)} samples) to {check_output_json}")
        
        self.length = len(self.dataset)
        print(f"Data scale: {self.length}")
        


        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.brief_txt_ratio = brief_txt_ratio
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        
        # Video params
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = (video_sample_size_h, video_sample_size_w)
        self.video_sample_size_w=video_sample_size_w
        self.video_sample_size_h=video_sample_size_h


        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        self.video_transforms = transforms.Compose(
            [
                # transforms.Resize((video_sample_size_h, video_sample_size_w)),  # Force resize to the specified height/width
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    inplace=True,
                ),
            ]
        )
        self.image_transforms   = transforms.Compose([
            transforms.Resize((video_sample_size_h, video_sample_size_w)),  # Force resize to the specified height/width
            # transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5],inplace=True,),
            
        ])

        # self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

        
    def rand_another(self, idx=None):
        """Return a random index to replace a failed sample."""
        new_idx = random.randint(0, self.length - 1)
        if idx is not None and new_idx == idx and self.length > 1:
            new_idx = (new_idx + 1) % self.length
        return new_idx
    
    def _load_single_source(self, path):
        """Load JSON data from a single file or directory.
        
        Handles both JSON formats uniformly:
        1. Single sample: {}
        2. Multiple samples: [{}, {}, ...]
        
        Path inference rules:
        - If path is a directory (e.g. xxx/A/info), then _data_root = xxx/A/ (one level up)
        - If path is a JSON file (e.g. xxx/A/info/x.json), then _data_root = xxx/A/ (two levels up)
        Relative paths inside the JSON are all resolved against _data_root.
        
        Args:
            path: Can be:
                - A single JSON file: "/path/to/A/info/data.json"
                - A directory containing JSON files: "/path/to/A/info/"
        
        Returns:
            List of data items loaded from the source, each with '_data_root' injected
        """
        dataset = []
        
        if os.path.isdir(path):
            data_root_for_items = os.path.dirname(os.path.abspath(path))
            json_files = sorted(glob.glob(os.path.join(path, '*.json')))
            print(f"[InsertionDataset] Found {len(json_files)} JSON files in directory: {path}")
            print(f"[InsertionDataset] Auto data_root: {data_root_for_items}")
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
                        print(f"  [InsertionDataset] Loaded {len(data)} samples from {os.path.basename(path)}")
                    else:
                        data['_data_root'] = data_root_for_items
                        dataset.append(data)
                        print(f"  [InsertionDataset] Loaded 1 sample from {os.path.basename(path)}")
                print(f"  [InsertionDataset] Auto data_root: {data_root_for_items}")
            except json.JSONDecodeError as e:
                print(f"  [WARNING] Invalid JSON in {path}: {e}")
            except Exception as e:
                print(f"  [WARNING] Failed to load {path}: {e}")
        else:
            raise ValueError(f"Invalid path: {path}. Must be a JSON file or directory.")
        
        return dataset
    
    def _get_video_length(self, video_path):
        """Get the number of frames in a video without loading all frames."""
        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            return len(video_reader)
        
    def _resolve_path(self, rel_path, data_info):
        """Resolve a relative path using per-item _data_root, fallback to self.data_root."""
        if os.path.isabs(rel_path):
            return rel_path
        item_root = data_info.get('_data_root')
        if item_root is not None:
            return os.path.join(item_root, rel_path)
        if self.data_root is not None:
            return os.path.join(self.data_root, rel_path)
        return rel_path

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        

        # Get video paths and text from JSON, resolving relative paths with the per-item _data_root
        src_video_path = self._resolve_path(data_info['input_video'], data_info)
        tar_video_path = self._resolve_path(data_info['gt_video'], data_info)
        ref_image_path = self._resolve_path(data_info['ref_img'], data_info)
        tar_image_path = self._resolve_path(data_info['gt_ref_img'], data_info)
        instruction = data_info['prompt']
        text = data_info['description']



        # # 1) load ref image
        # ref_image = Image.open(ref_image_path).convert("RGB")
        # ref_image_np = np.array(ref_image)  # HWC, uint8
        
        # # A) Truly unprocessed raw (original resolution)
        # ref_image_raw = torch.from_numpy(ref_image_np.copy()).to(torch.uint8)
        
        # # 2) Training branch (post-processed the same way as the video)
        # ref_image_np_resized = cv2.resize(
        #     ref_image_np, 
        #     (self.video_sample_size_w, self.video_sample_size_h), 
        #     interpolation=cv2.INTER_LINEAR
        # )

        tar_image = Image.open(tar_image_path).convert('RGB')
        tar_image_np = np.array(tar_image)  # HWC, uint8
        tar_image_tensor = torch.from_numpy(tar_image_np.copy()).permute(2, 0, 1).contiguous().float() / 255.0
        # tar_image_tensor = tar_image_tensor / 255.
        tar_image_tensor = self.image_transforms(tar_image_tensor).unsqueeze(0)

        ref_image = Image.open(ref_image_path).convert('RGB')
        ref_image_np = np.array(ref_image)  # HWC, uint8
        ref_image_np_resized = cv2.resize(
            ref_image_np, 
            (self.video_sample_size_w, self.video_sample_size_h), 
            interpolation=cv2.INTER_LINEAR
        )
        ref_image_raw = torch.from_numpy(ref_image_np_resized.copy()).to(torch.uint8)

        ref_image_tensor = torch.from_numpy(ref_image_np.copy()).permute(2, 0, 1).contiguous().float() / 255.0


        ref_image_tensor = self.image_transforms(ref_image_tensor).unsqueeze(0)


        tar_video_length = self._get_video_length(tar_video_path)
        src_video_length = self._get_video_length(src_video_path)
        common_video_length = min(tar_video_length, src_video_length)

        target_frames = self.video_sample_n_frames

        start_frame = int(common_video_length * self.video_length_drop_start)
        end_frame = int(common_video_length * self.video_length_drop_end)
        available_frames = end_frame - start_frame

        if available_frames <= 0:
            start_frame = 0
            end_frame = common_video_length
            available_frames = common_video_length

        if available_frames >= target_frames * self.video_sample_stride:
            max_start = end_frame - (target_frames - 1) * self.video_sample_stride
            start_idx = random.randint(start_frame, max_start)
            batch_index = np.arange(
                start_idx,
                start_idx + target_frames * self.video_sample_stride,
                self.video_sample_stride,
                dtype=int,
            )
        elif available_frames >= target_frames:
            batch_index = np.linspace(start_frame, end_frame - 1, target_frames, dtype=int)
        else:
            batch_index = np.arange(start_frame, end_frame, dtype=int)

            pad_length = target_frames - len(batch_index)
            if len(batch_index) > 0 and pad_length > 0:
                last_frame_idx = batch_index[-1]
                pad_indices = np.full(pad_length, last_frame_idx, dtype=int)
                batch_index = np.concatenate([batch_index, pad_indices], axis=0)
        
        # Load video frames
        with VideoReader_contextmanager(tar_video_path, num_threads=2) as video_reader:
            pixel_values = get_video_reader_batch(video_reader,batch_index)
        with VideoReader_contextmanager(src_video_path, num_threads=2) as video_reader:
            src_pixel_values = get_video_reader_batch(video_reader,batch_index)
        # pixel_values = self._load_video_frames(video_path, batch_index)
        # video_sample_size_h
        # video_sample_size_w

        resized_pixel_values = []
        
        for frame in pixel_values:
            # Use cv2 to quickly resize the raw numpy frame, keeping (H, W, C)
            resized_f = cv2.resize(frame, (self.video_sample_size_w, self.video_sample_size_h), interpolation=cv2.INTER_LINEAR)
            resized_pixel_values.append(resized_f)
        pixel_values = np.array(resized_pixel_values)

        resized_pixel_values = []
        for frame in src_pixel_values:
            resized_f = cv2.resize(frame, (self.video_sample_size_w, self.video_sample_size_h), interpolation=cv2.INTER_LINEAR)
            resized_pixel_values.append(resized_f)
        src_pixel_values = np.array(resized_pixel_values)

        
        src_first_frame_raw = torch.from_numpy(src_pixel_values[0]).to(torch.uint8)

        # If fewer than target_frames, pad with the last frame
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
        
        
        # Convert to tensor and normalize
        tar_pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
        tar_pixel_values = tar_pixel_values / 255.
        tar_pixel_values = self.video_transforms(tar_pixel_values)

        src_pixel_values = torch.from_numpy(src_pixel_values).permute(0, 3, 1, 2).contiguous()
        src_pixel_values = src_pixel_values / 255.
        src_pixel_values = self.video_transforms(src_pixel_values)
        
        # Random text drop for classifier-free guidance
        if random.random() < self.text_drop_ratio:
            text = ''
        
        return tar_pixel_values, src_pixel_values, tar_image_tensor,ref_image_tensor, instruction, text, ref_image_raw,src_first_frame_raw


    def __len__(self):
        return self.length

    def __getitem__(self, idx):


        try_count=0
        while try_count<20:
            sample = {}
            try:

                data_info_local = self.dataset[idx % len(self.dataset)]


                pixel_values, src_pixel_values, tar_image_tensor,ref_image_tensor, instruction, text, ref_image_raw,src_first_frame_raw = self.get_batch(idx)
                sample["tar_pixel_values"] = pixel_values
                sample["src_pixel_values"] = src_pixel_values
                sample["tar_image_tensor"] = tar_image_tensor
                sample["ref_image_tensor"] = ref_image_tensor
                sample["instruction"] = instruction
                sample["text"] = text
                sample["ref_image_raw"] = ref_image_raw
                sample["src_first_frame_raw"] = src_first_frame_raw
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)
                try_count+=1
        if len(sample) == 0:
            raise RuntimeError(f"Failed to fetch a valid sample after 20 retries, last idx={idx}")

        return sample

class PretrainVideoDataset(Dataset):
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
    ):
        if isinstance(ann_path, str):
            ann_paths = [ann_path]
        elif isinstance(ann_path, list):
            ann_paths = ann_path
        else:
            raise ValueError(f"ann_path must be str or list of str, got {type(ann_path)}")
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")

        # Load all JSON files from multiple sources
        self.dataset = []
        loaded_data_len=0
        
        for path in ann_paths:
            loaded_data = self._load_single_source(path)
            self.dataset.extend(loaded_data)
            loaded_data_len += len(loaded_data)
            print(f"  Loaded {len(loaded_data)} samples from {path}")
        print(f"  -------------------- Total loaded samples: {loaded_data_len} ----------------------")
        # Ensure every item has a 'type' field
        for item in self.dataset:
            if 'type' not in item:
                item['type'] = 'video'

        self.data_root = data_root
        
        # If check_output_json already exists, load it directly and skip the check
        if check_output_json is not None and os.path.exists(check_output_json) and check:
            print(f"  [INFO] Found cached filtered JSON at {check_output_json}, loading directly...")
            with open(check_output_json, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            print(f"  [INFO] Loaded {len(self.dataset)} samples from cached JSON")
        elif check:
            # Only check video file existence when check=True and no cache exists
            num_workers=80
            print(f"  [INFO] Checking video file existence with {num_workers} threads...")
            
            def check_video_exists(item):
                """Check whether a single video exists; returns (item, exists)"""
                video_path = item.get('file_path', '')
                if data_root is not None and not os.path.isabs(video_path):
                    video_path = os.path.join(data_root, video_path)
                return (item, os.path.exists(video_path))
            
            valid_dataset = []
            missing_count = 0
            
            # Use map for streaming processing to avoid creating all futures at once and exhausting memory
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for item, exists in tqdm(executor.map(check_video_exists, self.dataset), 
                                         total=len(self.dataset), desc="Checking videos"):
                    if exists:
                        valid_dataset.append(item)
                    else:
                        missing_count += 1
            
            if missing_count > 0:
                print(f"  [INFO] Removed {missing_count} samples with missing videos, {len(valid_dataset)} samples remaining")
            self.dataset = valid_dataset
            
            # If an output path is specified, save the filtered merged JSON
            if check_output_json is not None:
                output_dir = os.path.dirname(check_output_json)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                with open(check_output_json, 'w', encoding='utf-8') as f:
                    json.dump(self.dataset, f, ensure_ascii=False, indent=2)
                print(f"  [INFO] Saved filtered dataset ({len(self.dataset)} samples) to {check_output_json}")
        
        self.length = len(self.dataset)
        print(f"Data scale: {self.length}")
        

        # if ann_path.endswith('.csv'):
        #     with open(ann_path, 'r') as csvfile:
        #         dataset = list(csv.DictReader(csvfile))
        # elif ann_path.endswith('.json'):
        #     dataset = json.load(open(ann_path))
    





        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.brief_txt_ratio = brief_txt_ratio
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        
        # Video params
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = (video_sample_size_h, video_sample_size_w)
        self.video_sample_size_w=video_sample_size_w
        self.video_sample_size_h=video_sample_size_h


        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize((video_sample_size_h, video_sample_size_w)),  # Force resize to the specified height/width
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    inplace=True,
                ),
            ]
        )


        # self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

        
    def rand_another(self, idx=None):
        """Return a random index to replace a failed sample."""
        new_idx = random.randint(0, self.length - 1)
        if idx is not None and new_idx == idx and self.length > 1:
            new_idx = (new_idx + 1) % self.length
        return new_idx
    
    def _load_single_source(self, path):
        """Load JSON data from a single file or directory.
        
        Handles both JSON formats uniformly:
        1. Single sample: {}
        2. Multiple samples: [{}, {}, ...]
        
        Args:
            path: Can be:
                - A single JSON file: "/path/to/data.json"
                - A directory containing JSON files: "/path/to/json_dir/"
        
        Returns:
            List of data items loaded from the source
        """
        dataset = []
        
        if os.path.isdir(path):
            # Load all JSON files from directory
            json_files = sorted(glob.glob(os.path.join(path, '*.json')))
            print(f"[PretrainDataset] Found {len(json_files)} JSON files in directory: {path}")
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        # Unified handling: extend if it is a list, append if it is a single object
                        if isinstance(data, list):
                            dataset.extend(data)
                            print(f"    Loaded {len(data)} samples from {os.path.basename(json_file)}")
                        else:
                            dataset.append(data)
                            print(f"    Loaded 1 sample from {os.path.basename(json_file)}")
                except json.JSONDecodeError as e:
                    print(f"    [WARNING] Invalid JSON in {json_file}: {e}")
                except Exception as e:
                    print(f"    [WARNING] Failed to load {json_file}: {e}")
        elif path.endswith('.json'):
            # Load single JSON file
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Unified handling: extend if it is a list, append if it is a single object
                    if isinstance(data, list):
                        dataset.extend(data)
                        print(f"  [PretrainDataset] Loaded {len(data)} samples from {os.path.basename(path)}")
                    else:
                        dataset.append(data)
                        print(f"  [PretrainDataset] Loaded 1 sample from {os.path.basename(path)}")
            except json.JSONDecodeError as e:
                print(f"  [WARNING] Invalid JSON in {path}: {e}")
            except Exception as e:
                print(f"  [WARNING] Failed to load {path}: {e}")
        else:
            raise ValueError(f"Invalid path: {path}. Must be a JSON file or directory.")
        
        return dataset
    
    def _get_video_length(self, video_path):
        """Get the number of frames in a video without loading all frames."""
        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            return len(video_reader)
        
    def _get_video_path(self, path):
        """Get full video path, considering data_root."""
        if self.data_root is not None and not os.path.isabs(path):
            return os.path.join(self.data_root, path)
        return path
    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        

        # Get video path and text from JSON
        video_path = self._get_video_path(data_info['file_path'])
        if random.random() < self.brief_txt_ratio:
            text = data_info.get('text', '')
        else:
            text = data_info.get('dense_text', 'text')
            
        
        
        # Get the number of video frames
        video_length = self._get_video_length(video_path)
        
        if video_length == 0:
            raise ValueError(f"Video has 0 frames: {video_path}")
        
        # Compute the sampled frame indices
        target_frames = self.video_sample_n_frames
        
        # Crop the video range according to video_length_drop_start/end
        start_frame = int(video_length * self.video_length_drop_start)
        end_frame = int(video_length * self.video_length_drop_end)
        available_frames = end_frame - start_frame
        
        if available_frames <= 0:
            available_frames = video_length
            start_frame = 0
        
        # Sample with stride; if there are not enough frames, use all frames
        if available_frames >= target_frames * self.video_sample_stride:
            # Enough frames: sample with stride
            batch_index = np.arange(start_frame, start_frame + target_frames * self.video_sample_stride, self.video_sample_stride)
        elif available_frames >= target_frames:
            # Enough frames but not enough for the stride: sample uniformly
            batch_index = np.linspace(start_frame, end_frame - 1, target_frames, dtype=int)
        else:
            # Not enough frames: take all frames and pad
            # batch_index = np.arange(start_frame, end_frame, dtype=int)
            # Not enough frames: take all frames and pad with the last frame
            batch_index = np.arange(start_frame, end_frame, dtype=int)

            # 1. Compute how many frames still need to be padded
            pad_length = target_frames - len(batch_index)

            # 2. Make sure the video is not completely empty (to avoid errors just in case)
            if len(batch_index) > 0 and pad_length > 0:
                # Get the index of the last frame
                last_frame_idx = batch_index[-1]

                # Build an array filled with the last frame index, of length pad_length
                pad_indices = np.full(pad_length, last_frame_idx, dtype=int)

                # Concatenate the original indices with the padding indices
                batch_index = np.concatenate((batch_index, pad_indices))
        
        # Load video frames
        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            pixel_values = get_video_reader_batch(video_reader,batch_index)
        # pixel_values = self._load_video_frames(video_path, batch_index)
        # video_sample_size_h
        # video_sample_size_w
        resized_pixel_values = []
        for frame in pixel_values:
            # Use cv2 to quickly resize the raw numpy frame, keeping (H, W, C)
            resized_f = cv2.resize(frame, (self.video_sample_size_w, self.video_sample_size_h), interpolation=cv2.INTER_LINEAR)
            resized_pixel_values.append(resized_f)
        pixel_values = np.array(resized_pixel_values)

        first_frame_raw = torch.from_numpy(pixel_values[0]).to(torch.uint8)
        raw_video_tensor = torch.from_numpy(pixel_values).to(torch.uint8)
        
        # If fewer than target_frames, pad with the last frame
        if len(pixel_values) < target_frames:
            pad_count = target_frames - len(pixel_values)
            last_frame = pixel_values[-1:, ...]
            pixel_values = np.concatenate(
                [pixel_values, np.tile(last_frame, (pad_count, 1, 1, 1))], axis=0
            )
        
        
        # Convert to tensor and normalize
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        pixel_values = self.video_transforms(pixel_values)
        
        # Random text drop for classifier-free guidance
        if random.random() < self.text_drop_ratio:
            text = ''
        
        return pixel_values, text, 'video', video_path, first_frame_raw, raw_video_tensor   


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'video')
        try_count=0
        while try_count<20:
            sample = {}
            try:

                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'video')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, text, data_type, file_path, first_frame_raw, raw_video_tensor = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = text
                sample["data_type"] = data_type
                sample["idx"] = idx
                # if self.return_file_name:
                sample["file_name"] = os.path.basename(file_path)
                sample["first_frame"] = first_frame_raw
                sample["raw_video_tensor"] = raw_video_tensor
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)
                try_count+=1

        # if self.enable_inpaint and not self.enable_bucket:
        #     mask = get_random_mask(pixel_values.size())
        #     mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
        #     sample["mask_pixel_values"] = mask_pixel_values
        #     sample["mask"] = mask

        #     clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
        #     clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
        #     sample["clip_pixel_values"] = clip_pixel_values

        return sample


class ImageVideoControlDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.1, 
        video_length_drop_end=0.9,
        enable_inpaint=False,
        enable_camera_info=False,
        return_file_name=False,
        enable_subject_info=False,
        padding_subject_info=True,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.enable_camera_info = enable_camera_info
        self.enable_subject_info = enable_subject_info
        self.padding_subject_info = padding_subject_info

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        if self.enable_camera_info:
            self.video_transforms_camera = transforms.Compose(
                [
                    transforms.Resize(min(self.video_sample_size)),
                    transforms.CenterCrop(self.video_sample_size)
                ]
            )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            # transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
    
    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        video_id, text = data_info['file_path'], data_info['text']

        if data_info.get('type', 'image')=='video':
            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''

            control_video_id = data_info['control_file_path']
            
            if control_video_id is not None:
                if self.data_root is None:
                    control_video_id = control_video_id
                else:
                    control_video_id = os.path.join(self.data_root, control_video_id)
                
            if self.enable_camera_info:
                if control_video_id.lower().endswith('.txt'):
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0])
                        control_camera_values = torch.from_numpy(control_camera_values).permute(0, 3, 1, 2).contiguous()
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)
                        control_camera_values = self.video_transforms_camera(control_camera_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0], return_poses=True)
                        control_camera_values = torch.from_numpy(np.array(control_camera_values)).unsqueeze(0).unsqueeze(0)
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)[0][0]
                        control_camera_values = np.array([control_camera_values[index] for index in batch_index])
                else:
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                        control_camera_values = None
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)
                        control_camera_values = None
            else:
                if control_video_id is not None:
                    with VideoReader_contextmanager(control_video_id, num_threads=2) as control_video_reader:
                        try:
                            sample_args = (control_video_reader, batch_index)
                            control_pixel_values = func_timeout(
                                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                            )
                            resized_frames = []
                            for i in range(len(control_pixel_values)):
                                frame = control_pixel_values[i]
                                resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                                resized_frames.append(resized_frame)
                            control_pixel_values = np.array(resized_frames)
                        except FunctionTimedOut:
                            raise ValueError(f"Read {idx} timeout.")
                        except Exception as e:
                            raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                        if not self.enable_bucket:
                            control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                            control_pixel_values = control_pixel_values / 255.
                            del control_video_reader
                        else:
                            control_pixel_values = control_pixel_values

                        if not self.enable_bucket:
                            control_pixel_values = self.video_transforms(control_pixel_values)
                else:
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)
                control_camera_values = None
            
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = pixel_values.shape[-2:]
                else:
                    visual_height, visual_width = pixel_values.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_image = Image.open(subject_id[i])
                    width, height = subject_image.size
                    total_pixels = width * height

                    if self.padding_subject_info:
                        img = padding_image(subject_image, visual_width, visual_height)
                    else:
                        img = resize_image_with_target_area(subject_image, 1024 * 1024)

                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(np.array(img))
                if self.padding_subject_info:
                    subject_image = np.array(subject_images)
                else:
                    subject_image = subject_images
            else:
                subject_image = None

            return pixel_values, control_pixel_values, subject_image, control_camera_values, text, "video"
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)

            if random.random() < self.text_drop_ratio:
                text = ''

            control_image_id = data_info['control_file_path']

            if self.data_root is None:
                control_image_id = control_image_id
            else:
                control_image_id = os.path.join(self.data_root, control_image_id)

            control_image = Image.open(control_image_id).convert('RGB')
            if not self.enable_bucket:
                control_image = self.image_transforms(control_image).unsqueeze(0)
            else:
                control_image = np.expand_dims(np.array(control_image), 0)
            
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = image.shape[-2:]
                else:
                    visual_height, visual_width = image.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_image = Image.open(subject_id[i]).convert('RGB')
                    width, height = subject_image.size
                    total_pixels = width * height

                    if self.padding_subject_info:
                        img = padding_image(subject_image, visual_width, visual_height)
                    else:
                        img = resize_image_with_target_area(subject_image, 1024 * 1024)

                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(np.array(img))
                if self.padding_subject_info:
                    subject_image = np.array(subject_images)
                else:
                    subject_image = subject_images
            else:
                subject_image = None

            return image, control_image, subject_image, None, text, 'image'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, control_pixel_values, subject_image, control_camera_values, name, data_type = self.get_batch(idx)

                sample["pixel_values"] = pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["subject_image"] = subject_image
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx

                if self.enable_camera_info:
                    sample["control_camera_values"] = control_camera_values

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class ImageVideoSafetensorsDataset(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))

        self.data_root = data_root
        self.dataset = dataset
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.data_root is None:
            path = self.dataset[idx]["file_path"]
        else:
            path = os.path.join(self.data_root, self.dataset[idx]["file_path"])
        state_dict = load_file(path)
        return state_dict


class TextDataset(Dataset):
    def __init__(self, ann_path, text_drop_ratio=0.0):
        print(f"loading annotations from {ann_path} ...")
        with open(ann_path, 'r') as f:
            self.dataset = json.load(f)
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        self.text_drop_ratio = text_drop_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                item = self.dataset[idx]
                text = item['text']

                # Randomly drop text (for classifier-free guidance)
                if random.random() < self.text_drop_ratio:
                    text = ''

                sample = {
                    "text": text,
                    "idx": idx
                }
                return sample

            except Exception as e:
                print(f"Error at index {idx}: {e}, retrying with random index...")
                idx = np.random.randint(0, self.length - 1)

class InsertionDatasetExpand(Dataset):
    """Expanded version of InsertionDataset.
    
    Based on a pre-generated frame-count cache JSON (expand_cache_json), splits long videos into
    multiple contiguous segments of length video_sample_n_frames, each used as an independent training sample.
    
    The cache JSON must be generated in advance with scripts/wan2.1/probe_video_lengths.py.
    Format: {"input_video_path||gt_video_path": frame_count, ...}
    
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
            expand_cache_json: Path to the frame-count cache JSON file (must be generated in advance with scripts/wan2.1/probe_video_lengths.py).
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
                return item, all(os.path.exists(p) for p in [src_video, tar_video, ref_img, tar_img])

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

        # Load the frame-count cache and expand
        self._expand_dataset()

    # ------------------------------------------------------------------
    #  Expansion logic
    # ------------------------------------------------------------------

    def _load_frame_cache(self):
        """Load the frame-count cache from expand_cache_json; returns {dataset_idx: common_len}."""
        if self.expand_cache_json is None or not os.path.exists(self.expand_cache_json):
            raise FileNotFoundError(
                f"Frame-length cache not found: {self.expand_cache_json}\n"
                f"Please run scripts/wan2.1/probe_video_lengths.py first to generate it."
            )

        with open(self.expand_cache_json, 'r', encoding='utf-8') as f:
            cache_dict = json.load(f)
        print(f"  [Expand] Loaded frame-length cache: {len(cache_dict)} entries "
              f"from {self.expand_cache_json}")

        video_lengths = {}
        missing = 0
        for idx, info in enumerate(self._raw_dataset):
            key = f"{info['input_video']}||{info['gt_video']}"
            if key in cache_dict:
                video_lengths[idx] = cache_dict[key]
            else:
                missing += 1

        print(f"  [Expand] Cache matched: {len(video_lengths)}, missing: {missing}")
        return video_lengths

    def _expand_dataset(self):
        """Split each video into multiple contiguous segments of video_sample_n_frames frames."""
        target = self.video_sample_n_frames
        step = target - self.overlap_frames
        if step <= 0:
            raise ValueError(f"overlap_frames ({self.overlap_frames}) must be < video_sample_n_frames ({target})")

        video_lengths = self._load_frame_cache()

        self.expanded_items = []
        skipped = 0

        for dataset_idx in range(len(self._raw_dataset)):
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

        print(f"  [Expand] Original samples: {len(self._raw_dataset)}, "
              f"Expanded samples: {len(self.expanded_items)}, "
              f"Skipped: {skipped}")
        print(f"  [Expand] Expansion ratio: {len(self.expanded_items) / max(len(self._raw_dataset), 1):.2f}x")

        self.length = len(self.expanded_items)

        # Build the expanded dataset list so ImageVideoSampler can access it via .dataset[idx]
        self.dataset = [self._raw_dataset[di] for di, _, _ in self.expanded_items]

        # Cache frame counts for use in get_batch
        self._video_lengths = video_lengths

    # ------------------------------------------------------------------
    #  Utility methods
    # ------------------------------------------------------------------

    def rand_another(self, idx=None):
        new_idx = random.randint(0, self.length - 1)
        if idx is not None and new_idx == idx and self.length > 1:
            new_idx = (new_idx + 1) % self.length
        return new_idx

    def _load_single_source(self, path):
        dataset = []

        if os.path.isdir(path):
            data_root_for_items = os.path.dirname(os.path.abspath(path))
            json_files = sorted(glob.glob(os.path.join(path, '*.json')))
            print(f"[InsertionDatasetExpand] Found {len(json_files)} JSON files in directory: {path}")
            print(f"[InsertionDatasetExpand] Auto data_root: {data_root_for_items}")
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
                        print(f"  [InsertionDatasetExpand] Loaded {len(data)} samples from {os.path.basename(path)}")
                    else:
                        data['_data_root'] = data_root_for_items
                        dataset.append(data)
                        print(f"  [InsertionDatasetExpand] Loaded 1 sample from {os.path.basename(path)}")
                print(f"  [InsertionDatasetExpand] Auto data_root: {data_root_for_items}")
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

        src_video_path = self._resolve_path(data_info['input_video'], data_info)
        tar_video_path = self._resolve_path(data_info['gt_video'], data_info)
        ref_image_path = self._resolve_path(data_info['ref_img'], data_info)
        tar_image_path = self._resolve_path(data_info['gt_ref_img'], data_info)
        instruction = data_info['prompt']
        text = data_info['description']

        target_frames = self.video_sample_n_frames

        # Take frames sequentially
        batch_index = np.arange(seg_start, seg_start + seg_n_frames, dtype=int)

        # Clamp with the cached frame count to prevent out-of-range indices
        cached_len = self._video_lengths.get(dataset_idx)
        if cached_len is not None:
            batch_index = np.clip(batch_index, 0, cached_len - 1)

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

        # Load video frames
        with VideoReader_contextmanager(tar_video_path, num_threads=2) as video_reader:
            pixel_values = get_video_reader_batch(video_reader, batch_index)
        with VideoReader_contextmanager(src_video_path, num_threads=2) as video_reader:
            src_pixel_values = get_video_reader_batch(video_reader, batch_index)

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

                if len(sample) > 0:
                    break
            except Exception as e:
                dataset_idx = self.expanded_items[idx % self.length][0]
                print(e, self._raw_dataset[dataset_idx])
                idx = random.randint(0, self.length - 1)
                try_count += 1
        if len(sample) == 0:
            raise RuntimeError(f"Failed to fetch a valid sample after 20 retries, last idx={idx}")

        return sample
