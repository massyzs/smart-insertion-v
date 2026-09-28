import math
from typing import List, Optional

import torch
from torch import nn
from torchvision import transforms as v2

from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen2Config,
)
import os

def _find_subseq(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i+len(sub)] == sub:
            return i
    return -1

def compute_user_start_drop_idx(tokenizer, system_prompt: str) -> int:
    """
    Returns the token index where user content starts (just after `<|im_start|>user\n`).
    Works with Qwen3-VL / Qwen2.5-VL apply_chat_template.
    """
    # 1) Build a minimal conversation using the same chat template as tokenize_fn
    conv = []
    if system_prompt is not None:
        conv.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    SENTINEL = "<<<__SENTINEL_USER_TEXT__>>>"

    conv.append({"role": "user", "content": [{"type": "text", "text": SENTINEL}]})

    # 2) Render with apply_chat_template (same flags as in tokenize_fn)
    rendered = tokenizer.apply_chat_template(conv, add_generation_prompt=True)

    # 3) Tokenize both the full string and just the sentinel
    full_ids = tokenizer(text=rendered, return_tensors="pt", padding=False).input_ids[0].tolist()
    sent_ids = tokenizer(text=SENTINEL, return_tensors="pt", padding=False).input_ids[0].tolist()

    # 4) Find sentinel start in the full sequence
    start = _find_subseq(full_ids, sent_ids)
    if start == -1:
        # Very rare: if the sentinel got split weirdly, fall back to string search and re-tokenize prefix
        # to compute a robust boundary.
        prefix = rendered.split(SENTINEL)[0]
        start = len(tokenizer(prefix, return_tensors="pt").input_ids[0])
    return int(start)


class MLLMInContextConfig(PretrainedConfig):
    model_type = "mllm-in-context"

    def __init__(
        self,
        mllm_id: str = "Qwen3-VL",  # model id or local path of the Qwen-VL backbone
        num_metaqueries: int = 0,
        _gradient_checkpointing: bool = True,
        max_input_text_tokens: int = 1024,
        # For use_chat_template=True, system_prompt should be plain text (no special tokens)
        # apply_chat_template automatically adds <|im_start|>, <|im_end|> and other tokens
        # Note: the content order is [video] [image] [text]; this must be stated explicitly in the prompt
        # system_prompt_edit: str = "You are an assistant. The user will provide: 1) A source video, 2) A reference image, 3) A text instruction. Describe the key features of the input video (color, shape, size, texture, objects, background), then according to user's text instruction, explain how reference image should be adjusted to be suitable to the video style and how to be inserted into the video. Generate a new video that meets the user's requirements while maintaining consistency with the original video where appropriate.",
        # system_prompt_t2v: str = "You are an assistant. The user will provide text description to generate a video. Follow user's instruction and make the described scene more concrete. You need to describe the video in detail including the visual elements, actions, environment, and atmosphere to help generate a vivid video that matches the user's description.",
        # system_prompt_i2v: str = "You are an assistant. The user will provide text description and a reference image to generate a video. Follow user's instruction and describe the scene detaily (what object should act in what way). You need to describe the video in detail including the visual elements, actions, environment, and atmosphere to help generate a vivid video that matches the user's description.",
        system_prompt_ref_edit: str = "You are an AI image analyzer. Your task is to examine two images: the first image is the source image, and the second image defines the target style. Analyze the visual style of the second image and determine how the first image should be modified to match that style. Consider aspects such as color palette, lighting, texture, atmosphere, composition, and overall visual tone. Describe the necessary adjustments to the first image so that it visually fits the style of the second image while preserving the original content.",
        # NOTE: kept byte-identical to the prompt used during training (including the "accordig" typo); do not edit.
        system_prompt_edit: str = "You are an AI image analyzer. You will be given two images and a prompt describing how the object should be inserted into a scene. The first Image contains the object to be inserted and the second Image is the target scene. Analyze the two images and the prompt, then determine: 1) the precise suitable insertion position in the second Image accordig to coarse prompt, 2) the appropriate scale, orientation, and perspective of the object from the first Image, 3) how the object from the first Image should adjust its lighting, color tone, shadows, and style to match the environment in the second Image so the insertion looks natural and not artificially inserted. Prompt is '{XXX}'",

        system_prompt_i2v: str = "You are an AI image analyzer. Carefully examine the provided image and identify all visible objects. For each object, describe its exact location within the image, visual style, and key attributes such as color, shape, size, material, and spatial relationships with other objects.",


        system_prompt_t2v: str = "You are a cinematic prompt engineer. Expand the user's input into a high-quality video description. You MUST explicitly describe: 1) Visual details of the subject and environment, 2) Lighting and atmosphere, 3) Specific camera movements (e.g., pan, zoom, tracking), and 4) Dynamic actions. Make the description vivid, concrete, and visually rich.",



        use_chat_template: bool = True,
        crop_system_tokens: bool = True,
        crop_vision_tokens: bool = False,
        system_tokens_drop_idx: int = 0,
        mode: str = "t2v",  # "t2v" or "edit" or "i2v"
        **kwargs,
    ):
        super().__init__()
        self.system_prompt_ref_edit = system_prompt_ref_edit
        self.system_prompt_edit = system_prompt_edit
        self.system_prompt_i2v = system_prompt_i2v
        self.system_prompt_t2v = system_prompt_t2v
        self.mllm_id = mllm_id
        self.num_metaqueries = num_metaqueries
        self._gradient_checkpointing = _gradient_checkpointing
        self.max_input_text_tokens = max_input_text_tokens

        self.use_chat_template = use_chat_template
        self.crop_system_tokens = crop_system_tokens
        self.crop_vision_tokens = crop_vision_tokens
        self.system_tokens_drop_idx = system_tokens_drop_idx
        self.mode = mode
        


class MLLMInContext(PreTrainedModel):
    config_class = MLLMInContextConfig

    def __init__(
        self,
        
        config: MLLMInContextConfig,
    ) -> None:
        super().__init__(config)
        self._gradient_checkpointing = config._gradient_checkpointing
        self.config = config
        self.mode = self.config.mode
        if self.mode == "t2v":
            # self.tokenizer.system_prompt = self.config.system_prompt_t2v
            self.config.system_prompt = self.config.system_prompt_t2v
        elif self.mode == "edit":
            # self.tokenizer.system_prompt = self.config.system_prompt_edit
            self.config.system_prompt = self.config.system_prompt_edit
        elif self.mode == "i2v":
            # self.tokenizer.system_prompt = self.config.system_prompt_i2v
            self.config.system_prompt = self.config.system_prompt_i2v
        # Accepts Qwen3-VL / Qwen2.5-VL ids; the backbone is instantiated with Qwen3VLForConditionalGeneration
        if "Qwen3-VL" in config.mllm_id or "Qwen2.5-VL" in config.mllm_id:
            self.mllm_type = "qwenvl"
        else:
            raise ValueError(f"Unsupported model: {config.mllm_id}. Supported: Qwen3-VL, Qwen2.5-VL")
        
        if self.mllm_type == "qwenvl":
            
            self.mllm_backbone = Qwen3VLForConditionalGeneration.from_pretrained(
                config.mllm_id, 
                attn_implementation="sdpa", 
                # attn_implementation="flash_attention_2", 
                torch_dtype=torch.bfloat16
            )
            # self.mllm_backbone.model.config.use_sliding_window = False
            # self.mllm_backbone.model.config.sliding_window = None

            # If use metaquery
            if config.num_metaqueries > 0:
                # Handle the structural differences between Qwen2.5-VL and Qwen3-VL
                embed_tokens = self.mllm_backbone.get_input_embeddings()
                # print(f"Before resize embed_tokens: {embed_tokens.weight.shape}")
                num_embeddings = embed_tokens.num_embeddings
                self.num_embeddings = num_embeddings
                try:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2
                    )
                except:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2, mean_resizing=False
                    )
                embed_tokens = self.mllm_backbone.get_input_embeddings()
                # print(f"After resize embed_tokens: {embed_tokens.weight.shape}")

                def freeze_hook(grad):
                    # print(f"  [Query] Original tokens (frozen): {self.num_embeddings}")
                    
                    # print(f"  Gradient shape: {grad.shape}")
                    # print(f"  Pre-zero grad norm: {grad.norm().item():.6f}")
                    # print(f"  Pre-zero original token grad norm: {grad[:self.num_embeddings].norm().item():.6f}")
                    # print(f"  Pre-zero new token grad norm: {grad[self.num_embeddings:].norm().item():.6f}")
                    if grad is None:
                        print(f"  Gradient is None!")
                    elif torch.isnan(grad).any():
                        print(f"  Gradient contains NaN!")
                    elif grad.norm().item() == 0.0:
                        print(f"  All gradients are exactly zero - gradient flow broken!")
                    
                    # Zero out gradients for original tokens
                    grad[: self.num_embeddings].zero_()
                    
                    # print(f"  Post-zero original token grad norm: {grad[:self.num_embeddings].norm().item():.6f}")
                    # print(f"  Post-zero new token grad norm: {grad[self.num_embeddings:].norm().item():.6f}")
                    
                    return grad
                embed_tokens.weight.register_hook(freeze_hook)
            
            # Get hidden_size from embed_tokens (compatible with Qwen2-VL and Qwen3-VL)
            embed_tokens = self.mllm_backbone.get_input_embeddings()
            self.mllm_hidden_size = embed_tokens.weight.shape[1]  # [vocab_size, hidden_size]
            # print(f"[QWEN] hidden_size from embed_tokens: {self.mllm_hidden_size}")
            min_pixels = 256 * 28 * 28
            # max_pixels = 1280 * 28 * 28
            max_pixels = 480 * 854 
            self.tokenizer = AutoProcessor.from_pretrained(
                config.mllm_id, 
                min_pixels=min_pixels, 
                max_pixels=max_pixels
            ) # Qwen2_5_VLProcessor
            self.tokenizer.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = None
            # 3B 2048
            # 7B 3584

        else:
            raise ValueError(f"Unsupported model: {config.mllm_id}")

        self.tokenizer.mllm_type = self.mllm_type
        self.tokenizer.max_input_text_tokens = config.max_input_text_tokens
        self.tokenizer.num_metaqueries = config.num_metaqueries
        if self.mode == "t2v":
            self.tokenizer.system_prompt = config.system_prompt_t2v
        elif self.mode == "edit":
            self.tokenizer.system_prompt = config.system_prompt_edit
        elif self.mode == "i2v":
            self.tokenizer.system_prompt = config.system_prompt_i2v
        else:
            return ValueError(f"Unsupported mode: {self.mode}. Supported: t2v, edit")
        # self.tokenizer.system_prompt = config.system_prompt
        # self.tokenizer.system_prompt_edit = config.system_prompt_edit
        # self.tokenizer.system_prompt_t2v = config.system_prompt_t2v
        self.tokenizer.use_chat_template = getattr(config, 'use_chat_template', True)
        self.tokenizer.crop_system_tokens = getattr(config, 'crop_system_tokens', True)

        
        drop_idx = compute_user_start_drop_idx(self.tokenizer, config.system_prompt)
        # self.tokenizer.system_tokens_drop_idx = drop_idx
        # print(f"[AUTO-CROP] Detected system_tokens_drop_idx={drop_idx}")
        self.tokenizer.system_tokens_drop_idx = drop_idx

        self.pad_token_id = getattr(
            self.tokenizer, "tokenizer", self.tokenizer
        ).pad_token_id

        # If use Metaqueies we need to add special token
        if config.num_metaqueries > 0:
            print(f"Using metaqueries with {config.num_metaqueries} query")
            tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        f"<pad_token_{i}>"
                        for i in range(num_embeddings - len(tokenizer))
                    ]
                }
            )
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<begin_of_img>", "<end_of_img>"]
                    + [f"<img{i}>" for i in range(self.tokenizer.num_metaqueries)]
                }
            )
            self.boi_token_id = tokenizer.convert_tokens_to_ids("<begin_of_img>")
            self.eoi_token_id = tokenizer.convert_tokens_to_ids("<end_of_img>")

        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.gradient_checkpointing_enable(
                    {"use_reentrant": False}
                )
                print("Enable Gradient Checkpoint for MLLM backbone")
            except:
                pass


    def get_tokenizer(self):
        return self.tokenizer

    def get_tokenize_fn(self):
        return self.tokenize_fn

    def get_resize_fn(self):
        return self.resize_fn
    
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Extract hidden states using attention mask, similar to QwenImage pipeline"""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def _crop_system_tokens(self, hidden_states_list: List[torch.Tensor], drop_idx: int = 0):
        """Crop system prompt tokens from the beginning of sequences"""
        if drop_idx > 0:
            return [h[drop_idx:] for h in hidden_states_list]
        return hidden_states_list
    
    def _repad_to_max_length(self, hidden_states_list: List[torch.Tensor]):
        """Re-pad sequences to maximum length after cropping"""
        if not hidden_states_list:
            return None, None
            
        # Create attention masks for each sequence
        attn_mask_list = [torch.ones(h.size(0), dtype=torch.long, device=h.device) for h in hidden_states_list]
        
        # Find maximum sequence length
        max_seq_len = max([h.size(0) for h in hidden_states_list])
        
        # Pad sequences to max length
        padded_hidden_states = torch.stack([
            torch.cat([h, h.new_zeros(max_seq_len - h.size(0), h.size(1))]) 
            for h in hidden_states_list
        ])
        
        # Pad attention masks
        padded_attention_mask = torch.stack([
            torch.cat([mask, mask.new_zeros(max_seq_len - mask.size(0))]) 
            for mask in attn_mask_list
        ])
        
        return padded_hidden_states, padded_attention_mask

    @staticmethod
    @torch.no_grad()
    def tokenize_fn(
        tokenizer, 
        texts,         # ["" x b] one sentence per example
        images=None,   # [[PIL.Image.Image x num] x b]
        second_image=None,   # [[PIL.Image.Image x num] x b]
        videos=None,   # [[torch.tensor (f h w c) 0-255 x num] x b]
        text_response=None,
        add_queires=True,  # For video/image generation we add queires otherwise for text generation we don't add them.
        add_generation_prompt=True
    ):
        if not isinstance(texts, List):
            texts = [texts]

        # Check if we should use chat template or direct tokenization
        if not tokenizer.use_chat_template:
            assert not images
            print(f"[DEBUG] Using direct tokenization (no chat template)")
            # print(f"[DEBUG] texts(s) before tokenization: {texts}")
            # Direct tokenization - no images, no chat template
            text_inputs = tokenizer(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.max_input_text_tokens,
            )
            
            # print(f"[DEBUG] Direct tokenization - input_ids shape: {text_inputs['input_ids'].shape}")
            return text_inputs.values()

        # Chat template mode (original behavior)
        # print(f"[DEBUG] Using chat template mode")
        
        prefix = (
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": tokenizer.system_prompt}],
                },
            ]
            if tokenizer.system_prompt is not None
            else []
        )

        # if not add_generation_prompt or tokenizer.num_metaqueries <= 0:
        #     suffix = ""
        # else:  # metauqery token
        #     suffix = (
        #         "\n<begin_of_img>"
        #         + "".join([f"<img{i}>" for i in range(tokenizer.num_metaqueries)])
        #         + "<end_of_img><|im_end|>"
        #     )
        suffix = ""

        texts = [
            tokenizer.decode(
                tokenizer(text=text, return_tensors="pt", padding=False).input_ids[
                    0, : tokenizer.max_input_text_tokens
                ]
            )
            for text in texts
        ]

        if images is not None and len(images) == 0:
            images = None
        if images is not None:
            # If images is not a list, wrap it in a list
            if not isinstance(images, list):
                images = [images]
            # If each batch item is not a list, wrap it in a single-element list (or empty list if None)
            for i, img in enumerate(images):
                if img and not isinstance(img, list):
                    images[i] = [img]

        if second_image is not None and len(second_image) == 0:
            second_image = None
        if second_image is not None:
            if not isinstance(second_image, list):
                second_image = [second_image]
            for i, img in enumerate(second_image):
                if img and not isinstance(img, list):
                    second_image[i] = [img]
        
        if videos is not None and len(videos) == 0:
            videos = None
        if videos is not None:
            if not isinstance(videos, list):
                videos = [videos]
            for i, vids in enumerate(videos):
                if vids and not isinstance(vids, list):
                    videos[i] = [vids]

        batch_size = len(texts)
        if images is not None and len(images) != batch_size:
            raise ValueError(f"images batch ({len(images)}) must match texts ({batch_size})")
        if videos is not None and len(videos) != batch_size:
            raise ValueError(f"videos batch ({len(videos)}) must match texts ({batch_size})")

        # Build conversations: videos first, then images, then text
        # Order: [video] [image] [text], consistent with the system_prompt description:
        # "1) A source video, 2) A reference image, 3) A text instruction"
        # If a sample has no images/videos, it’s just the text.
        conversations = []
        for i in range(batch_size):
            content = []
            imgs = images[i] if images is not None else None
            second_imgs = second_image[i] if second_image is not None else None
            vids = videos[i] if videos is not None else None
            # VIDEO first (consistent with the system_prompt order: 1) source video)
            if vids:
                content.extend([{"type": "video"} for _ in vids])
            # IMAGE second (consistent with the system_prompt order: 2) reference image)
            if imgs:
                content.extend([{"type": "image"} for _ in imgs])
            if second_imgs:
                content.extend([{"type": "image"} for _ in second_imgs])
            content.append({"type": "text", "text": texts[i]})

            conversations.append(
                prefix
                + [
                    {
                        "role": "user",
                        "content": content,
                    },
                ]
            )

        kwargs = {}
        if images is not None or second_image is not None:
            merged_images = []
            for i in range(batch_size):
                sample_images = []
                if images is not None and images[i]:
                    sample_images.extend(images[i])
                if second_image is not None and second_image[i]:
                    sample_images.extend(second_image[i])
                merged_images.append(sample_images)
            if any(merged_images):
                kwargs["images"] = merged_images
        if videos is not None:
            kwargs["videos"] = videos

        prompts = [
            tokenizer.apply_chat_template(
                conv, 
                add_generation_prompt=True
            )
            for conv in conversations
        ]
        if text_response is not None:
            prompts = [p + t.strip() for p, t in zip(prompts, text_response)]
        if tokenizer.num_metaqueries > 0 and add_queires:
            prompts = [p + suffix for p in prompts]


        
        inputs = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            # truncation=True,  # we don't want to truncate image token
            # max_length=max_len,
            **kwargs,
        )

      
        # DEBUG PRINT
        if "input_ids" in inputs:
            decoded_inputs = tokenizer.batch_decode(
                inputs["input_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            for i, decoded in enumerate(decoded_inputs):
                # print(f"[DEBUG] \n--- Decoded input {i} ---\n{repr(decoded)}")
                pass
        else:
            print("[DEBUG] No input_ids found in inputs.")
       
        # DEBUG: Log the keys returned by QwenVL tokenizer
        # print(f"[DEBUG] QwenVL tokenizer returned keys: {list(inputs.keys())}")
        # for key, value in inputs.items():
        #     if hasattr(value, 'shape'):
        #         print(f"[DEBUG] {key}: shape={value.shape}, dtype={value.dtype}")
        #     else:
        #         print(f"[DEBUG] {key}: {type(value)}")
        
        return inputs

    def _tok_id(self, s: str):
        tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        try:
            tid = tok.convert_tokens_to_ids(s)
            return tid if isinstance(tid, int) and tid != -1 else None
        except Exception:
            return None

    def _crop_hidden_bs1(self,
                    input_ids: torch.Tensor,        # [1, T]
                    attention_mask: torch.Tensor,   # [1, T]
                    last_hidden: torch.Tensor       # [1, T, D]
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        B=1. If vision markers exist, keep tokens strictly AFTER the last <|vision_end|>.
        Otherwise, crop system tokens using tokenizer.system_tokens_drop_idx.
        Returns: (prompt_embeds [1, L, D], new_attn [1, L])
        """
        assert input_ids.shape[0] == 1 and attention_mask.shape[0] == 1 and last_hidden.shape[0] == 1
        ids  = input_ids[0]           # [T]
        attn = attention_mask[0]      # [T]
        hs   = last_hidden[0]         # [T, D]
        assert ids.shape[0] == attn.shape[0] == hs.shape[0]
        T, D = hs.shape

        valid = (attn == 1).nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            # nothing valid; return a single zero token for shape sanity
            print("[KEEP-TEXT] ERROR ! No valid tokens in attention_mask, returning dummy zero.")
            return hs.new_zeros(1, 1, D), attn.new_zeros(1, 1)

        start_idx = None
        if self.config.crop_vision_tokens:
            ve_id = self._tok_id("<|vision_end|>")
            if ve_id is not None:
                ve_pos = (ids == ve_id).nonzero(as_tuple=False).flatten()
                if ve_pos.numel() > 0:
                    # vision present: keep AFTER the vision block
                    start_idx = int(ve_pos.max().item()) + 1
                    # print(f"[KEEP-TEXT] Found <|vision_end|> at positions {ve_pos.tolist()}, using start_idx={start_idx}")

        if start_idx is None:
            # no vision: crop system tokens
            drop_idx = int(getattr(self.tokenizer, "system_tokens_drop_idx", 0))
            start_idx = int(valid.min().item() + drop_idx)
            # print(f"[KEEP-TEXT] No <|vision_end|> found → using system_tokens_drop_idx={drop_idx}, start_idx={start_idx}")

        # end at last valid token
        end_idx = int(valid.max().item()) + 1
        start_idx = max(0, min(start_idx, end_idx))  # clamp + guard
        # print(f"[KEEP-TEXT] Final slice: start={start_idx}, end={end_idx}, total_len={T}")

        kept = hs[start_idx:end_idx]                 # [L, D]
        if kept.numel() == 0:
            print("[KEEP-TEXT] Slice resulted in empty tensor, returning dummy zero.")
            return hs.new_zeros(1, 1, D), attn.new_zeros(1, 1)

        # --- DEBUG: show a small decoded window after crop ---
        try:
            tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            window_ids = ids[start_idx : end_idx].tolist()
            window_text = tok.decode(window_ids, skip_special_tokens=False)
            # print(f"[KEEP-TEXT] Preview after crop → {repr(window_text)}")
        except Exception as e:
            print(f"[KEEP-TEXT] Preview decode failed: {e}")

        new_attn = attn.new_ones(kept.shape[0])      # [L]
        # print(f"[KEEP-TEXT] Kept hidden states shape={kept.shape}, new_attn shape={new_attn.shape}")
        return kept.unsqueeze(0), new_attn.unsqueeze(0)


    def _extract_text_and_queries_bs1(
        self,
        input_ids: torch.Tensor,        # [1, T]
        attention_mask: torch.Tensor,   # [1, T]
        last_hidden: torch.Tensor       # [1, T, D]
    ):
        """
        Returns:
            embeds : [1, L, D]   (text first, then query tokens)
            attn   : [1, L]
        Assumes bs=1.
        """
        assert input_ids.shape[0] == 1 and attention_mask.shape[0] == 1 and last_hidden.shape[0] == 1

        ids  = input_ids[0]       # [T]
        attn = attention_mask[0]  # [T]
        hs   = last_hidden[0]     # [T, D]
        T, D = hs.shape

        # --- valid span (handles left padding) ---
        valid = (attn == 1).nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            print("[TEXT+QUERY] No valid tokens; returning empty.")
            return hs.new_zeros(1, 0, D), attn.new_zeros(1, 0)

        first_valid = int(valid.min().item())
        end_idx = int(valid.max().item()) + 1

        # --- choose start_idx: vision crop > system crop ---
        start_idx = None

        def _tok_id(token_str: str):
            tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            try:
                tid = tok.convert_tokens_to_ids(token_str)
                return tid if isinstance(tid, int) and tid != -1 else None
            except Exception:
                return None

        # always crop vision token
        ve_id = _tok_id("<|vision_end|>")
        if ve_id is not None:
            ve_pos = (ids == ve_id).nonzero(as_tuple=False).flatten()
            if ve_pos.numel() > 0:
                start_idx = int(ve_pos.max().item()) + 1
                # print(f"[TEXT+QUERY] vision_end at {ve_pos.tolist()} → start_idx={start_idx}")

        if start_idx is None:
            drop_idx = int(getattr(self.tokenizer, "system_tokens_drop_idx", 0))
            start_idx = first_valid + drop_idx
            # print(f"[TEXT+QUERY] no vision_end → drop system drop_idx={drop_idx}, start_idx={start_idx}")

        start_idx = max(0, min(start_idx, end_idx))

        kept_hs  = hs[start_idx:end_idx]     # [L, D]
        kept_ids = ids[start_idx:end_idx]    # [L]
        L = kept_hs.shape[0]

        if L == 0:
            # print("[TEXT+QUERY] crop produced empty; returning empty.")
            return hs.new_zeros(1, 0, D), attn.new_zeros(1, 0)

        try:
            tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            window_text = tok.decode(
                kept_ids.tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            # print(f"[TEXT+QUERY] Preview after crop → {repr(window_text)}")
        except Exception as e:
            print(f"[TEXT+QUERY] Preview decode failed: {e}")

        # --- split text vs query ---
        device = kept_hs.device
        text_mask  = torch.ones(L, dtype=torch.bool, device=device)
        query_mask = torch.zeros(L, dtype=torch.bool, device=device)

        if getattr(self.tokenizer, "num_metaqueries", 0) > 0:
            boi = getattr(self, "boi_token_id", None)
            eoi = getattr(self, "eoi_token_id", None)

            if boi is not None and eoi is not None:
                boi_pos = (kept_ids == boi).nonzero(as_tuple=False).flatten()
                eoi_pos = (kept_ids == eoi).nonzero(as_tuple=False).flatten()

                if boi_pos.numel() > 0 and eoi_pos.numel() > 0:
                    boi_i = int(boi_pos[0].item())
                    eoi_i = int(eoi_pos[0].item())

                    if eoi_i > boi_i + 1:
                        query_mask[boi_i + 1 : eoi_i] = True

                    text_mask[boi_i : eoi_i + 1] = False
                else:
                    print("[TEXT+QUERY] BOI/EOI not found → all tokens treated as text.")
            else:
                print("[TEXT+QUERY] missing BOI/EOI ids → all tokens treated as text.")

        # --- concat text then queries ---
        text_hs  = kept_hs[text_mask]     # [Lt, D]
        query_hs = kept_hs[query_mask]    # [Lq, D]

        concat_hs = torch.cat([text_hs, query_hs], dim=0)
        concat_attn = torch.ones(concat_hs.shape[0], device=device, dtype=attn.dtype)

        # print(
            # f"[TEXT+QUERY] final concat shape={concat_hs.shape} "
            # f"(text={text_hs.shape}, query={query_hs.shape})"
        # )

        return concat_hs.unsqueeze(0), concat_attn.unsqueeze(0)

    def encode_condition(
        self, input_ids, attention_mask, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_ts,mode,
    ):
        if self.mllm_type == "qwenvl":
            outputs = self.mllm_backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                output_hidden_states=True,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                mode=mode
            )
            last_hidden = outputs.hidden_states[-1]  # Last layer hidden states
            # print(f"[MLLM] QwenVL hidden states shape: {last_hidden.shape}")
        else:
            raise ValueError(f"Unsupported model: {self.mllm_type}")


        
        if input_ids.shape[0] == 1:  # 1. batch size 1: process directly
            if self.tokenizer.num_metaqueries > 0:
                prompt_embeds, attention_mask = self._extract_text_and_queries_bs1(
                    input_ids, attention_mask, last_hidden
                )
            else:
                prompt_embeds, attention_mask = self._crop_hidden_bs1(input_ids, attention_mask, last_hidden)
                # print(f"[TEXT-ONLY per rule] {prompt_embeds.shape}")
        else:  # 2. batch size > 1: process each sample in a loop
            embeds_list = []
            attn_list = []
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                # Take a single sample, keeping the shape as [1, T, ...] to satisfy the assertions in the _bs1 functions
                curr_input_ids = input_ids[i : i+1]
                curr_mask = attention_mask[i : i+1]
                curr_hidden = last_hidden[i : i+1]
                if self.tokenizer.num_metaqueries > 0:
                    emb, _ = self._extract_text_and_queries_bs1(
                        curr_input_ids, curr_mask, curr_hidden
                    )
                else:
                    emb, _ = self._crop_hidden_bs1(curr_input_ids, curr_mask, curr_hidden)
                # _repad_to_max_length expects a list of [L, D] (i.e., with the batch dim removed)
                embeds_list.append(emb.squeeze(0))
                # attn_list.append(attention_mask.squeeze(0))
            prompt_embeds, attention_mask = self._repad_to_max_length(embeds_list)
        
        # Return raw
        return prompt_embeds, attention_mask
    
    def generation(
        self, input_ids, attention_mask, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_ts
    ):
        if self.mllm_type == "qwenvl":
            generated_ids = self.mllm_backbone.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                max_new_tokens=1000,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            output_text = self.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        else:
            raise ValueError(f"Unsupported model: {self.mllm_type}")
        return output_text


    @torch.no_grad()
    def get_prompt_embeddings(self, prompts, prompt2sys=None,images=None, second_image=None,videos=None, device=None, dtype=None, mode="t2v"):
        """
        Get encoder_hidden_states from the Qwen encoder.
        
        Args:
            prompts: List of text prompts
            images: [[PIL.Image,...] x b] or None
            videos: [[torch.tensor (f h w c) 0-255] x b] or None
            device: Target device
            dtype: Target dtype
            mode: "t2v" (text only) or "edit" (text + images + videos)
        Returns:
            prompt_embeds: [B, seq_len, hidden_size]
            prompt_attention_mask: [B, seq_len]
        """
        # if mode == "t2v":
        #     # t2v mode: text only
        #     images = None
        #     videos = None
        # elif mode == "edit":
        #     # edit mode: use images and videos if provided
        #     if not images:
        #         images = None
        #     if not videos:
        #         videos = None
        # elif mode == "i2v":
        #     videos = None
        # else:
        #     raise ValueError(f"Unsupported mode: {mode}. Use 't2v' or 'edit'.")
        if mode == "ref_edit":
            self.config.system_prompt = self.config.system_prompt_ref_edit
            original_drop_idx = self.tokenizer.system_tokens_drop_idx
            original_system_prompt = self.tokenizer.system_prompt

            # if prompt2sys is not None:
            #     self.config.system_prompt = self.config.system_prompt.replace("{XXX}", prompt2sys)
            drop_idx = compute_user_start_drop_idx(self.tokenizer, self.config.system_prompt)
            # print(f"[AUTO-CROP] Detected system_tokens_drop_idx={drop_idx}")
            self.tokenizer.system_tokens_drop_idx = drop_idx
            self.tokenizer.system_prompt = self.config.system_prompt

        elif mode == "edit":
            self.config.system_prompt = self.config.system_prompt_edit
            original_drop_idx = self.tokenizer.system_tokens_drop_idx
            original_system_prompt = self.tokenizer.system_prompt

            if isinstance(prompt2sys, str):
                self.config.system_prompt = self.config.system_prompt.replace("{XXX}", prompt2sys)
            elif isinstance(prompt2sys, list):
                self.config.system_prompt = self.config.system_prompt.replace("{XXX}", prompt2sys[0])

            drop_idx = compute_user_start_drop_idx(self.tokenizer, self.config.system_prompt)
            # print(f"[AUTO-CROP] Detected system_tokens_drop_idx={drop_idx}")
            self.tokenizer.system_tokens_drop_idx = drop_idx
            self.tokenizer.system_prompt = self.config.system_prompt





        batch = self.tokenize_fn(self.tokenizer, prompts, images, second_image)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        prompt_embeds, prompt_attention_mask = self.encode_condition(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            mode=mode,
        )

        if mode == "ref_edit":
            self.tokenizer.system_tokens_drop_idx = original_drop_idx
            self.tokenizer.system_prompt = original_system_prompt
            self.config.system_prompt = self.config.system_prompt_edit
        elif mode == "edit":
            self.tokenizer.system_tokens_drop_idx = original_drop_idx
            self.tokenizer.system_prompt = original_system_prompt
            self.config.system_prompt = self.config.system_prompt_edit

        # return prompt_embeds[].to(dtype)
        return prompt_embeds.to(dtype) if dtype else prompt_embeds, prompt_attention_mask


# ============================================================================
# Example: how to pass source_video + reference_image + prompt
# ============================================================================

def example_prepare_inputs_for_video_with_reference(
    tokenizer,  # Qwen3VLProcessor (from MLLMInContext.get_tokenizer())
    source_video: torch.Tensor,      # shape: (F, H, W, C), dtype: uint8, range: 0-255
    reference_image,                 # PIL.Image.Image
    prompt: str,                     # User text instruction
    num_frames: int = None,          # Optional: number of frames to sample (if None, use all frames)
):
    """
    Prepare inputs for Qwen3VL, supporting source video + reference image + prompt.
    
    Args:
        tokenizer: Qwen3VLProcessor (obtained via MLLMInContext.get_tokenizer())
        source_video: source video tensor
            - shape: (F, H, W, C) where F=num frames, H=height, W=width, C=num channels (3)
            - dtype: torch.uint8
            - range: 0-255
        reference_image: PIL.Image.Image reference image
        prompt: user text instruction, e.g. "Insert the person from the reference image into the video"
        num_frames: optional, number of frames to sample
    
    Returns:
        dict: contains all arguments to be passed to encode_condition
    
    Usage example:
        ```python
        # 1. Load the video (using torchvision or decord)
        import torchvision.io as io
        video_frames, _, _ = io.read_video("source.mp4")  # (F, H, W, C), uint8
        
        # 2. Load the reference image
        from PIL import Image
        ref_image = Image.open("reference.png").convert("RGB")
        
        # 3. Prepare inputs
        inputs = example_prepare_inputs_for_video_with_reference(
            tokenizer=mllm_model.get_tokenizer(),
            source_video=video_frames,
            reference_image=ref_image,
            prompt="Naturally blend the character from the reference image into the video scene",
            num_frames=16,  # sample 16 frames
        )
        
        # 4. Encode
        prompt_embeds, attention_mask = mllm_model.encode_condition(**inputs)
        ```
    """
    # -------------------------------------------------------------------------
    # Handle frame sampling
    # -------------------------------------------------------------------------
    F, H, W, C = source_video.shape
    
    # if num_frames is not None and num_frames < F:
    #     # Uniformly sample the specified number of frames
    #     indices = torch.linspace(0, F - 1, num_frames).long()
    #     source_video = source_video[indices]
    #     print(f"[PREPARE] Sampled from {F} frames down to {num_frames} frames")
    
    # -------------------------------------------------------------------------
    # Format into the layout expected by tokenize_fn
    # -------------------------------------------------------------------------
    # videos: [[torch.tensor (F, H, W, C)] x batch_size]
    # images: [[PIL.Image] x batch_size]
    # texts: [str x batch_size]
    
    batch_videos = [[source_video]]           # batch_size=1, 1 video
    batch_images = [[reference_image]]        # batch_size=1, 1 reference image
    batch_texts = [prompt]                    # batch_size=1
    
    # -------------------------------------------------------------------------
    # Call tokenize_fn
    # -------------------------------------------------------------------------
    # Content construction order: video first, then image, then text last
    # i.e.: [video1, video2, ...] [image1, image2, ...] [text]
    # This is consistent with the description in system_prompt:
    # "1) A source video, 2) A reference image, 3) A text instruction"
    
    inputs = MLLMInContext.tokenize_fn(
        tokenizer=tokenizer,
        texts=batch_texts,
        images=batch_images,      # reference image
        videos=batch_videos,      # source video
        add_queires=True,
        add_generation_prompt=True
    )
    
    # -------------------------------------------------------------------------
    # Prepare the return dict
    # -------------------------------------------------------------------------
    # The inputs returned by tokenize_fn is a BatchFeature containing:
    # - input_ids
    # - attention_mask
    # - pixel_values (images)
    # - image_grid_thw
    # - pixel_values_videos (videos)
    # - video_grid_thw
    # - second_per_grid_ts (Qwen3VL-specific)
    
    result = {
        "input_ids": inputs.get("input_ids"),
        "attention_mask": inputs.get("attention_mask"),
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "pixel_values_videos": inputs.get("pixel_values_videos"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "second_per_grid_ts": inputs.get("second_per_grid_ts"),
    }
    

    
    return result


def get_special_tokens_info():
    """
    Return the special token information used by Qwen3VL.
    
    Special token description:
    - <|im_start|> : message start token
    - <|im_end|>   : message end token
    - <|vision_start|> : start of visual content
    - <|vision_end|>   : end of visual content
    - <|image_pad|>    : image placeholder
    - <|video_pad|>    : video placeholder
    
    When use_chat_template=True:
    - These tokens are added automatically by apply_chat_template
    - system_prompt only needs to be plain text
    
    When use_chat_template=False:
    - These tokens must be included in the prompt manually
    """
    return {
        "message_start": "<|im_start|>",
        "message_end": "<|im_end|>",
        "vision_start": "<|vision_start|>",
        "vision_end": "<|vision_end|>",
        "image_pad": "<|image_pad|>",
        "video_pad": "<|video_pad|>",
        
        # For metaquery (if enabled)
        "begin_of_img": "<begin_of_img>",
        "end_of_img": "<end_of_img>",
    }