# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference-image fidelity reward for Ref2VA using CLIP.

Ref2VA conditions generation on one or more reference images (and optionally
reference video/audio, see ``examples/diffusionnft_trainer/minimax_h3/prepare_ref2va_data.py``).
Existing cross-modal rewards (:mod:`clap`, :mod:`imagebind`) only measure
audio/video alignment against the *text prompt* -- they say nothing about
whether the generated video actually preserves the subject/appearance given
in the reference. This scorer closes that gap for image references: it
embeds the reference image(s) and a handful of sampled generated-video frames
with CLIP and reports their mean cosine similarity.

Video and audio references (``extra_info["source_videos"]`` /
``extra_info["source_audios"]``) are not scored here; only
``extra_info["source_images"]``, which every Ref2VA row may carry regardless
of task variant, is used.
"""

import threading

import torch
import torch.nn.functional as F
from PIL import Image
from verl.utils.device import get_device_name

_DEFAULT_MODEL = "openai/clip-vit-base-patch32"
_DEFAULT_NUM_FRAMES = 4
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()


def _load_clip(model_name_or_path: str, device: str):
    key = (model_name_or_path, device)
    if key not in _MODEL_CACHE:
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_name_or_path).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name_or_path)
        _MODEL_CACHE[key] = (model, processor)
    return _MODEL_CACHE[key]


def _reference_images(extra_info: dict) -> list[Image.Image]:
    paths = extra_info.get("source_images") or []
    if isinstance(paths, str):
        paths = [paths]
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        raise KeyError("Ref fidelity reward requires reference image paths in extra_info['source_images'].")
    return images


def _sample_generated_frames(solution_image, num_frames: int) -> list[Image.Image]:
    from verl_omni.utils.reward_score.reward_utils import normalize_video_tensor, video_tensor_to_pil_frames

    if solution_image is None:
        raise ValueError("Ref fidelity reward requires generated video/image in solution_image.")
    solution_image = torch.as_tensor(solution_image)
    if solution_image.ndim == 3:
        video = solution_image.unsqueeze(0)
    else:
        video = solution_image
    video = normalize_video_tensor(video)

    frame_count = video.shape[0]
    sample_count = max(1, min(num_frames, frame_count))
    indices = torch.linspace(0, frame_count - 1, sample_count).round().long()
    return video_tensor_to_pil_frames(video[indices])


def compute_score_ref_fidelity(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    device: str | None = None,
    model_name_or_path: str = _DEFAULT_MODEL,
    num_frames: int = _DEFAULT_NUM_FRAMES,
    **kwargs,
) -> dict:
    """Compute CLIP cosine similarity between Ref2VA reference image(s) and generated frames."""
    del data_source, ground_truth, kwargs
    device = device or get_device_name()

    reference_images = _reference_images(extra_info)
    generated_frames = _sample_generated_frames(solution_image, num_frames)

    with _MODEL_LOCK, torch.no_grad():
        model, processor = _load_clip(model_name_or_path, device)
        reference_inputs = processor(images=reference_images, return_tensors="pt").to(device)
        generated_inputs = processor(images=generated_frames, return_tensors="pt").to(device)
        reference_embeds = F.normalize(model.get_image_features(**reference_inputs), p=2, dim=-1)
        generated_embeds = F.normalize(model.get_image_features(**generated_inputs), p=2, dim=-1)
        reference_embed = F.normalize(reference_embeds.mean(dim=0), p=2, dim=-1)
        generated_embed = F.normalize(generated_embeds.mean(dim=0), p=2, dim=-1)
        similarity = (reference_embed * generated_embed).sum().float().item()

    return {
        "score": similarity,
        "ref_fidelity_num_references": len(reference_images),
        "ref_fidelity_num_frames": len(generated_frames),
    }
