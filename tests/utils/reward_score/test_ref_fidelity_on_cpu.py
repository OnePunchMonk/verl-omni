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
"""CPU tests for the Ref2VA reference-fidelity reward."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from PIL import Image


def _load_file(dotted_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stub_verl_omni_reward_utils(repo_root: Path) -> None:
    """Register ``verl_omni.utils.reward_score.reward_utils`` without importing
    ``verl_omni/__init__.py``, whose heavy pipeline registrations are unrelated
    to this scorer and needn't be installed to test it."""
    package_names = ("verl_omni", "verl_omni.utils", "verl_omni.utils.reward_score")
    for name in package_names:
        sys.modules.setdefault(name, ModuleType(name))
    reward_utils_path = repo_root / "verl_omni/utils/reward_score/reward_utils.py"
    sys.modules["verl_omni.utils.reward_score.reward_utils"] = _load_file(
        "verl_omni.utils.reward_score.reward_utils", reward_utils_path
    )


def _load_module():
    repo_root = Path(__file__).parents[3]
    _stub_verl_omni_reward_utils(repo_root)
    return _load_file("ref_fidelity", repo_root / "verl_omni/utils/reward_score/ref_fidelity.py")


ref_fidelity = _load_module()


class _FakeProcessor:
    """Maps each fixed-color PIL image to a distinct one-hot feature vector."""

    _COLOR_TO_FEATURE = {
        (255, 0, 0): [1.0, 0.0, 0.0],
        (0, 255, 0): [0.0, 1.0, 0.0],
        (0, 0, 255): [0.0, 0.0, 1.0],
    }

    def __call__(self, *, images, return_tensors="pt"):
        features = [self._COLOR_TO_FEATURE[image.getpixel((0, 0))] for image in images]
        return _FakeBatch(torch.tensor(features))


class _FakeBatch(dict):
    def __init__(self, pixel_values):
        super().__init__(pixel_values=pixel_values)

    def to(self, device):
        return self


class _FakeModel:
    def get_image_features(self, pixel_values):
        return pixel_values


def _solid_image(color: tuple[int, int, int], size: int = 4) -> Image.Image:
    return Image.new("RGB", (size, size), color)


def _solid_video(color: tuple[int, int, int], num_frames: int = 3, size: int = 4) -> torch.Tensor:
    frame = torch.tensor(color, dtype=torch.uint8).view(3, 1, 1).expand(3, size, size)
    return frame.unsqueeze(0).repeat(num_frames, 1, 1, 1)


def test_reference_images_requires_source_images():
    with pytest.raises(KeyError, match="source_images"):
        ref_fidelity._reference_images({})


def test_reference_images_accepts_single_path(tmp_path):
    image_path = tmp_path / "ref.png"
    _solid_image((255, 0, 0)).save(image_path)

    images = ref_fidelity._reference_images({"source_images": str(image_path)})

    assert len(images) == 1
    assert images[0].getpixel((0, 0)) == (255, 0, 0)


def test_sample_generated_frames_rejects_missing_solution():
    with pytest.raises(ValueError, match="requires generated video/image"):
        ref_fidelity._sample_generated_frames(None, num_frames=4)


def test_sample_generated_frames_caps_at_available_frame_count():
    video = _solid_video((0, 255, 0), num_frames=2)

    frames = ref_fidelity._sample_generated_frames(video, num_frames=8)

    assert len(frames) == 2


def test_compute_score_ref_fidelity_perfect_match(monkeypatch, tmp_path):
    image_path = tmp_path / "ref.png"
    _solid_image((0, 255, 0)).save(image_path)
    monkeypatch.setattr(ref_fidelity, "_load_clip", lambda model_name_or_path, device: (_FakeModel(), _FakeProcessor()))

    result = ref_fidelity.compute_score_ref_fidelity(
        data_source="minimax_h3_ref2va",
        solution_image=_solid_video((0, 255, 0)),
        ground_truth="a green scene",
        extra_info={"source_images": [str(image_path)]},
        device="cpu",
    )

    assert result["score"] == pytest.approx(1.0)
    assert result["ref_fidelity_num_references"] == 1
    assert result["ref_fidelity_num_frames"] == 3


def test_compute_score_ref_fidelity_orthogonal_mismatch(monkeypatch, tmp_path):
    image_path = tmp_path / "ref.png"
    _solid_image((255, 0, 0)).save(image_path)
    monkeypatch.setattr(ref_fidelity, "_load_clip", lambda model_name_or_path, device: (_FakeModel(), _FakeProcessor()))

    result = ref_fidelity.compute_score_ref_fidelity(
        data_source="minimax_h3_ref2va",
        solution_image=_solid_video((0, 0, 255)),
        ground_truth="a blue scene",
        extra_info={"source_images": [str(image_path)]},
        device="cpu",
    )

    assert result["score"] == pytest.approx(0.0, abs=1e-6)


def test_compute_score_ref_fidelity_averages_multiple_references(monkeypatch, tmp_path):
    red_path = tmp_path / "red.png"
    green_path = tmp_path / "green.png"
    _solid_image((255, 0, 0)).save(red_path)
    _solid_image((0, 255, 0)).save(green_path)
    monkeypatch.setattr(ref_fidelity, "_load_clip", lambda model_name_or_path, device: (_FakeModel(), _FakeProcessor()))

    result = ref_fidelity.compute_score_ref_fidelity(
        data_source="minimax_h3_ref2va",
        solution_image=_solid_video((255, 0, 0)),
        ground_truth="a red scene",
        extra_info={"source_images": [str(red_path), str(green_path)]},
        device="cpu",
    )

    assert result["ref_fidelity_num_references"] == 2
    assert 0.0 < result["score"] < 1.0
