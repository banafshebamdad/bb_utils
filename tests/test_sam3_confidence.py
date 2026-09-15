import numpy as np
import pytest

from bb_utils.data_preparation import segmentation_runner
from bb_utils.segmentation.sam3_backend import Sam3Backend
from bb_utils.segmentation.utils import rotate_confidence


torch = pytest.importorskip("torch")


def _backend_with_states(states):
    backend = object.__new__(Sam3Backend)
    backend._infer_prompt_states = lambda image, target_classes: iter(states)
    return backend


class _SoftBackend:
    supports_soft_confidence = True

    def __init__(self, output):
        self.output = output

    def segment_confidence(self, image, target_classes):
        return self.output


class _BinaryBackend:
    def segment(self, image, target_classes):
        return np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)


def test_segment_confidence_no_detections_returns_zero_float32_map():
    backend = _backend_with_states([
        ("person", {
            "scores": torch.empty(0),
            "masks_logits": torch.empty((0, 1, 2, 3)),
        })
    ])

    output = backend.segment_confidence(np.zeros((2, 3, 3), dtype=np.uint8), [0])

    assert output.shape == (2, 3)
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output, np.zeros((2, 3), dtype=np.float32))


def test_segment_confidence_weights_probabilities_without_second_sigmoid():
    mask_probabilities = torch.tensor([[[[0.2, 0.8], [0.4, 1.0]]]])
    backend = _backend_with_states([
        ("person", {"scores": torch.tensor([0.4]), "masks_logits": mask_probabilities})
    ])

    output = backend.segment_confidence(np.zeros((2, 2, 3), dtype=np.uint8), [0])

    np.testing.assert_allclose(
        output,
        np.array([[0.08, 0.32], [0.16, 0.4]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_segment_confidence_uses_pixelwise_maximum_and_validates_range():
    backend = _backend_with_states([
        ("person", {
            "scores": torch.tensor([0.9, 0.5]),
            "masks_logits": torch.tensor([[
                [[0.2, 0.9], [0.8, 0.1]],
            ], [
                [[0.9, 0.4], [0.3, 1.0]],
            ]]),
        })
    ])

    output = backend.segment_confidence(np.zeros((2, 2, 3), dtype=np.uint8), [0])

    np.testing.assert_allclose(
        output,
        np.array([[0.45, 0.81], [0.72, 0.5]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.isfinite(output).all()
    assert np.all((0.0 <= output) & (output <= 1.0))


def test_segment_confidence_rejects_non_probability_masks_logits():
    backend = _backend_with_states([
        ("person", {
            "scores": torch.tensor([0.5]),
            "masks_logits": torch.tensor([[[[1.1]]]]),
        })
    ])

    with pytest.raises(RuntimeError, match="second sigmoid"):
        backend.segment_confidence(np.zeros((1, 1, 3), dtype=np.uint8), [0])


def test_soft_mode_writes_single_float32_confidence_key(tmp_path, monkeypatch):
    source = tmp_path / "frame.png"
    source.touch()
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(segmentation_runner, "_load_as_rgb", lambda path: image)
    output_path = tmp_path / "frame.npz"

    segmentation_runner.segment_frame(
        source, output_path, _SoftBackend(np.full((2, 3), 0.25, dtype=np.float32)),
        [0], dilation_radius=0, output_mode="soft_confidence",
    )

    with np.load(output_path, allow_pickle=False) as data:
        assert data.files == ["pedestrian_confidence"]
        assert data["pedestrian_confidence"].dtype == np.float32
        np.testing.assert_allclose(data["pedestrian_confidence"], 0.25)


def test_binary_mode_preserves_mask_key_and_uint8_output(tmp_path, monkeypatch):
    source = tmp_path / "frame.png"
    source.touch()
    monkeypatch.setattr(
        segmentation_runner, "_load_as_rgb", lambda path: np.zeros((2, 3, 3), dtype=np.uint8)
    )
    output_path = tmp_path / "frame.npz"

    segmentation_runner.segment_frame(
        source, output_path, _BinaryBackend(), [0], dilation_radius=0,
    )

    with np.load(output_path, allow_pickle=False) as data:
        assert data.files == ["mask"]
        assert data["mask"].dtype == np.uint8
        assert set(np.unique(data["mask"])).issubset({0, 1})


def test_soft_mode_rejects_dilation_and_unsupported_backend(tmp_path):
    source = tmp_path / "frame.png"
    source.touch()

    with pytest.raises(ValueError, match="mask_dilation_px"):
        segmentation_runner.segment_frame(
            source, tmp_path / "dilated.npz", _SoftBackend(np.zeros((2, 3))), [0],
            dilation_radius=1, output_mode="soft_confidence",
        )
    with pytest.raises(RuntimeError, match="does not support soft_confidence"):
        segmentation_runner.segment_frame(
            source, tmp_path / "unsupported.npz", _BinaryBackend(), [0],
            dilation_radius=0, output_mode="soft_confidence",
        )


def test_soft_mode_rotation_back_preserves_source_alignment(tmp_path, monkeypatch):
    source = tmp_path / "sequence_L_1.png"
    source.touch()
    source_image = np.zeros((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(segmentation_runner, "_load_as_rgb", lambda path: source_image)
    rotated_confidence = np.arange(6, dtype=np.float32).reshape(3, 2) / 10
    output_path = tmp_path / "frame.npz"

    segmentation_runner.segment_frame(
        source, output_path, _SoftBackend(rotated_confidence), [0], dilation_radius=0,
        preprocessing_cfg={"pre_rotation_deg": 90}, rotate_mask_back=True,
        output_mode="soft_confidence",
    )

    with np.load(output_path, allow_pickle=False) as data:
        confidence = data["pedestrian_confidence"]
    assert confidence.shape == source_image.shape[:2]
    np.testing.assert_allclose(confidence, rotate_confidence(rotated_confidence, -90))