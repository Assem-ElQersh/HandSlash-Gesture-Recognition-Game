"""Tests for the processing pipeline: resampling and normalization."""

import numpy as np
import pytest
from processing.resampling import resample, path_length
from processing.normalization import normalize, translate_to_origin, scale_to_unit_box, pca_align
from processing.features import extract_features, compute_velocity


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

def make_line(n=30):
    """Straight horizontal line from (0,0) to (100,0)."""
    x = np.linspace(0, 100, n)
    y = np.zeros(n)
    return np.column_stack([x, y]).astype(np.float32)


def make_circle(n=50):
    """Unit circle trajectory."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)]).astype(np.float32)


# ------------------------------------------------------------------ #
#  Resampling                                                          #
# ------------------------------------------------------------------ #

class TestResampling:
    def test_output_shape(self):
        pts = make_line(30)
        out = resample(pts, n=64)
        assert out.shape == (64, 2), f"Expected (64,2), got {out.shape}"

    def test_output_dtype(self):
        out = resample(make_line(), n=64)
        assert out.dtype == np.float32

    def test_preserves_endpoints_approx(self):
        pts = make_line(30)
        out = resample(pts, n=64)
        assert out[0, 0] == pytest.approx(pts[0, 0], abs=1e-3)
        assert out[-1, 0] == pytest.approx(pts[-1, 0], abs=1e-3)

    def test_even_spacing(self):
        pts = make_line(30)
        out = resample(pts, n=64)
        dists = np.hypot(np.diff(out[:, 0]), np.diff(out[:, 1]))
        assert np.std(dists) < 1e-4, "Resampled points are not evenly spaced"

    def test_single_point_input(self):
        pts = np.array([[5.0, 10.0]])
        out = resample(pts, n=64)
        assert out.shape == (64, 2)
        assert np.allclose(out, [[5.0, 10.0]])

    def test_custom_n(self):
        for n in [16, 32, 64, 128]:
            out = resample(make_line(20), n=n)
            assert out.shape == (n, 2)

    def test_path_length(self):
        pts = make_line(100)
        length = path_length(pts)
        assert length == pytest.approx(100.0, rel=0.01)


# ------------------------------------------------------------------ #
#  Normalization                                                       #
# ------------------------------------------------------------------ #

class TestNormalization:
    def test_translate_centroid_at_origin(self):
        pts = make_line(64)
        out = translate_to_origin(pts)
        assert out.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-5)

    def test_scale_unit_box(self):
        pts = translate_to_origin(make_line(64))
        out = scale_to_unit_box(pts)
        assert np.abs(out).max() <= 1.0 + 1e-6

    def test_normalize_output_shape(self):
        pts = make_line(64)
        out = normalize(pts)
        assert out.shape == pts.shape

    def test_normalize_dtype(self):
        out = normalize(make_line(64))
        assert out.dtype == np.float32

    def test_normalize_centred(self):
        pts = make_line(64)
        out = normalize(pts)
        assert out.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-5)

    def test_normalize_bounded(self):
        pts = make_line(64)
        out = normalize(pts)
        assert np.abs(out).max() <= 1.0 + 1e-6

    def test_scale_invariance(self):
        pts = make_line(64)
        out1 = normalize(pts)
        out2 = normalize(pts * 100)
        assert np.allclose(out1, out2, atol=1e-5)

    def test_translation_invariance(self):
        pts = make_line(64)
        out1 = normalize(pts)
        out2 = normalize(pts + np.array([500, 300]))
        assert np.allclose(out1, out2, atol=1e-5)

    def test_pca_align_output_shape(self):
        pts = translate_to_origin(make_line(64))
        out = pca_align(pts)
        assert out.shape == (64, 2)

    def test_pca_align_centred(self):
        pts = translate_to_origin(make_line(64))
        out = pca_align(pts)
        assert out.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-4)


# ------------------------------------------------------------------ #
#  Feature extraction                                                  #
# ------------------------------------------------------------------ #

class TestFeatures:
    def test_extract_features_shape(self):
        pts = normalize(resample(make_line(30), 64))
        feats = extract_features(pts)
        assert feats.shape == (64, 4)

    def test_extract_features_dtype(self):
        pts = normalize(resample(make_line(30), 64))
        feats = extract_features(pts)
        assert feats.dtype == np.float32

    def test_velocity_first_point_zero(self):
        pts = make_line(64)
        vel = compute_velocity(pts)
        assert vel[0, 0] == pytest.approx(0.0)
        assert vel[0, 1] == pytest.approx(0.0)

    def test_velocity_shape(self):
        pts = make_line(64)
        vel = compute_velocity(pts)
        assert vel.shape == (64, 2)

    def test_xy_columns_preserved(self):
        pts = normalize(resample(make_line(30), 64))
        feats = extract_features(pts)
        assert np.allclose(feats[:, :2], pts)
