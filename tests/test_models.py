"""Tests for model interface compliance and basic predict shapes."""

import numpy as np
import pytest
import tempfile
import os


GESTURES = ["slash_left", "slash_right", "circle"]
N_CLASSES = len(GESTURES)
N_TRAIN = 30
N_TEST = 6
SEQ_LEN = 64
FEAT_DIM = 4


def _make_data(n):
    """Synthetic (n, 64, 4) float32 dataset with random labels."""
    X = np.random.randn(n, SEQ_LEN, FEAT_DIM).astype(np.float32)
    y = np.random.randint(0, N_CLASSES, n)
    return X, y


def _fit_and_check(model, X_train, y_train, X_test):
    """Fit model and verify output shapes/types."""
    model.fit(X_train, y_train)
    assert model._is_fitted

    preds = model.predict(X_test)
    assert preds.shape == (N_TEST,), f"predict shape mismatch: {preds.shape}"
    assert preds.dtype in (np.int32, np.int64, int), f"bad dtype: {preds.dtype}"
    assert all(0 <= p < N_CLASSES for p in preds)

    proba = model.predict_proba(X_test)
    assert proba.shape == (N_TEST, N_CLASSES), f"proba shape mismatch: {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4), "probabilities must sum to 1"

    # Single sample (2D input)
    single_pred = model.predict(X_test[0])
    assert single_pred.shape == (1,)

    label = model.predict_label(X_test[0])
    assert label in GESTURES


def _check_save_load(model, X_test, factory_fn):
    """Save, reload, and check predictions match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        model.save(path)

        loaded = factory_fn()
        loaded.load(path)
        assert loaded._is_fitted

        orig_preds = model.predict(X_test)
        load_preds = loaded.predict(X_test)
        assert np.array_equal(orig_preds, load_preds), "predictions differ after reload"


# ------------------------------------------------------------------ #
#  DTW                                                                 #
# ------------------------------------------------------------------ #

class TestDTW:
    def test_fit_predict(self):
        from models.dtw import DTWClassifier
        X_train, y_train = _make_data(N_TRAIN)
        X_test, _ = _make_data(N_TEST)
        model = DTWClassifier(GESTURES)
        _fit_and_check(model, X_train, y_train, X_test)

    def test_save_load(self):
        from models.dtw import DTWClassifier
        X_train, y_train = _make_data(N_TRAIN)
        X_test, _ = _make_data(N_TEST)
        model = DTWClassifier(GESTURES)
        model.fit(X_train, y_train)
        _check_save_load(model, X_test, lambda: DTWClassifier(GESTURES))


# ------------------------------------------------------------------ #
#  HMM                                                                 #
# ------------------------------------------------------------------ #

class TestHMM:
    def test_fit_predict(self):
        pytest.importorskip("hmmlearn")
        from models.hmm import HMMClassifier
        X_train, y_train = _make_data(N_TRAIN)
        X_test, _ = _make_data(N_TEST)
        model = HMMClassifier(GESTURES, n_states=3, n_iter=10)
        _fit_and_check(model, X_train, y_train, X_test)

    def test_save_load(self):
        pytest.importorskip("hmmlearn")
        from models.hmm import HMMClassifier
        X_train, y_train = _make_data(N_TRAIN)
        X_test, _ = _make_data(N_TEST)
        model = HMMClassifier(GESTURES, n_states=3, n_iter=5)
        model.fit(X_train, y_train)
        _check_save_load(model, X_test, lambda: HMMClassifier(GESTURES))


# ------------------------------------------------------------------ #
#  CNN                                                                 #
# ------------------------------------------------------------------ #

class TestCNN:
    def test_fit_predict(self):
        pytest.importorskip("torch")
        from models.cnn import CNNClassifier
        X_train, y_train = _make_data(N_TRAIN)
        X_test, _ = _make_data(N_TEST)
        model = CNNClassifier(GESTURES, channels=(16, 32), dropout=0.0)
        model.fit(X_train, y_train, epochs=2, batch_size=8)
        _fit_and_check(model, X_train, y_train, X_test)

    def test_save_load(self):
        pytest.importorskip("torch")
        from models.cnn import CNNClassifier
        X_train, y_train = _make_data(N_TRAIN)
        X_test, _ = _make_data(N_TEST)
        model = CNNClassifier(GESTURES, channels=(16, 32))
        model.fit(X_train, y_train, epochs=2, batch_size=8)
        _check_save_load(model, X_test, lambda: CNNClassifier(GESTURES, channels=(16, 32)))


# ------------------------------------------------------------------ #
#  LSTM                                                                #
# ------------------------------------------------------------------ #

class TestLSTM:
    def test_fit_predict(self):
        pytest.importorskip("torch")
        from models.lstm import LSTMClassifier
        X_train, y_train = _make_data(N_TRAIN)
        X_test, _ = _make_data(N_TEST)
        model = LSTMClassifier(GESTURES, hidden_size=16, num_layers=1, dropout=0.0)
        model.fit(X_train, y_train, epochs=2, batch_size=8)
        _fit_and_check(model, X_train, y_train, X_test)

    def test_save_load(self):
        pytest.importorskip("torch")
        from models.lstm import LSTMClassifier
        X_train, y_train = _make_data(N_TRAIN)
        X_test, _ = _make_data(N_TEST)
        model = LSTMClassifier(GESTURES, hidden_size=16, num_layers=1)
        model.fit(X_train, y_train, epochs=2, batch_size=8)
        _check_save_load(
            model, X_test,
            lambda: LSTMClassifier(GESTURES, hidden_size=16, num_layers=1),
        )


# ------------------------------------------------------------------ #
#  Transformer                                                         #
# ------------------------------------------------------------------ #

class TestTransformer:
    def test_fit_predict(self):
        pytest.importorskip("torch")
        from models.transformer import TransformerClassifier
        X_train, y_train = _make_data(N_TRAIN)
        X_test, _ = _make_data(N_TEST)
        model = TransformerClassifier(
            GESTURES, d_model=16, nhead=2, num_encoder_layers=1,
            dim_feedforward=32, dropout=0.0,
        )
        model.fit(X_train, y_train, epochs=2, batch_size=8)
        _fit_and_check(model, X_train, y_train, X_test)

    def test_save_load(self):
        pytest.importorskip("torch")
        from models.transformer import TransformerClassifier
        X_train, y_train = _make_data(N_TRAIN)
        X_test, _ = _make_data(N_TEST)
        model = TransformerClassifier(
            GESTURES, d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32,
        )
        model.fit(X_train, y_train, epochs=2, batch_size=8)
        _check_save_load(
            model, X_test,
            lambda: TransformerClassifier(
                GESTURES, d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32,
            ),
        )
