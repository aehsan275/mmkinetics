import matplotlib.pyplot as plt
import numpy as np
import pytest

from mm_kinetics.evaluation import plots


@pytest.mark.parametrize("confidence,label", [(0.9, "90%"), (0.95, "95%"), (0.925, "92.5%")])
@pytest.mark.parametrize("unbounded", [False, True])
def test_prediction_labels_and_unbounded_bands(tmp_path, monkeypatch, confidence, label, unbounded):
    captured = {}
    save = plots._save

    def inspect_and_save(fig, path):
        ax = fig.axes[0]
        captured["labels"] = ax.get_legend_handles_labels()[1]
        captured["notes"] = [text.get_text() for text in ax.texts]
        captured["collections"] = len(ax.collections)
        captured["limits"] = ax.get_ylim()
        return save(fig, path)

    monkeypatch.setattr(plots, "_save", inspect_and_save)
    truth = np.array([1.0, 2.0, 3.0])
    lower = np.full(3, -np.inf) if unbounded else truth - 0.5
    upper = np.full(3, np.inf) if unbounded else truth + 0.5
    path = plots.plot_predictions(
        truth, truth, lower, upper, tmp_path / "predictions.png", confidence_level=confidence
    )
    assert path.exists() and path.stat().st_size > 0
    assert np.isfinite(captured["limits"]).all()
    if unbounded:
        assert captured["collections"] == 0
        assert label in captured["notes"][0] and "unbounded for 3/3" in captured["notes"][0]
    else:
        assert captured["collections"] == 1
        assert f"Calibrated {label} interval" in captured["labels"]
        assert not captured["notes"]


def test_mixed_bounds_and_default_confidence(tmp_path, monkeypatch):

    def inspect(fig, path):
        ax = fig.axes[0]
        assert "Calibrated 95% interval" in ax.get_legend_handles_labels()[1]
        assert "unbounded for 1/3" in ax.texts[0].get_text()
        vertices = ax.collections[0].get_paths()[0].vertices
        assert np.isfinite(vertices).all()
        plt.close(fig)
        return path

    monkeypatch.setattr(plots, "_save", inspect)
    truth = np.array([1.0, 2.0, 3.0])
    plots.plot_predictions(
        truth,
        truth,
        np.array([0.0, -np.inf, 2.0]),
        np.array([2.0, np.inf, 4.0]),
        tmp_path / "mixed.png",
    )
