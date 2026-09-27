from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from pi05_temporal_metrics import temporal_stratified_metrics
from train_pi05_lerobot_adapter import split_by_episode


class FakeDataset:
    def __init__(self, episodes: list[int]) -> None:
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"episode_index": torch.tensor(self.episodes[index], dtype=torch.long)}


def main() -> None:
    dataset = FakeDataset([0, 0, 1, 1, 2, 2, 3, 3])
    manifest = {
        "schema": "project_2026_lerobot_export_v0",
        "frames": 8,
        "episodes": 4,
        "train_episode_indices": [0, 2, 3],
        "val_episode_indices": [1],
    }
    with tempfile.TemporaryDirectory(prefix="pi05-split-smoke-") as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        train, val, mode = split_by_episode(
            dataset,
            val_fraction=0.15,
            seed=123,
            max_records=None,
            split_manifest=manifest_path,
        )
        assert train == [0, 1, 4, 5, 6, 7]
        assert val == [2, 3]
        assert mode == "export_manifest_episode_split"
        try:
            split_by_episode(
                dataset,
                val_fraction=0.15,
                seed=123,
                max_records=4,
                split_manifest=manifest_path,
            )
        except ValueError as exc:
            assert "do not pass --max-records" in str(exc)
        else:
            raise AssertionError("split manifest must reject max_records")

        manifest.update(train_episode_indices=[0, 1, 2, 3], val_episode_indices=[],
                        training_use={"scope": "user_authorized_real10_train_only_prototype"})
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        kwargs = dict(val_fraction=.15, seed=123, max_records=None, split_manifest=manifest_path)
        train, val, mode = split_by_episode(dataset, **kwargs, train_only=True)
        assert train == list(range(8)) and val == [] and mode == "real10_export_train_only"
        try:
            split_by_episode(dataset, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("empty validation must still require explicit train-only")
        manifest["training_use"]["scope"] = "some_other_dataset"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            split_by_episode(dataset, **kwargs, train_only=True)
        except ValueError:
            pass
        else:
            raise AssertionError("train-only must not silently waive other dataset splits")

    metrics = temporal_stratified_metrics(
        episode_indices=torch.tensor([0, 0, 0, 1, 1, 1]),
        frame_indices=torch.tensor([0, 1, 2, 0, 1, 2]),
        piper_target=torch.tensor([1, 1, 2, 2, 1, 1]),
        piper_pred=torch.tensor([1, 2, 2, 2, 1, 2]),
        elite_translation_target=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        elite_translation_pred=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
    )
    assert metrics["first_frame_records"] == 2
    assert metrics["eligible_temporal_records"] == 4
    assert metrics["transition_records"] == 2
    assert metrics["steady_records"] == 2
    assert metrics["model_piper_transition"]["accuracy"] == 1.0
    assert metrics["model_piper_steady"]["accuracy"] == 0.0
    assert metrics["previous_piper_baseline"]["accuracy"] == 0.5
    assert metrics["previous_elite_translation_baseline"]["records"] == 4
    print("pi05_temporal_split_smoke_ok")


if __name__ == "__main__":
    main()
