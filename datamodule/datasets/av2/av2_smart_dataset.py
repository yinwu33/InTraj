from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

from .smart_av2_preprocess import process_single_scenario
from .smart_token_processor import TokenProcessor


class AV2SMARTDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        raw_dir: str | None = None,
        processed_dir: str | None = None,
        transform: Callable | None = None,
        token_size: int = 2048,
        agent_token_path: str | None = None,
        map_token_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = os.path.expanduser(os.path.normpath(root))
        self.split = split
        self.raw_dir = os.path.expanduser(os.path.normpath(raw_dir or os.path.join(self.root, split)))
        self.processed_dir = os.path.expanduser(
            os.path.normpath(processed_dir or os.path.join(self.root, "cache", "av2_smart", split))
        )
        self.transform = transform
        self.token_processor = TokenProcessor(
            token_size=token_size,
            agent_token_path=agent_token_path,
            map_token_path=map_token_path,
        )

        os.makedirs(self.processed_dir, exist_ok=True)
        if os.path.isdir(self.raw_dir):
            self.scenario_ids = sorted(
                name for name in os.listdir(self.raw_dir) if os.path.isdir(os.path.join(self.raw_dir, name))
            )
        else:
            self.scenario_ids = sorted(Path(self.processed_dir).glob("*.pkl"))
            self.scenario_ids = [path.stem for path in self.scenario_ids]

    def __len__(self) -> int:
        return len(self.scenario_ids)

    def _processed_path(self, scenario_id: str) -> Path:
        return Path(self.processed_dir) / f"{scenario_id}.pkl"

    def _ensure_processed(self, scenario_id: str) -> Path:
        processed_path = self._processed_path(scenario_id)
        if processed_path.is_file():
            return processed_path

        log_dir = Path(self.raw_dir) / scenario_id
        if not log_dir.is_dir():
            raise FileNotFoundError(f"Missing AV2 SMART raw scenario dir: {log_dir}")
        process_single_scenario(log_dir=log_dir, output_dir=Path(self.processed_dir))
        if not processed_path.is_file():
            raise FileNotFoundError(f"SMART preprocessing did not create {processed_path}")
        return processed_path

    def __getitem__(self, idx: int):
        scenario_id = self.scenario_ids[idx]
        processed_path = self._ensure_processed(scenario_id)
        with open(processed_path, "rb") as handle:
            data = pickle.load(handle)
        data = self.token_processor.preprocess(data)
        if self.transform is not None:
            data = self.transform(data)
        return data

