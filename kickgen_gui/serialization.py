"""Save and load a Pipeline to/from JSON."""

from __future__ import annotations

import json
from typing import Any

from kickgen.channel import Channel
from kickgen.pipeline import Pipeline
from kickgen.registry import BLOCK_REGISTRY


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Serialize *pipeline* to a JSON file at *path*."""
    channels_data = []
    for ch_name, channel in pipeline.channels:
        blocks_data = []
        for blk_name, block in channel.blocks:
            # Determine block type name from the registry
            type_name = type(block).__name__
            blocks_data.append(
                {
                    "type": type_name,
                    "name": blk_name,
                    "params": block.get_params(),
                }
            )
        channels_data.append(
            {
                "name": ch_name,
                "pan": channel.pan,
                "gain_db": channel.gain_db,
                "blocks": blocks_data,
            }
        )

    data: dict[str, Any] = {
        "master_gain_db": pipeline.master_gain_db,
        "use_limiter": pipeline.use_limiter,
        "channels": channels_data,
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_pipeline(path: str) -> Pipeline:
    """Deserialize a Pipeline from the JSON file at *path*.

    Uses :data:`~kickgen.registry.BLOCK_REGISTRY` to reconstruct blocks by
    type name.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    channels: list[tuple[str, Channel]] = []
    for ch_data in data.get("channels", []):
        blocks: list[tuple[str, Any]] = []
        for blk_data in ch_data.get("blocks", []):
            type_name: str = blk_data["type"]
            blk_name: str = blk_data["name"]
            params: dict[str, float] = blk_data.get("params", {})

            if type_name not in BLOCK_REGISTRY:
                raise ValueError(
                    f"Unknown block type '{type_name}'. "
                    f"Available: {list(BLOCK_REGISTRY.keys())}"
                )
            block_cls = BLOCK_REGISTRY[type_name]
            block = block_cls()
            if params:
                block.set_params(**params)
            blocks.append((blk_name, block))

        channel = Channel(
            blocks,
            pan=float(ch_data.get("pan", 0.0)),
            gain_db=float(ch_data.get("gain_db", 0.0)),
        )
        channels.append((ch_data["name"], channel))

    pipeline = Pipeline(
        channels,
        master_gain_db=float(data.get("master_gain_db", 0.0)),
        use_limiter=bool(data.get("use_limiter", True)),
    )
    return pipeline
