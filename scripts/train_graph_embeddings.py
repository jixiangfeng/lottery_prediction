#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快乐8 号码共现图嵌入训练脚本
----------------------------
基于随机游走 + Skip-gram（负采样）的 Node2Vec 简化版本，输出供特征增强使用的嵌入缓存。
支持 CPU / GPU（NVIDIA / AMD ROCm）自动选择，可通过 `--device` 手动指定。
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GRAPH_EMBED_CONFIG, PATHS
from src.analysis.shared_download import ensure_data_available

Number = int


def _load_draws(file_path: Path, limit: int | None = None) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(file_path)
    data = df.to_numpy(dtype=int)
    if limit is not None:
        return data[:limit]
    return data


def _extract_numbers(row: Sequence[int]) -> List[int]:
    return [int(n) for n in row[1:] if int(n) > 0]


def _build_cooccurrence_graph(draws: np.ndarray, decay: float) -> np.ndarray:
    adjacency = np.zeros((80, 80), dtype=float)
    for idx, row in enumerate(draws):
        numbers = [n for n in _extract_numbers(row) if 1 <= n <= 80]
        if len(numbers) < 2:
            continue
        weight = decay**idx
        for i in numbers:
            for j in numbers:
                if i != j:
                    adjacency[i - 1, j - 1] += weight
    return adjacency


def _random_walk(adjacency: np.ndarray, start: int, walk_length: int, rng: random.Random) -> List[int]:
    node = start
    walk = [node]
    for _ in range(walk_length - 1):
        weights = adjacency[node - 1]
        if weights.sum() <= 0:
            node = rng.randint(1, 80)
        else:
            probs = weights / weights.sum()
            node = rng.choices(range(1, 81), weights=probs, k=1)[0]
        walk.append(node)
    return walk


def _generate_walks(
    adjacency: np.ndarray,
    num_walks: int,
    walk_length: int,
    seed: int,
) -> List[List[int]]:
    rng = random.Random(seed)
    walks: List[List[int]] = []
    nodes = list(range(1, 81))
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walks.append(_random_walk(adjacency, node, walk_length, rng))
    return walks


class SkipGramDataset(Dataset[Tuple[int, int]]):
    def __init__(self, walks: Sequence[Sequence[int]], window_size: int):
        pairs: List[Tuple[int, int]] = []
        for walk in walks:
            for idx, center in enumerate(walk):
                left = max(0, idx - window_size)
                right = min(len(walk), idx + window_size + 1)
                for context_idx in range(left, right):
                    if context_idx == idx:
                        continue
                    context = walk[context_idx]
                    pairs.append((center - 1, context - 1))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        return self.pairs[index]


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, centers: torch.Tensor, contexts: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        center_vec = self.embeddings(centers)
        context_vec = self.embeddings(contexts)
        positive_score = torch.sum(center_vec * context_vec, dim=1)
        positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-9).mean()

        negative_vec = self.embeddings(negatives)
        negative_score = torch.bmm(negative_vec, center_vec.unsqueeze(2)).squeeze(2)
        negative_loss = -torch.log(torch.sigmoid(-negative_score) + 1e-9).sum(dim=1).mean()
        return positive_loss + negative_loss


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练快乐8号码共现图嵌入（Node2Vec 简化版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lottery", default="kl8", help="彩票代号")
    parser.add_argument("--embedding-dim", type=int, default=int(GRAPH_EMBED_CONFIG.get("embedding_dim", 32)))
    parser.add_argument("--walk-length", type=int, default=16, help="随机游走长度")
    parser.add_argument("--num-walks", type=int, default=20, help="每个节点随机游走次数")
    parser.add_argument("--window-size", type=int, default=4, help="Skip-gram 窗口大小")
    parser.add_argument("--epochs", type=int, default=25, help="训练轮次")
    parser.add_argument("--batch-size", type=int, default=512, help="训练批大小")
    parser.add_argument("--negative-samples", type=int, default=5, help="每个正样本的负样本数")
    parser.add_argument("--learning-rate", type=float, default=0.025, help="学习率")
    parser.add_argument("--decay", type=float, default=float(GRAPH_EMBED_CONFIG.get("decay", 0.92)), help="历史期权重衰减")
    parser.add_argument("--max-samples", type=int, default=0, help="限制使用的历史期数（0 表示不限）")
    parser.add_argument("--device", default="auto", help="计算设备（auto/cpu/cuda/mps/xpu 等）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", default=GRAPH_EMBED_CONFIG.get("cache_file"), help="嵌入输出路径（npz）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ensure_data_available(name=args.lottery, download_flag=1)
    data_path = PATHS["data"] / args.lottery / "data.csv"
    draws = _load_draws(data_path, limit=args.max_samples if args.max_samples > 0 else None)
    adjacency = _build_cooccurrence_graph(draws, decay=args.decay)
    walks = _generate_walks(adjacency, num_walks=args.num_walks, walk_length=args.walk_length, seed=args.seed)

    dataset = SkipGramDataset(walks, window_size=args.window_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = _resolve_device(args.device)
    model = SkipGramModel(vocab_size=80, embedding_dim=args.embedding_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    negatives_distribution = np.power(adjacency.sum(axis=1) + 1.0, 0.75)
    negatives_distribution = negatives_distribution / negatives_distribution.sum()
    negatives_alias = np.cumsum(negatives_distribution)

    def sample_negatives(batch_size: int, k: int) -> torch.Tensor:
        values = np.searchsorted(negatives_alias, np.random.rand(batch_size, k))
        return torch.from_numpy(values.astype(np.int64)).to(device)

    start_ts = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for centers, contexts in dataloader:
            centers = centers.to(device)
            contexts = contexts.to(device)
            negatives = sample_negatives(len(centers), args.negative_samples)
            optimizer.zero_grad()
            loss = model(centers, contexts, negatives)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"[Epoch {epoch}/{args.epochs}] 平均损失 {avg_loss:.4f}")

    embeddings = model.embeddings.weight.detach().cpu().numpy()
    output_path = Path(args.output or GRAPH_EMBED_CONFIG.get("cache_file", PATHS["data_cache"] / "graph_embeddings.npz"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "embedding_dim": args.embedding_dim,
        "walk_length": args.walk_length,
        "num_walks": args.num_walks,
        "window_size": args.window_size,
        "epochs": args.epochs,
        "negative_samples": args.negative_samples,
        "learning_rate": args.learning_rate,
        "decay": args.decay,
        "seed": args.seed,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(dataset),
        "duration_seconds": round(time.time() - start_ts, 2),
    }
    np.savez(output_path, embeddings=embeddings, metadata=metadata)
    print(f"嵌入训练完成，已写入 {output_path}")


if __name__ == "__main__":
    main()
