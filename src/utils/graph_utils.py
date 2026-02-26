from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GraphStaticData:
    adjacency: np.ndarray
    normalized_adjacency: np.ndarray
    neighbors: List[List[int]]
    dist_to_end: np.ndarray
    dist_to_mine: np.ndarray


def build_adjacency(num_nodes: int, edges_1_based: Sequence[Tuple[int, int]]) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for a, b in edges_1_based:
        i = int(a) - 1
        j = int(b) - 1
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    return adjacency


def build_neighbors(adjacency: np.ndarray) -> List[List[int]]:
    return [np.where(adjacency[i] > 0)[0].astype(int).tolist() for i in range(adjacency.shape[0])]


def shortest_distances(adjacency: np.ndarray, target: int) -> np.ndarray:
    n = adjacency.shape[0]
    dist = np.full(n, np.inf, dtype=np.float32)
    dist[target] = 0.0
    queue: List[int] = [target]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for nxt in np.where(adjacency[node] > 0)[0].astype(int).tolist():
            if np.isinf(dist[nxt]):
                dist[nxt] = dist[node] + 1.0
                queue.append(nxt)
    unreachable = np.isinf(dist)
    if np.any(unreachable):
        max_finite = float(np.max(dist[~unreachable])) if np.any(~unreachable) else float(n)
        dist[unreachable] = max_finite + 1.0
    return dist


def min_distance_to_set(adjacency: np.ndarray, targets: Iterable[int]) -> np.ndarray:
    targets = list(targets)
    if not targets:
        return np.zeros(adjacency.shape[0], dtype=np.float32)
    all_dist = [shortest_distances(adjacency, t) for t in targets]
    stacked = np.stack(all_dist, axis=0)
    return np.min(stacked, axis=0)


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    a_hat = adjacency + np.eye(n, dtype=np.float32)
    degree = np.sum(a_hat, axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.clip(degree, 1e-8, None))
    return (a_hat * inv_sqrt[:, None]) * inv_sqrt[None, :]


def build_graph_static_data(
    num_nodes: int,
    edges_1_based: Sequence[Tuple[int, int]],
    end_node: int,
    mines: Sequence[int],
) -> GraphStaticData:
    adjacency = build_adjacency(num_nodes, edges_1_based)
    neighbors = build_neighbors(adjacency)
    dist_to_end = shortest_distances(adjacency, end_node)
    dist_to_mine = min_distance_to_set(adjacency, mines)
    norm_adj = normalize_adjacency(adjacency)
    return GraphStaticData(
        adjacency=adjacency,
        normalized_adjacency=norm_adj,
        neighbors=neighbors,
        dist_to_end=dist_to_end,
        dist_to_mine=dist_to_mine,
    )
