from collections import deque
from store import TripleStore


def _normalise(term: str) -> str:
    return term.strip().lower().replace(" ", "_")


def bfs_query(store: TripleStore, p: str, q: str, max_depth: int = 5) -> dict:
    """
    Find all paths from p to q in the triple store up to max_depth.

    Polarity propagates multiplicatively along each path:
        pol(A=>B=>C) = pol(A,B) * pol(B,C)

    Bias score across all found paths:
        bias = Σ (path_polarity * min_weight_along_path) / n_paths

    Returns:
        {
            "known":      bool,
            "paths":      [ { "chain": [...], "polarity": int, "min_weight": float } ],
            "bias_score": float,   # None if not known
        }
    """
    p_key = _normalise(p)
    q_key = _normalise(q)

    if p_key == q_key:
        return {"known": True, "paths": [{"chain": [p_key], "polarity": 1, "min_weight": 1.0}], "bias_score": 1.0}

    # BFS — queue entries: (current_node, path_so_far, cumulative_polarity, min_weight_so_far)
    queue   = deque()
    queue.append((p_key, [p_key], 1, float("inf")))

    found_paths = []
    visited_at  = {p_key: 0}   # node -> earliest depth seen (allow revisit at deeper levels for multi-path)

    while queue:
        current, path, cum_pol, min_w = queue.popleft()
        depth = len(path) - 1

        if depth >= max_depth:
            continue

        for edge in store.get_edges(current):
            neighbour = edge["q"]
            edge_pol  = edge["polarity"]
            edge_w    = edge["weight"]

            new_pol   = cum_pol * edge_pol
            new_min_w = min(min_w, edge_w)
            new_path  = path + [neighbour]

            if neighbour == q_key:
                found_paths.append({
                    "chain":      new_path,
                    "polarity":   new_pol,
                    "min_weight": new_min_w,
                })
                continue   # don't extend past the target

            # only revisit a node if we're at a shallower depth than before
            if neighbour not in visited_at or visited_at[neighbour] > depth + 1:
                visited_at[neighbour] = depth + 1
                queue.append((neighbour, new_path, new_pol, new_min_w))

    if not found_paths:
        return {"known": False, "paths": [], "bias_score": None}

    bias = sum(fp["polarity"] * fp["min_weight"] for fp in found_paths) / len(found_paths)

    return {
        "known":      True,
        "paths":      found_paths,
        "bias_score": round(bias, 4),
    }
