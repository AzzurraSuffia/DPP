import networkx as nx

def top_k_overlap(G: nx.Graph, G_anon: nx.Graph, query_fn, k: int = 10):
    v1 = query_fn(G)
    v2 = query_fn(G_anon)

    top1 = set(sorted(v1, key=v1.get, reverse=True)[:k])
    top2 = set(sorted(v2, key=v2.get, reverse=True)[:k])

    # fraction of elements in common
    return len(top1 & top2) / k

def scalar_error(G: nx.Graph, G_anon: nx.Graph, query_fn, error_type: str = "relative"):

    q_orig = query_fn(G)
    q_anon = query_fn(G_anon)

    if error_type == "absolute":
        return abs(q_orig - q_anon)

    if error_type == "relative":
        if q_orig == 0:
            return 0.0
        return abs(q_orig - q_anon) / abs(q_orig)

    raise ValueError("error_type must be 'absolute' or 'relative'")
