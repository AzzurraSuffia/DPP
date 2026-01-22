import networkx as nx

def top_k_overlap(G: nx.Graph, G_anon: nx.Graph, query_fn, top_k: int = 10):
    v1 = query_fn(G)
    v2 = query_fn(G_anon)

    top1 = set(sorted(v1, key=v1.get, reverse=True)[:top_k])
    top2 = set(sorted(v2, key=v2.get, reverse=True)[:top_k])

    # fraction of elements in common
    return len(top1 & top2) / top_k

def scalar_relative_error(G: nx.Graph, G_anon: nx.Graph, query_fn):
    """
    Compute normalized error for additive-only changes (edges added, components merged).
    Returns a number in [0,1], higher = more distortion.
    """
    q_orig = query_fn(G)
    q_anon = query_fn(G_anon)
    n = G.number_of_nodes()
    
    if query_fn == nx.number_of_edges:
        q_max = n * (n - 1) // 2
        return (q_anon - q_orig) / (q_max - q_orig) if q_max != q_orig else 0.0

    elif query_fn == nx.number_connected_components:
        return (q_orig - q_anon) / (q_orig - 1) if q_orig != 1 else 0.0

    else:
        # fallback: use capped relative error
        return min(abs(q_anon - q_orig) / abs(q_orig), 1.0)

def avg_path_length_lcc(G):
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)

    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc)
    return nx.average_shortest_path_length(G_lcc)

def diameter_lcc(G):
    if nx.is_connected(G):
        return nx.diameter(G)

    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc)
    return nx.diameter(G_lcc)

def compute_perquery_errors(G: nx.Graph, G_anon: nx.Graph, top_k: int = 10):

    errors = {}

    # --- Node-level queries (top-k overlap) ---
    errors["degree_centrality"] = 1 - top_k_overlap(G, G_anon, nx.degree_centrality, top_k)

    errors["local_clustering"] = 1 - top_k_overlap(G, G_anon, nx.clustering, top_k)

    errors["betweenness_centrality"] = 1 - top_k_overlap(G, G_anon, nx.betweenness_centrality, top_k)

    errors["closeness_centrality"] = 1 - top_k_overlap(G, G_anon, nx.closeness_centrality, top_k)

    # --- Global scalar queries --
    errors["number_of_edges"] = scalar_relative_error(G, G_anon, nx.number_of_edges)

    return errors

def compute_queryability_score(query_errors: dict, weights: dict = None):
    if weights is None:
        weights = {q: 1.0 for q in query_errors}

    total_weight = sum(weights.values())
    score = sum(query_errors[q] * weights[q] for q in query_errors)
    score = score / total_weight
    return score