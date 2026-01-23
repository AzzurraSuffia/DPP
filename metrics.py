import networkx as nx
from label_domain import LabelDomain

def compute_global_loss(G_original: nx.Graph, G_anonymized: nx.Graph, label_domain: LabelDomain | None = None) -> float:
    """
    Compute the global information loss between an original graph and its anonymized version.
    """
    if G_original.number_of_nodes() != G_anonymized.number_of_nodes():
            raise ValueError("Graph size mismatch: anonymized graph must preserve number of nodes.")

    num_nodes = G_original.number_of_nodes()

    # 1. Structural Loss (Edges added)
    num_edges_orig = G_original.number_of_edges()
    num_edges_anon = G_anonymized.number_of_edges()
    
    added_edges = num_edges_anon - num_edges_orig
    max_edges = num_nodes * (num_nodes - 1) // 2
    max_added_edges = max_edges - num_edges_orig

    # 2. Label Loss (NCP)
    total_ncp = 0.0
    max_ncp = 0.0
    
    if label_domain is not None:
            if not all('label' in d for _, d in G_original.nodes(data=True)):
                    raise ValueError("Original graph has missing labels.")

            if not all('label' in d for _, d in G_anonymized.nodes(data=True)):
                    raise ValueError("Anonymized graph has missing labels.")

            for node in G_original.nodes():
                    generalized_label = G_anonymized.nodes[node]['label']
                    total_ncp += label_domain.normalized_certainty_penalty(generalized_label)

            max_ncp = num_nodes * label_domain.normalized_certainty_penalty(label_domain.root)

    total_loss = added_edges + total_ncp
    max_loss = max_added_edges + max_ncp

    return total_loss / max_loss

def top_k_overlap(G: nx.Graph, G_anon: nx.Graph, query_fn, top_k: int = 10):
    v1 = query_fn(G)
    v2 = query_fn(G_anon)

    top1 = set(sorted(v1, key=v1.get, reverse=True)[:top_k])
    top2 = set(sorted(v2, key=v2.get, reverse=True)[:top_k])

    # fraction of elements in common
    return len(top1 & top2) / top_k

def scalar_relative_error(G: nx.Graph, G_anon: nx.Graph, query_fn, q_max):
    q_orig = query_fn(G)
    q_anon = query_fn(G_anon)
    n = G.number_of_nodes()
    
    return (q_anon - q_orig) / (q_max - q_orig) if q_max != q_orig else 0.0

def compute_perquery_errors(G: nx.Graph, G_anon: nx.Graph, top_k: int = 10):
    """
    Compute per-query errors for several graph metrics:
    degree, clustering, betweenness, closeness, and number of edges.
    Returns a dictionary mapping each query to its error.
    """

    errors = {}
    n = G.number_of_nodes()
    q_max_edges = n * (n - 1) // 2

    # For queries returning vectors, use top-k overlap
    errors["degree_centrality"] = 1 - top_k_overlap(G, G_anon, nx.degree_centrality, top_k)

    errors["local_clustering"] = 1 - top_k_overlap(G, G_anon, nx.clustering, top_k)

    errors["betweenness_centrality"] = 1 - top_k_overlap(G, G_anon, nx.betweenness_centrality, top_k)

    errors["closeness_centrality"] = 1 - top_k_overlap(G, G_anon, nx.closeness_centrality, top_k)

    # For queries returning scalars, use relative error
    errors["number_of_edges"] = scalar_relative_error(G, G_anon, nx.number_of_edges, q_max_edges)

    return errors

def compute_queryability_score(query_errors: dict, weights: dict = None):
    """
    Aggregate per-query errors into a single queryability score.
    Returns a float between 0 and 1.
    """
    if weights is None:
        weights = {q: 1.0 for q in query_errors}

    total_weight = sum(weights.values())
    score = sum(query_errors[q] * weights[q] for q in query_errors)
    score = score / total_weight
    return score