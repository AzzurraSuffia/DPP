import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism
from tqdm import tqdm
import time
import numpy as np
import random
from MDFScoder import MDFSCoder
from LabelDomain import LabelDomain

def plot_component(comp, title="Component"):
    pos = nx.spring_layout(comp, seed=42)  # deterministic layout
    plt.figure(figsize=(6, 6))
    
    # Draw nodes
    nx.draw_networkx_nodes(comp, pos, node_color='skyblue', node_size=600, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(comp, pos, width=1.5, alpha=0.7)

    nx.draw_networkx_labels(comp, pos, font_size=10, font_color='black')
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()

def devise_equivalence_classes(EquivalenceClassDict: dict):
    unique_groups = set()
    for group in EquivalenceClassDict.values():
        unique_groups.add(tuple(sorted(group)))
    return unique_groups

def check_isomorphic_classes(G: nx.Graph, EquivalenceClassDict: dict) -> list:
    
    unique_groups = devise_equivalence_classes(EquivalenceClassDict)
    
    violating_groups = []

    for group in unique_groups:
        nodes = list(group)
        
        # Groups of size 1 are trivially isomorphic
        if len(nodes) < 2:
            continue
            
        # Take the first node as the 'Reference'
        ref_node = nodes[0]
        ref_scope = [ref_node] + list(G.neighbors(ref_node))
        ref_subgraph = G.subgraph(ref_scope)
        
        # Compare every other node in the group to the reference
        group_failed = False
        for other_node in nodes[1:]:
            other_scope = [other_node] + list(G.neighbors(other_node))
            other_subgraph = G.subgraph(other_scope)
            
            # Check Isomorphism
            GM = isomorphism.GraphMatcher(
                ref_subgraph, 
                other_subgraph)
            
            if not GM.is_isomorphic():
                print(f"Isomorphism verification FAILED for group: {group}")
                print(f"Mismatch between reference node {ref_node} and node {other_node}")
                
                violating_groups.append(group)
                break 
    
    return violating_groups

def random_iso_pair(n, p, seed):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    nodes = list(G.nodes())
    perm = dict(zip(nodes, random.sample(nodes, len(nodes))))
    H = nx.relabel_nodes(G, perm)
    return G, H

def random_non_iso_pair(n, p, seeds):
    if seeds[0] == seeds[1]:
        raise ValueError("Seeds should be different")
    G = nx.erdos_renyi_graph(n, p, seeds[0])
    H = nx.erdos_renyi_graph(n, p, seeds[1])

    if not nx.is_isomorphic(G, H):
        return G, H
    else:
        return None

def positive_benchmark(ns, p, seeds):
    results_MDFS = []
    results_vf2 = []
    coder = MDFSCoder()

    for n in tqdm(ns, desc="Graph size"):
        times_MDFS = []
        times_vf2 = []

        for seed in seeds:
            
            G, H = random_iso_pair(n, p, seed)

            start_time = time.perf_counter() # more refined resolution than time.time()
            coder.is_isomoprhic(G, H)
            end_time = time.perf_counter()
            times_MDFS.append(end_time - start_time)

            start_time = time.perf_counter()
            nx.is_isomorphic(G, H) #GraphMatcher inside
            end_time = time.perf_counter()
            times_vf2.append(end_time - start_time)

        results_MDFS.append(np.mean(times_MDFS))
        results_vf2.append(np.mean(times_vf2))

    return results_MDFS, results_vf2

def negative_benchmark(ns, p, seeds):
    results_MDFS = []
    results_vf2 = []
    coder = MDFSCoder()

    for n in tqdm(ns, desc="Graph size"):
        times_MDFS = []
        times_vf2 = []

        for pair_seeds in seeds:
            
            G, H = random_non_iso_pair(n, p, pair_seeds)

            start_time = time.perf_counter()
            coder.is_isomoprhic(G, H)
            end_time = time.perf_counter()
            times_MDFS.append(end_time - start_time)

            start_time = time.perf_counter()
            nx.is_isomorphic(G, H) #GraphMatcher inside
            end_time = time.perf_counter()
            times_vf2.append(end_time - start_time)

        results_MDFS.append(np.mean(times_MDFS))
        results_vf2.append(np.mean(times_vf2))

    return results_MDFS, results_vf2

def calculate_global_loss(G_original: nx.Graph, G_anonymized: nx.Graph, label_domain: LabelDomain | None = None) -> float:

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