import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism

def compare_components(comp1, comp2):
    """
    Compares two neighborhood components C1 and C2 based on the paper's rules:
    1. |V(C1)| < |V(C2)|
    2. |E(C1)| < |E(C2)|
    3. DFS(C1) < DFS(C2) (using M-DFS edge comparison rules)
    
    Input: comp = (num_nodes, num_edges, dfs_code)
    Output: -1 if c1 < c2, 1 if c1 > c2, 0 if equal
    """
    v1, e1, code1 = comp1
    v2, e2, code2 = comp2
    
    # Priority 1: Number of Vertices
    if v1 < v2: return -1
    if v1 > v2: return 1
    
    # Priority 2: Number of Edges
    if e1 < e2: return -1
    if e1 > e2: return 1
    
    # Priority 3: DFS Code Lexicographical Comparison
    # We must compare code1 and code2 edge-by-edge using the paper's rules
    min_len = min(len(code1), len(code2))
    for k in range(min_len):
        edge_res = compare_edges_paper(code1[k], code2[k])
        if edge_res != 0:
            return edge_res
            
    # If one is a prefix of the other (shouldn't happen if |E| is equal, but for safety)
    if len(code1) < len(code2): return -1
    if len(code1) > len(code2): return 1
    
    return 0

def compare_edges_paper(e1, e2):
    """
    Compares two edges based on the 4 Rules.
    This logic must match exactly what is inside your MDFSCoder.
    """
    u1, v1 = e1
    u2, v2 = e2
    fw1 = u1 < v1
    fw2 = u2 < v2

    # Rule 1: Backward < Forward
    if not fw1 and fw2: return -1
    if fw1 and not fw2: return 1

    # Rule 2: Both Backward
    if not fw1 and not fw2:
        if u1 < u2: return -1
        if u1 > u2: return 1
        if v1 < v2: return -1
        if v1 > v2: return 1
        return 0

    # Rule 3: Both Forward
    if fw1 and fw2:
        if u1 > u2: return -1 # Deeper source is smaller
        if u1 < u2: return 1
        if v1 < v2: return -1
        if v1 > v2: return 1
        return 0
    return 0

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