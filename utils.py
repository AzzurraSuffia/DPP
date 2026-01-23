import networkx as nx
from networkx.algorithms import isomorphism

def derive_equivalence_classes(EquivalenceClassDict: dict):
    """
    Derive the unique equivalence classes from a node-to-class mapping.
    """
    unique_groups = set()
    for group in EquivalenceClassDict.values():
        unique_groups.add(tuple(sorted(group)))
    return unique_groups

def check_isomorphic_classes(G: nx.Graph, EquivalenceClassDict: dict) -> list:
    """
    Verify whether each equivalence class in EquivalenceClassDict is satisfied in G.
    Returns a list of groups (tuples of nodes) that violate isomorphism constraints.
    """
    unique_groups = derive_equivalence_classes(EquivalenceClassDict)
    
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
        for other_node in nodes[1:]:
            other_scope = [other_node] + list(G.neighbors(other_node))
            other_subgraph = G.subgraph(other_scope)
            
            # Check Isomorphism
            GM = isomorphism.GraphMatcher(ref_subgraph, other_subgraph)
            
            if not GM.is_isomorphic():
                # Optional debug message
                #print(f"Isomorphism verification FAILED for group: {group}")
                #print(f"Mismatch between reference node {ref_node} and node {other_node}")        
                violating_groups.append(group)
                break 
    
    return violating_groups

def is_k_anonymous(G: nx.Graph, k: int) -> bool:
    """
    Checks if a graph satisfies k-anonymity based on 1-neighborhood isomorphism.
    """

    # Step 0. Check only structure or structure and labels
    node_matcher = None
    if all('label' in G.nodes[n] for n in G.nodes):
        node_matcher = isomorphism.categorical_node_match('label', None)

    # Step 1. Pre-calculate all subgraphs
    node_neighborhoods = {}
    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        node_neighborhoods[u] = G.subgraph(neighbors + [u])

    # Step 2. Group nodes into equivalence classes
    equivalence_classes = []
    visited = set()
    nodes = list(G.nodes())

    for i in range(len(nodes)):
        u = nodes[i]
        if u in visited:
            continue

        # Start a new group with u
        current_group = [u]
        visited.add(u)
        sub_u = node_neighborhoods[u]
        
        # Optimization: Pre-calculate invariants for u
        u_nodes = sub_u.number_of_nodes()
        u_edges = sub_u.number_of_edges()

        # Compare with all other unvisited nodes
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            if v in visited:
                continue

            sub_v = node_neighborhoods[v]

            # Optimization: neighbors count check
            if sub_v.number_of_nodes() != u_nodes:
                continue

            # Optimization: edge count check
            if sub_v.number_of_edges() != u_edges:
                continue

            # Isomorphism check
            if nx.is_isomorphic(sub_u, sub_v, node_match=node_matcher):
                current_group.append(v)
                visited.add(v)

        equivalence_classes.append(current_group)

    # 3. Check k-anonymity constraint
    min_group_size = float('inf')
    
    if not equivalence_classes:
        min_group_size = 0
    else:
        min_group_size = min(len(group) for group in equivalence_classes)

    is_satisfied = min_group_size >= k

    return is_satisfied

