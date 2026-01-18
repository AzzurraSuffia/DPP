import networkx as nx
from networkx.algorithms import isomorphism
from functools import cmp_to_key
from exceptions import AnonymizationImpossibleError

class SocialAnonymizer:
    def __init__(self):
        """
        Initialize the SocialAnonymizer with cost parameters.
        
        Args:
            alpha: Cost parameter (currently unused in cost calculation)
            beta: Edge cost multiplier
            gamma: Vertex cost multiplier
        """
        self.alpha = 0 # default
        self.beta = 1 # default
        self.gamma = 1 # default
    
    @staticmethod
    def bfs_order(subgraph: nx.Graph, start: int) -> list:
        """Perform BFS traversal and return node order."""
        visited = set()
        queue = [start] 
        order = []

        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                order.append(node)
                neighbors = sorted(subgraph.neighbors(node), key=lambda n: (subgraph.degree(n), n), reverse=True)
                for neighbor in sorted(neighbors): 
                    if neighbor not in visited and neighbor not in queue:
                        queue.append(neighbor)

        return order

    def anonymization_cost(self, Nu1: nx.Graph, Nv1: nx.Graph, Nu2: nx.Graph, Nv2: nx.Graph) -> float:
        """
        Calculate the anonymization cost between original and anonymized neighborhoods.
        
        Args:
            Nu1: Original neighborhood of u
            Nv1: Original neighborhood of v
            Nu2: Anonymized neighborhood of u
            Nv2: Anonymized neighborhood of v
            
        Returns:
            Total cost (edge cost + vertex cost)
        """
        H1 = nx.compose(Nu1, Nv1)
        H2 = nx.compose(Nu2, Nv2)

        added_edges = [(v1, v2) for v1, v2 in H2.edges() if not H1.has_edge(v1, v2)]
        edge_cost = self.beta * len(added_edges)

        vertex_cost = self.gamma * (len(H2.nodes()) - len(H1.nodes()))

        return edge_cost + vertex_cost

    @staticmethod
    def choose_starting_matching_nodes(c_u: nx.Graph, c_v: nx.Graph) -> tuple[int, int]:
        """Choose the best starting pair of nodes for matching."""
        degree_label_candidate_pair = None
        max_degree = -1

        # Step 1: degree-label matches
        for x in sorted(c_u.nodes()):
            deg_x = c_u.degree[x]
            
            for y in sorted(c_v.nodes()):
                deg_y = c_v.degree[y]
                
                if deg_x == deg_y:
                    if deg_x > max_degree or (deg_x == max_degree and (x, y) < degree_label_candidate_pair):
                        degree_label_candidate_pair = (x, y)
                        max_degree = deg_x

        if degree_label_candidate_pair:
            return degree_label_candidate_pair
        
        # Step 2: fallback - compute anonymization cost
        min_cost = float('inf')
        cost_candidate_pair = None

        for x in sorted(c_u.nodes()):
            deg_x = c_u.degree[x]

            for y in sorted(c_v.nodes()):
                deg_y = c_v.degree[y]

                cost = abs(deg_x - deg_y)

                if cost < min_cost or (cost == min_cost and (x, y) < cost_candidate_pair):
                    min_cost = cost
                    cost_candidate_pair = (x, y)

        return cost_candidate_pair

    @staticmethod
    def neighborhood_size_key(G: nx.Graph, v: int) -> tuple:
        """Calculate sorting key based on neighborhood size."""
        neighbors = list(G.neighbors(v))
        sub_nodes = [v] + neighbors
        subgraph = G.subgraph(sub_nodes)
        
        num_vertices = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()

        return (num_vertices, num_edges, v)

    @staticmethod
    def make_components_isomorphic(comp1: nx.Graph, comp2: nx.Graph, node_map_1_2: dict, node_map_2_1: dict) -> tuple[nx.Graph, nx.Graph]:
        """Make two components isomorphic by adding missing edges."""
        comp1_anon, comp2_anon = comp1.copy(), comp2.copy()

        # Equalize neighborhoods
        for (w, z) in comp1_anon.edges():
            if not comp2_anon.has_edge(node_map_1_2[w], node_map_1_2[z]):
                comp2_anon.add_edge(node_map_1_2[w], node_map_1_2[z])

        for (w, z) in comp2_anon.edges():
            if not comp1_anon.has_edge(node_map_2_1[w], node_map_2_1[z]):
                comp1_anon.add_edge(node_map_2_1[w], node_map_2_1[z])

        return comp1_anon, comp2_anon

    @staticmethod
    def find_missing_matching_vertex(G: nx.Graph, s: int, anonymized: dict, constraints: list) -> int:
        """Find a suitable matching vertex that satisfies constraints."""
        # Nodes that are NOT in constraints and NOT currently anonymized
        unanon_nodes = sorted(n for n, v in anonymized.items() if not v and n not in constraints)

        if unanon_nodes:
            search_space = unanon_nodes
        else:
            # Fallback: Nodes not in constraints (even if anonymized, though this triggers rollback)
            search_space = sorted(n for n in G.nodes() if n not in constraints)

        if not search_space:
            # Explicitly raise error if no nodes are left
            raise AnonymizationImpossibleError(
                f"Cannot add neighbor to node. Graph saturated or constraints too strict.")

        deg_dict = {n: G.degree(n) for n in search_space}
        min_deg = min(deg_dict.values())

        min_deg_nodes = sorted(n for n, d in deg_dict.items() if d == min_deg)

        return min_deg_nodes[0]

    @staticmethod
    def build_constraints(G: nx.Graph, center: int, *extra_constraints: set[int]) -> set[int]:
        """Build constraint set for vertex selection."""
        constraints = {center}
        constraints.update(G.neighbors(center))
        for s in extra_constraints:
            constraints.update(s)

        return constraints

    def build_component(self, G: nx.Graph, ref_comp: nx.Graph, ref_node: int, src_node: int, EquivalenceClassDict: dict) -> tuple[nx.Graph, nx.Graph, dict]:
        """Build a new component matching the reference component."""
        node_map_1_2 = {}
        node_map_2_1 = {}
        anonymized = {n: data['anonymized'] for n, data in G.nodes(data=True)}

        for ref_node_in_comp in sorted(ref_comp.nodes()):
            constraints = self.build_constraints(G, src_node, set(node_map_2_1.keys()))
            
            new_vertex = self.find_missing_matching_vertex(G, ref_node_in_comp, anonymized, constraints)

            if anonymized.get(new_vertex):
                for node in EquivalenceClassDict[new_vertex]:
                    anonymized[node] = False

            node_map_1_2[ref_node_in_comp] = new_vertex
            node_map_2_1[new_vertex] = ref_node_in_comp

        src_comp = G.subgraph(node_map_2_1.keys())
        
        ref_comp_anon, src_comp_anon = self.make_components_isomorphic(ref_comp, src_comp, node_map_1_2, node_map_2_1)

        return ref_comp_anon, src_comp_anon, node_map_1_2

    def match_and_generalize_components(self, G: nx.Graph, comp1: nx.Graph, comp2: nx.Graph, c1: int, c2: int, EquivalenceClassDict: dict) -> tuple[nx.Graph, nx.Graph, dict]:
        """Match and generalize two components."""
        anonymized = {n: data['anonymized'] for n, data in G.nodes(data=True)}

        (s1, s2) = self.choose_starting_matching_nodes(comp1, comp2)
        
        order1, order2 = self.bfs_order(comp1, s1), self.bfs_order(comp2, s2)

        node_map_1_2 = {}
        node_map_2_1 = {}

        min_len = min(len(order1), len(order2))

        # 1. Match existing nodes by BFS order
        for i in range(min_len):
            n1 = order1[i]
            n2 = order2[i]
            node_map_1_2[n1] = n2
            node_map_2_1[n2] = n1

        # 2. Handle nodes in comp1 that have no match in comp2
        for n1 in order1[min_len:]:
            constraints = self.build_constraints(G, c2, set(node_map_2_1.keys()))
            
            new_vertex = self.find_missing_matching_vertex(G, n1, anonymized, constraints)

            if anonymized[new_vertex]:
                for node in EquivalenceClassDict[new_vertex]:
                    anonymized[node] = False

            node_map_1_2[n1] = new_vertex
            node_map_2_1[new_vertex] = n1

        # 3. Handle nodes in comp2 that have no match in comp1
        for n2 in order2[min_len:]:
            constraints = self.build_constraints(G, c1, set(node_map_1_2.keys()))
            
            new_vertex = self.find_missing_matching_vertex(G, n2, anonymized, constraints)

            if anonymized[new_vertex]:
                for node in EquivalenceClassDict[new_vertex]:
                    anonymized[node] = False

            node_map_1_2[new_vertex] = n2
            node_map_2_1[n2] = new_vertex

        # 4. Create Anonymized Copies
        comp1_nodes = list(node_map_1_2.keys())
        comp2_nodes = list(node_map_2_1.keys())
        
        sub_c1 = G.subgraph(comp1_nodes)
        sub_c2 = G.subgraph(comp2_nodes)
        
        comp1_anon, comp2_anon = self.make_components_isomorphic(sub_c1, sub_c2, node_map_1_2, node_map_2_1)

        return comp1_anon, comp2_anon, node_map_1_2

    def most_similar_component(self, G: nx.Graph, ref_comp_id: int, component_graphs: dict, unmatched_component_ids: set[int], 
                               ref_node: int, src_node: int, EquivalenceClassDict: dict) -> tuple[int, nx.Graph, nx.Graph, dict]:
        """Find the most similar component based on anonymization cost."""
        best_cost = float('inf')
        most_similar_id = float('inf')
        best_ref_anon = None
        best_unm_anon = None
        best_mapping = {}
        
        sorted_unmatched_ids = sorted(unmatched_component_ids)
        
        ref_comp = component_graphs[ref_node][ref_comp_id]

        for unm_id in sorted_unmatched_ids:
            unm_comp = component_graphs[src_node][unm_id]

            ref_anon, unm_anon, current_mapping = self.match_and_generalize_components(
                G, ref_comp, unm_comp, ref_node, src_node, EquivalenceClassDict)

            cost = self.anonymization_cost(ref_comp, unm_comp, ref_anon, unm_anon)

            if cost < best_cost or (cost == best_cost and unm_id < most_similar_id):
                best_cost = cost
                best_ref_anon = ref_anon
                best_unm_anon = unm_anon
                most_similar_id = unm_id
                best_mapping = current_mapping

        return most_similar_id, best_ref_anon, best_unm_anon, best_mapping

    @staticmethod
    def get_neighborhood_components(G: nx.Graph, u: int, v: int) -> tuple[dict, dict]:
        """Extract neighborhood components for two nodes."""
        Nu = G.subgraph(sorted(G.neighbors(u)))
        Nv = G.subgraph(sorted(G.neighbors(v)))

        Cu = [sorted(c) for c in nx.connected_components(Nu)]
        Cv = [sorted(c) for c in nx.connected_components(Nv)]

        Cu.sort(key=lambda nodes: nodes[0] if nodes else float('inf'))
        Cv.sort(key=lambda nodes: nodes[0] if nodes else float('inf'))

        component_graphs = {
            u: {i: Nu.subgraph(nodes) for i, nodes in enumerate(Cu)},
            v: {i: Nv.subgraph(nodes) for i, nodes in enumerate(Cv)}
        }

        unmatched_components = {
            u: set(component_graphs[u].keys()),
            v: set(component_graphs[v].keys())
        }

        return component_graphs, unmatched_components

    @staticmethod
    def resolve_node(node_id: int, member_id: int, seed: int, group_mappings: dict) -> int:
        """Resolve a node ID using group mappings."""
        if node_id == seed:
            return member_id
        
        mapping = group_mappings.get(member_id, {})
        if node_id in mapping:
            return mapping[node_id]
            
        return node_id

    def sync_group_changes(self, G: nx.Graph, current_group: list, seed: int, changes: list, group_mappings: dict) -> set:
        """Synchronize changes across group members."""
        touched_nodes_during_sync = set()

        for member in current_group:
            if member == seed: continue
            if member not in group_mappings: continue

            for change in changes:
                if change['type'] == 'add_node':
                    seed_node_added = change['u']
                    
                    target_node_for_member = seed_node_added

                    constraints = set(G.neighbors(member))
                    constraints.add(member)
                    
                    if target_node_for_member in constraints:
                        anonymized_status = {n: data.get('anonymized', False) for n, data in G.nodes(data=True)}
                        
                        target_node_for_member = self.find_missing_matching_vertex(
                            G, seed_node_added, anonymized_status, constraints)

                    group_mappings[member][seed_node_added] = target_node_for_member
                    
                    if not G.has_edge(member, target_node_for_member):
                        G.add_edge(member, target_node_for_member)
                        
                    touched_nodes_during_sync.add(target_node_for_member)

                elif change['type'] == 'add_edge':
                    u_seed, v_seed = change['u'], change['v']
                    
                    u_mem = self.resolve_node(u_seed, member, seed, group_mappings)
                    v_mem = self.resolve_node(v_seed, member, seed, group_mappings)
                    
                    if u_mem != v_mem and not G.has_edge(u_mem, v_mem):
                        if G.has_edge(u_seed, v_seed):
                            attr = G.edges[u_seed, v_seed]
                            G.add_edge(u_mem, v_mem, **attr)
                        else:
                            G.add_edge(u_mem, v_mem)
                    
                        touched_nodes_during_sync.add(u_mem)
                        touched_nodes_during_sync.add(v_mem)
                    
        return touched_nodes_during_sync

    @staticmethod
    def update_graph(G: nx.Graph, c_u_anon: nx.Graph, c_v_anon: nx.Graph, u: int, v: int):
        """Update the graph with anonymized components."""
        G.add_nodes_from(c_u_anon.nodes(data=True))
        G.add_edges_from(c_u_anon.edges(data=True))
        for node in c_u_anon:
            G.add_edge(u, node)

        G.add_nodes_from(c_v_anon.nodes(data=True))
        G.add_edges_from(c_v_anon.edges(data=True))
        for node in c_v_anon:
            G.add_edge(v, node)

    def find_perfect_comp_matches(self, u: int, v: int, component_graphs: dict, unmatched_components: dict) -> tuple[dict, dict]:
        """Find perfectly matching isomorphic components."""

        perfect_mapping = {}
        matches_found = []

        # Keep track of already matched components in v
        matched_v = set()

        for cu_id in list(unmatched_components[u]):  # iterate safely
            c_u = component_graphs[u][cu_id]

            for cv_id in list(unmatched_components[v]):
                if cv_id in matched_v:
                    continue  # skip already paired component

                c_v = component_graphs[v][cv_id]

                gm = isomorphism.GraphMatcher(c_u, c_v)
                if gm.is_isomorphic(): # note: i could be using the exact minimum dfs code but this would have taken eternity 
                    perfect_mapping.update(gm.mapping)
                    matches_found.append((cu_id, cv_id))
                    matched_v.add(cv_id) 
                    break  # break inner loop; go to next cu_id

        # Remove matched components from unmatched sets
        for cu_id, cv_id in matches_found:
            unmatched_components[u].discard(cu_id)
            unmatched_components[v].discard(cv_id)

        return unmatched_components, perfect_mapping

    def _force_merge_components(self, G, node_u, component_dict, orphan_cid=None):
        """
        Helper to merge an orphan component into the main body of the neighborhood.
        Adds an edge between the orphan component and another component.
        """
        # Get all component IDs
        all_cids = list(component_dict.keys())
        
        if len(all_cids) < 2:
            raise Exception("No components to merge, only one component")

        # Identify which components to merge
        # If orphan_cid is provided, merge it with the first available other component
        cid_1 = orphan_cid if orphan_cid is not None else all_cids[0]
        
        # Find a cid_2 that is not cid_1
        cid_2 = None
        for c in all_cids:
            if c != cid_1:
                cid_2 = c
                break
        
        if cid_2 is None: return 

        # Pick one node from each component
        # We sort by degree to pick "hub" nodes (better utility usually)
        nodes_1 = sorted(component_dict[cid_1].nodes(), key=lambda n: G.degree(n), reverse=True)
        nodes_2 = sorted(component_dict[cid_2].nodes(), key=lambda n: G.degree(n), reverse=True)
        
        u1 = nodes_1[0]
        u2 = nodes_2[0]
        
        # Add the edge -> This merges the two components
        if not G.has_edge(u1, u2):
            G.add_edge(u1, u2)
            # We don't need to record 'changes' here because the restart 
            # will catch the new topology in the snapshot comparison.

    def anonymize_pair(self, G: nx.Graph, u: int, v: int, EquivalenceClassDict: dict) -> tuple[list, dict, set]:
        """Anonymize a pair of nodes by making their neighborhoods isomorphic."""
        mapping = {}
        changes = []
        touched_nodes = set()

        # Snapshot of neighborhoods
        original_u_nodes = set(G.neighbors(u)).copy()
        original_u_edges = set(G.subgraph(list(original_u_nodes) + [u]).edges()).copy()
        original_v_nodes = set(G.neighbors(v)).copy()
        original_v_edges = set(G.subgraph(list(original_v_nodes) + [v]).edges()).copy()

        neighborhood_stable = False
        
        while not neighborhood_stable:
            neighborhood_stable = True 
            mapping = {u: v}

            # Get neighborhood components
            component_graphs, unmatched_components = self.get_neighborhood_components(G, u, v)

            #print(f"Components of node {u}", [value.nodes() for key, value in component_graphs[u].items()])
            #print(f"Components of node {v}", [value.nodes() for key, value in component_graphs[v].items()])
            # Perfect Matches
            unmatched_components, perfect_mapping = self.find_perfect_comp_matches(u, v, component_graphs, unmatched_components)
            mapping.update(perfect_mapping)

            while unmatched_components[u] and unmatched_components[v]:
                largest_u_id = max(unmatched_components[u], key=lambda cid: len(component_graphs[u][cid]))
                largest_v_id = max(unmatched_components[v], key=lambda cid: len(component_graphs[v][cid]))
                
                largest_u_component = component_graphs[u][largest_u_id]
                largest_v_component = component_graphs[v][largest_v_id]

                if len(largest_u_component) >= len(largest_v_component):
                    target, source = u, v
                    target_cid, source_cid = largest_u_id, largest_v_id
                else:
                    target, source = v, u
                    target_cid, source_cid = largest_v_id, largest_u_id

                try:
                    most_sim_id, target_anon_g, source_anon_g, local_mapping = self.most_similar_component(
                        G, target_cid, component_graphs, unmatched_components[source],
                        ref_node=target, src_node=source, EquivalenceClassDict=EquivalenceClassDict
                    )

                    if source == v:
                        mapping.update(local_mapping)
                    else:
                        for target_node, source_node in local_mapping.items():
                            mapping[source_node] = target_node

                    self.update_graph(G, target_anon_g, source_anon_g, target, source)

                    unmatched_components[target].remove(target_cid)
                    unmatched_components[source].remove(most_sim_id)

                except AnonymizationImpossibleError:
                    #print(f"DEBUG: Cannot extend component in {source} to match component in {target}. Merging components in {target} instead.")
                    self._force_merge_components(G, source, component_graphs[source])
                    neighborhood_stable = False
                    break

            if not neighborhood_stable: continue
            
            # Orphaned Components
            if unmatched_components[u]:
                target, source, extras = u, v, unmatched_components[u].copy()
            else:
                target, source, extras = v, u, unmatched_components[v].copy()

            for cid in extras:
                extra_comp = component_graphs[target][cid]

                try:
                    # Try to create the component in 'source'
                    target_anon_g, source_anon_g, local_mapping = self.build_component(
                        G, extra_comp, target, source, EquivalenceClassDict)

                except AnonymizationImpossibleError:
                    # --- SOLUTION IMPLEMENTATION ---
                    #print(f"DEBUG: Cannot create new component in {source}. Merging components in {target} instead.")
                    self._force_merge_components(G, target, component_graphs[target], cid)
                    neighborhood_stable = False
                    break
                    
                if source == v:
                    mapping.update(local_mapping)
                else:
                    for target_node, source_node in local_mapping.items():
                        mapping[source_node] = target_node

                self.update_graph(G, target_anon_g, source_anon_g, target, source)
                unmatched_components[target].remove(cid)

        # Calculate Changes & Touched Nodes
        current_u_nodes = set(G.neighbors(u))
        
        for n in current_u_nodes:
            if n not in original_u_nodes:
                changes.append({'type': 'add_node', 'u': n})
                touched_nodes.add(n)

        current_subgraph_u = G.subgraph(list(current_u_nodes) + [u])
        
        for e in current_subgraph_u.edges():
            if (e[0], e[1]) not in original_u_edges and (e[1], e[0]) not in original_u_edges:
                changes.append({'type': 'add_edge', 'u': e[0], 'v': e[1]})
                touched_nodes.add(e[0])
                touched_nodes.add(e[1])

        current_v_nodes = set(G.neighbors(v))
        
        for n in current_v_nodes:
            if n not in original_v_nodes:
                touched_nodes.add(n)

        current_subgraph_v = G.subgraph(list(current_v_nodes) + [v])
        for e in current_subgraph_v.edges():
            if (e[0], e[1]) not in original_v_edges and (e[1], e[0]) not in original_v_edges:
                touched_nodes.add(e[0])
                touched_nodes.add(e[1])

        return changes, mapping, touched_nodes

    def anonymize_graph(self, G: nx.Graph, k: int, alpha: float, beta: float, gamma: float) -> tuple[nx.Graph, dict]:
        """
        Main method: Anonymize the graph to satisfy k-anonymity.
        """
        if k > G.number_of_nodes():
            raise ValueError(
                f"impossible anonymization: k={k} exceeds number of nodes ({G.number_of_nodes()}).")
        
        self.alpha = alpha
        self.beta = beta
        self.gama = gamma
        
        G_anon = G.copy()
        VertexList = sorted(G_anon.nodes, key=lambda v: self.neighborhood_size_key(G_anon, v), reverse=True)

        EquivalenceClassDict = {}
        nx.set_node_attributes(G_anon, False, 'anonymized')

        while VertexList:
            SeedVertex = VertexList.pop(0)
            deg_seed = G_anon.degree[SeedVertex]

            # --- Candidate Selection ---
            costs = {}
            for v in VertexList:
                deg_v = G_anon.degree[v]
                costs[v] = abs(deg_seed - deg_v)

            if len(VertexList) >= (2*k - 1):
                sorted_cost = sorted(costs.items(), key=lambda item: item[1])
                candidate_set = [node for node, _ in sorted_cost[:k-1]]
            else: 
                candidate_set = VertexList.copy()

            current_group = [SeedVertex] + candidate_set
            
            # Temporarily register the group mapping (will be finalized if successful)
            for v in current_group:
                EquivalenceClassDict[v] = current_group

            #print("---------------------")
            #print("Seed Vertex: ", SeedVertex)
            #print("Processing group ", current_group)

            # --- RESTART MECHANISM START ---
            group_stable = False
            while not group_stable:
                group_stable = True  # Assume success, set to False if we need to restart
                group_mappings = {} 
                
                # We collect broken groups here, but apply them only if the group stabilizes
                pending_broken_groups = [] 

                for j in range(1, len(current_group)):
                    uj = current_group[j]
                    processed_members = set(current_group[1:j]) # Members processed in previous iterations

                    #print(f"Processing pair {SeedVertex}-{uj}")
                    
                    # 1. Anonymize Pair
                    changes, mapping, primary_touched = self.anonymize_pair(G_anon, SeedVertex, uj, EquivalenceClassDict)
                    group_mappings[uj] = mapping

                    # --- DETECTION LOGIC START ---
                    # Check if 'primary_touched' contains any member we already processed (indices 1 to j-1).
                    # If so, the isomorphism for those members is potentially broken because 
                    # an edge was added to them outside of the sync process.
                    intra_group_conflict = primary_touched.intersection(processed_members)
                    
                    if intra_group_conflict:
                        #print(f"!!! CONFLICT DETECTED: Node(s) {intra_group_conflict} within current group modified.")
                        #print("!!! Restarting group processing...")
                        group_stable = False
                        break # Break the 'for j' loop, 'while' loop will restart
                    # --- DETECTION LOGIC END ---

                    # 2. Sync changes to previous members
                    secondary_touched = set()
                    if j > 1:
                        #print(f"Synching changes to {current_group[1:j]}")
                        secondary_touched = self.sync_group_changes(G_anon, current_group[1:j], SeedVertex, changes, group_mappings)
                        # Note: sync_group_changes explicitly handles keeping 1..j-1 consistent, so 
                        # we don't usually check secondary_touched for intra-group conflict here.

                    # 3. Collect external broken groups (Ripple Effect)
                    all_touched_nodes = primary_touched.union(secondary_touched)
                    
                    for node in all_touched_nodes:
                        # If we touched a node OUTSIDE current group that was already anonymized
                        if node not in current_group:
                            if G_anon.nodes[node].get('anonymized') == True:
                                broken_group = EquivalenceClassDict.get(node, [node])
                                if broken_group not in pending_broken_groups:
                                    pending_broken_groups.append(broken_group)

                # End of 'for j' loop.
                # If group_stable is still True, we finished the group successfully.
                
                if group_stable:
                    # Apply the external broken groups logic now that we are sure this group is done
                    all_broken_members = set()
                    for bg in pending_broken_groups:
                        for member in bg:
                            if member not in all_broken_members:
                                all_broken_members.add(member)
                                G_anon.nodes[member]['anonymized'] = False
                                EquivalenceClassDict.pop(member, None) # Remove mapping
                                if member not in VertexList:
                                    VertexList.append(member)
            # --- RESTART MECHANISM END ---

            # Finalize current group
            for node in current_group:
                G_anon.nodes[node]['anonymized'] = True
                if node in VertexList:
                    VertexList.remove(node)

            # Re-sort VertexList as degrees might have changed
            VertexList.sort(key=lambda v: self.neighborhood_size_key(G_anon, v), reverse=True)

        return G_anon, EquivalenceClassDict