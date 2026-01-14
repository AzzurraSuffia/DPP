from functools import cmp_to_key

# note: if you introduce back labels, you should define a code also for isolated nodes

class MDFSCoder:
    def __init__(self):
        self.best_code = None

    def get_code(self, G):
        if G.number_of_nodes() == 0: return []
        self.best_code = None
        
        # Sort for deterministic execution
        nodes = sorted(list(G.nodes()))

        for start_node in nodes:
            # Init State
            mapping = {start_node: 0}
            current_code = []
            used_edges = set() # Needed for undirected bookkeeping
            
            self._dfs_search(G, mapping, current_code, used_edges)
            
        return self.best_code

    def _dfs_search(self, G, mapping, current_code, used_edges):
        # --- 1. PRUNING ---
        if self.best_code is not None:
            # If current partial code is already worse than best found, stop.
            if self._compare_code_sequences(current_code, self.best_code) == 1:
                return

        # --- 2. SUCCESS ---
        if len(current_code) == G.number_of_edges():
            self.best_code = list(current_code)
            return

        # --- 3. GENERATE CANDIDATES ---
        # Exact: Finds ALL valid next edges (Backtracking branching)
        candidates = self._get_valid_extensions(G, mapping, used_edges)
        
        # Sort to try the most promising edge first
        candidates.sort(key=cmp_to_key(self._compare_candidates))

        # --- 4. RECURSE (Branching) ---
        for cand in candidates:
            edge = cand['edge']
            real_target = cand['real_target']
            real_edge_key = cand['real_edge_key']

            # Apply State
            current_code.append(edge)
            used_edges.add(real_edge_key)
            if cand['type'] == 'fwd': mapping[real_target] = edge[1]

            # Recurse
            self._dfs_search(G, mapping, current_code, used_edges)

            # Backtrack (Restore State)
            if cand['type'] == 'fwd': del mapping[real_target]
            used_edges.remove(real_edge_key)
            current_code.pop()

    def _get_valid_extensions(self, G, mapping, used_edges):
        """Finds all forward/backward edges from the rightmost path."""
        candidates = []
        dfs_to_real = {v: k for k, v in mapping.items()}
        current_max_dfs = len(mapping) - 1

        for u_dfs in range(current_max_dfs, -1, -1):
            u_real = dfs_to_real[u_dfs]
            neighbors = G.neighbors(u_real)
            found_at_level = False
            
            for v_real in neighbors:
                edge_key = tuple(sorted((u_real, v_real)))
                if edge_key in used_edges: continue

                if v_real in mapping:
                    if u_dfs == current_max_dfs: # Back edge constraint
                        candidates.append({
                            'type': 'back', 'edge': (u_dfs, mapping[v_real]),
                            'real_target': v_real, 'real_edge_key': edge_key
                        })
                        found_at_level = True
                else:
                    candidates.append({
                        'type': 'fwd', 'edge': (u_dfs, current_max_dfs + 1),
                        'real_target': v_real, 'real_edge_key': edge_key
                    })
                    found_at_level = True
            
            if found_at_level: break
        return candidates

    def _compare_candidates(self, c1, c2):
        return self._compare_edges(c1['edge'], c2['edge'])

    def _compare_code_sequences(self, codeA, codeB):
        min_len = min(len(codeA), len(codeB))
        for k in range(min_len):
            res = self._compare_edges(codeA[k], codeB[k])
            if res != 0: return res
        if len(codeA) < len(codeB): return -1
        if len(codeA) > len(codeB): return 1
        return 0

    def _compare_edges(self, e1, e2):
        """Paper's 4 Rules"""
        u1, v1 = e1
        u2, v2 = e2
        fw1 = u1 < v1
        fw2 = u2 < v2
        
        if not fw1 and fw2: return -1
        if fw1 and not fw2: return 1
        if not fw1 and not fw2: # Both Back
            if u1 < u2: return -1
            if u1 > u2: return 1
            if v1 < v2: return -1
            if v1 > v2: return 1
            return 0
        if fw1 and fw2: # Both Fwd
            if u1 > u2: return -1
            if u1 < u2: return 1
            if v1 < v2: return -1
            if v1 > v2: return 1
            return 0
        return 0