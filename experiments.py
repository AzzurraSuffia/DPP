import networkx as nx
import random
import time
import numpy as np
from tqdm import tqdm
from mdfs_coder import MDFSCoder
from social_anonymizer import SocialAnonymizer
from utils import check_isomorphic_classes

def random_iso_pair(n, p, seed):
    """
    Generate a deterministic pair of isomorphic graphs.
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    nodes = list(G.nodes())
    perm = dict(zip(nodes, random.sample(nodes, len(nodes))))
    H = nx.relabel_nodes(G, perm)
    return G, H

def random_non_iso_pair(n, p, seeds):
    """
    Generate a deterministic pair of non-isomorphic graphs. 
    """
    if seeds[0] == seeds[1]:
        raise ValueError("Seeds should be different")
    G = nx.erdos_renyi_graph(n, p, seeds[0])
    H = nx.erdos_renyi_graph(n, p, seeds[1])

    if not nx.is_isomorphic(G, H):
        return G, H
    else:
        return None

def positive_benchmark(sizes, p, trials=5, rng_seed=None):
    """
    Benchmark the performance of MDFSCoder and NetworkX VF2 on pairs of isomorphic graphs.
    """
    rng = random.Random(rng_seed)
    results_MDFS = []
    results_vf2 = []
    coder = MDFSCoder()

    for size in tqdm(sizes, desc="Graph size"):
        times_MDFS = []
        times_vf2 = []

        for _ in range(trials):
            seed = rng.randint(0, 100000)
            G, H = random_iso_pair(size, p, seed)

            start_time = time.perf_counter() # more refined resolution than time.time()
            coder.is_isomoprhic(G, H)
            times_MDFS.append(time.perf_counter() - start_time)

            start_time = time.perf_counter()
            nx.is_isomorphic(G, H) # GraphMatcher called inside
            times_vf2.append(time.perf_counter() - start_time)

        results_MDFS.append(np.mean(times_MDFS))
        results_vf2.append(np.mean(times_vf2))

    return results_MDFS, results_vf2
    
def negative_benchmark(sizes, p, trials=5, rng_seed=None):
    """
    Benchmark the performance of MDFSCoder and NetworkX VF2 on pairs of non-isomorphic graphs.
    """
    results_MDFS = []
    results_vf2 = []
    coder = MDFSCoder()

    # Create a local random generator for deterministic seed generation
    rng = random.Random(rng_seed) if rng_seed is not None else random

    for size in tqdm(sizes, desc="Graph size"):
        times_MDFS = []
        times_vf2 = []

        # Generate seeds for non-isomorphic pairs
        pair_seeds_list = [(rng.randint(0, 100_000), rng.randint(0, 100_000)) for _ in range(trials)]

        for pair_seeds in pair_seeds_list:
            G, H = random_non_iso_pair(size, p, pair_seeds)

            start_time = time.perf_counter()
            coder.is_isomoprhic(G, H)
            end_time = time.perf_counter()
            times_MDFS.append(end_time - start_time)

            start_time = time.perf_counter()
            nx.is_isomorphic(G, H)
            end_time = time.perf_counter()
            times_vf2.append(end_time - start_time)

        results_MDFS.append(np.mean(times_MDFS) if times_MDFS else 0)
        results_vf2.append(np.mean(times_vf2) if times_vf2 else 0)

    return results_MDFS, results_vf2

def run_metric_comparison_experiment(
    n,
    m,
    k_values,
    seeds,
    num_seeds,
    experiment_func,
    alpha=0,
    beta=1,
    gamma=1,
    fallback_rng_seed=999,
    **metric_kwargs
):
    """
    Run experiments to investigate a metric on BA and ER graphs using a local deterministic RNG.
    This function generates BA (Barabási-Albert) and ER (Erdős-Rényi) graphs with the same
    number of nodes and edges, anonymizes them using the SocialAnonymizer, and computes a
    user-provided metric. Results are averaged over multiple seeds to reduce randomness.
    """
    
    # Local RNG avoids affecting global random state
    local_rng = random.Random(fallback_rng_seed)
    
    social_anonymizer = SocialAnonymizer()
    ba_results = []
    er_results = []

    for k in k_values:
        curr_ba = []
        curr_er = []

        seed_idx = 0
        successful = 0

        while successful < num_seeds:
            # prioritize seeds from provided list first, then fallback to local RNG
            if seed_idx < len(seeds):
                seed = seeds[seed_idx]
                seed_idx += 1
            else:
                seed = local_rng.randint(0, 100_000)

            try:
                # BA model
                G_ba = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
                actual_edges = G_ba.number_of_edges()
                G_anon_ba, eq_classes_ba = social_anonymizer.anonymize_graph(G_ba, k=k, alpha=alpha, beta=beta, gamma=gamma)
                if len(check_isomorphic_classes(G_anon_ba, eq_classes_ba)) != 0:
                    raise Exception("BA result is not isomorphic")
                
                # get the metric
                ba_value = experiment_func(G_ba, G_anon_ba, **metric_kwargs)

                # ER model
                G_er = nx.gnm_random_graph(n=n, m=actual_edges, seed=seed)
                G_anon_er, eq_classes_er = social_anonymizer.anonymize_graph(G_er, k=k, alpha=alpha, beta=beta, gamma=gamma)
                if len(check_isomorphic_classes(G_anon_er, eq_classes_er)) != 0:
                    raise Exception("ER result is not isomorphic")
                
                # get the metric
                er_value = experiment_func(G_er, G_anon_er, **metric_kwargs)

                curr_ba.append(ba_value)
                curr_er.append(er_value)
                successful += 1

            except Exception as e:
                print(f"[k={k}] seed {seed} discarded: {e}")

        ba_results.append(np.mean(curr_ba))
        er_results.append(np.mean(curr_er))

    return ba_results, er_results
