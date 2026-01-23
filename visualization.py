import networkx as nx
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from utils import check_isomorphic_classes

def plot_graph(comp, title="Graph"):
    """
    Plot graph with spirng layout.
    """

    pos = nx.spring_layout(comp, seed=42)  # deterministic
    plt.figure(figsize=(6, 6))
    
    # Draw nodes
    nx.draw_networkx_nodes(comp, pos, node_color='skyblue', node_size=250, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(comp, pos, width=1.5, alpha=0.7)

    nx.draw_networkx_labels(comp, pos, font_size=10, font_color='black')
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()

def plot_runtime_comparison(ns, times_a, times_b, label_a, label_b, suptitle):

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ns, times_a, marker='o', color='tab:blue')
    axes[0].set_title(label_a)
    axes[0].set_xlabel('Number of nodes (n)')
    axes[0].set_ylabel('Average runtime (seconds)')
    axes[0].grid(True, which='both')

    axes[1].plot(ns, times_b, marker='o', color='tab:orange')
    axes[1].set_title(label_b)
    axes[1].set_xlabel('Number of nodes (n)')
    axes[1].set_ylabel('Average runtime (seconds)')
    axes[1].grid(True, which='both')

    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def generate_degree_anonymization_gif(
    G_original: nx.Graph,
    social_anonymizer,
    k_values: list[int],
    output_dir: str,
    gif_name: str = "degree_evolution.gif",
    duration: int = 1000
):
    """
    Create a GIF showing the evolution of node degrees during k-anonymization.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    n = G_original.number_of_nodes()

    for k in k_values:
        # Anonymize the graph
        G_anon, eq_classes = social_anonymizer.anonymize_graph(
            G_original, k=k, alpha=0, beta=1, gamma=1
        )

        # Optional: verify isomorphism
        if len(check_isomorphic_classes(G_anon, eq_classes)) != 0:
            raise Exception(f"Anonymization for k={k} failed isomorphism check.")

        degrees = [d for _, d in G_anon.degree()]

        # Create figure with two subplots
        fig, (ax_graph, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

        # Left subplot: graph
        pos = nx.spring_layout(G_anon, seed=42)
        labels = {node: str(node) for node in G_anon.nodes()}

        nx.draw_networkx_nodes(G_anon, pos, ax=ax_graph, node_color='skyblue', node_size=300)
        nx.draw_networkx_edges(G_anon, pos, ax=ax_graph, alpha=0.6)
        nx.draw_networkx_labels(
            G_anon, pos, labels=labels, ax=ax_graph,
            horizontalalignment='center', verticalalignment='center',
            font_size=10, font_color='black'
        )
        ax_graph.set_title(f"k={k}")
        ax_graph.axis("off")

        # Right subplot: degree histogram
        ax_hist.hist(
            degrees,
            bins=range(min(degrees), max(degrees) + 2),
            color="skyblue",
            edgecolor="black",
            align="left"
        )
        ax_hist.set_xlabel("Degree")
        ax_hist.set_ylabel("Number of nodes")
        ax_hist.set_title("Degree distribution")
        ax_hist.set_xticks(range(0, n))
        ax_hist.set_yticks(range(0, n+1))
        ax_hist.grid(True, axis="y", alpha=0.7)

        plt.tight_layout()

        # Save frame
        frame_path = os.path.join(output_dir, f"frame_k_{k}.png")
        plt.savefig(frame_path, dpi=150)
        plt.close(fig)

        frames.append(imageio.imread(frame_path))

    # Save GIF
    gif_path = os.path.join(output_dir, gif_name)
    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"GIF saved to {gif_path}")

def generate_betweenness_anonymization_gif(
    G_original: nx.Graph,
    social_anonymizer,
    k_values: list[int],
    output_dir: str,
    gif_name: str = "betweenness_evolution.gif",
    duration: int = 1000
):
    """
    Create a GIF showing the evolution of node betweenness centrality during k-anonymization.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    
    for k in k_values:
        # Anonymize the graph
        G_anon, eq_classes = social_anonymizer.anonymize_graph(G_original, k=k, alpha=0, beta=1, gamma=1)
        
        # Verify isomorphism
        if len(check_isomorphic_classes(G_anon, eq_classes)) != 0:
            raise Exception(f"Anonymization for k={k} failed isomorphism check.")

        # Compute betweenness centrality
        bc = nx.betweenness_centrality(G_anon, normalized=True)
        bc_values = [bc[node] for node in G_anon.nodes()]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G_anon, seed=42)
        
        labels = {node: str(node) for node in G_anon.nodes()}

        # Draw nodes with betweenness coloring
        nx.draw_networkx_nodes(
            G_anon, pos, node_color=bc_values, cmap=plt.cm.viridis,
            node_size=300, ax=ax, vmin=0, vmax=1)
        nx.draw_networkx_edges(G_anon, pos, ax=ax, alpha=0.6)
        nx.draw_networkx_labels(G_anon, pos, labels=labels, font_color='white', ax=ax) #font_weight='bold',
        
        ax.set_title(f"k={k}", fontsize=14)
        ax.axis("off")
        
        # colorbar with fixed range
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label("Betweenness Centrality")
        
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_k_{k}.png")
        plt.savefig(frame_path, dpi=150)
        plt.close(fig)
        
        frames.append(imageio.imread(frame_path))
    
    # Save GIF
    gif_path = os.path.join(output_dir, gif_name)
    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"GIF saved to {gif_path}")

def generate_closeness_anonymization_gif(
    G_original: nx.Graph,
    social_anonymizer,
    k_values: list[int],
    output_dir: str,
    gif_name: str = "closeness_evolution.gif",
    duration: int = 1000
):
    """
    Create a GIF showing the evolution of node closeness centrality during k-anonymization.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    
    for k in k_values:
        # Anonymize the graph
        G_anon, eq_classes = social_anonymizer.anonymize_graph(G_original, k=k, alpha=0, beta=1, gamma=1)
        
        # Verify isomorphism
        if len(check_isomorphic_classes(G_anon, eq_classes)) != 0:
            raise Exception(f"Anonymization for k={k} failed isomorphism check.")
        
        # Compute closeness centrality
        cc = nx.closeness_centrality(G_anon)
        cc_values = [cc[node] for node in G_anon.nodes()]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G_anon, seed=42)
        
        labels = {node: str(node) for node in G_anon.nodes()}

        # Draw nodes with closeness coloring
        nx.draw_networkx_nodes(
            G_anon, pos, node_color=cc_values, cmap=plt.cm.viridis,
            node_size=300, ax=ax, vmin=0, vmax=1)
        nx.draw_networkx_edges(G_anon, pos, ax=ax, alpha=0.6)
        nx.draw_networkx_labels(G_anon, pos, labels=labels, font_color='white', ax=ax) #font_weight='bold',
        
        ax.set_title(f"k={k}", fontsize=14)
        ax.axis("off")
        
        # colorbar with fixed range
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label("Closeness Centrality")
        
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_k_{k}.png")
        plt.savefig(frame_path, dpi=150)
        plt.close(fig)
        
        frames.append(imageio.imread(frame_path))
    
    # Save GIF
    gif_path = os.path.join(output_dir, gif_name)
    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"GIF saved to {gif_path}")