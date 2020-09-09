import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import identity
import os
import pickle

def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                            neighbors_key: str = None, probabilities_key: str = None, first_travel_key: str = None,
                            quiet: bool = False) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks


def get_probabilities(current_node, A, node_labels_to_int, probabilities_key: str = None, p: float = 1, q: float = 1):

    results = []
    current_node_pos = node_labels_to_int[current_node]
    int_to_node_labels = {v:k for k,v in node_labels_to_int.items()}
    for source_pos in A[current_node_pos, :].nonzero()[1]:

        
        id_mat = identity(A.shape[0]).tocsr()
        unnormalized_weights  = (A[current_node_pos, :] + ((1/p)-1)*(A[current_node_pos, :].multiply(A[source_pos, :])) + ((1/q)-1)*(id_mat[source_pos, :])).data
        normalized_weights = unnormalized_weights / unnormalized_weights.sum()
        source = int_to_node_labels[source_pos]
        results.append((source, normalized_weights))


    pd = (current_node, {probabilities_key: dict(results)})    
    return pd

def get_first_travel(node, A, node_labels_to_int, first_travel_key: str = None, p: float = 1, q: float = 1):
    node_pos = node_labels_to_int[node]
    
    ftd = (node, {first_travel_key: (A[node_pos, :] / np.sum(A[node_pos, :])).data})
    
    #ftd[node][first_travel_key] = (A[node, :] / np.sum(A[node, :])).data
    return ftd

def get_neighbors(node, A, node_labels_to_int):

    node_pos = node_labels_to_int[node]
    neighbors = {"neighbors": list(A[node_pos, :].nonzero()[1])}

    return (node, neighbors)

def get_probabilities_chunked(chunk, chunkid, output_folder, *args):
    probs = []
    for node in chunk:
        r = get_probabilities(node, *args)
        probs.append(r)
    
    filename = "".join([str(chunkid), '.pkl'])
    filename = os.path.join(output_folder, filename)
    pickle.dump(probs, open( filename, "wb" ) )
