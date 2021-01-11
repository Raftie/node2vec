import os
from collections import defaultdict

import numpy as np
import networkx as nx
import gensim
from joblib import Parallel, delayed
from tqdm import tqdm
import glob 
import pickle
import logging
from .parallel import parallel_generate_walks, get_first_travel, get_probabilities, get_neighbors, get_probabilities_chunked, get_first_travel_chunked, get_neighbors_chunked

logger = logging.getLogger(__name__)

#Main class
class Node2Vec:
    #Still no idea why this is needed
    FIRST_TRAVEL_KEY = 'first_travel_key'
    #Still no idea why this is needed
    PROBABILITIES_KEY = 'probabilities'
    #Still no idea why this is needed
    NEIGHBORS_KEY = 'neighbors'
                                                                                                      
    #Name for the data attribute containing the edge weights
    WEIGHT_KEY = 'weight'
    #Name for the parameter containing the number of walks per node
    NUM_WALKS_KEY = 'num_walks'
    #Name for the parameter containing the length of the walks
    WALK_LENGTH_KEY = 'walk_length'
    #Name of the parameter containing the p value
    P_KEY = 'p'
    #Name of the parameter containing the q value
    Q_KEY = 'q'

    def __init__(self, graph: nx.Graph, dimensions: int = 128, walk_length: int = 80, num_walks: int = 10, p: float = 1,
                 q: float = 1, weight_key: str = 'weight', workers: int = 1, sampling_strategy: dict = None,
                 quiet: bool = False, temp_folder: str = None, experimental: bool = False, chunksize: int = 0):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param p: Return hyper parameter (default: 1)
        :param q: Inout parameter (default: 1)
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :param workers: Number of workers for parallel execution (default: 1)
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        """
        self.graph = graph
        #Parameter containing the embedding dimensions
        self.dimensions = dimensions
        #Parameter containing the length of individual walks
        self.walk_length = walk_length
        #Parameter containing the number of walks for each individual node
        self.num_walks = num_walks
        #Parameter p
        self.p = p
        #Parameter q
        self.q = q
        #Parameter containing the weight attribute of network edges
        self.weight_key = weight_key
        #Parameter containing the number of workers to work in parallel
        self.workers = workers
        self.quiet = quiet
        self.experimental = experimental
        self.d_graph = defaultdict(dict)
        self.chunksize = chunksize

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"

        if self.experimental:
            self._precompute_probabilities_experimental()
        else:
            self._precompute_probabilities()

        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:

            # Init probabilities dict for first travel
            # self.PROBABILITIES_KEY is just a placeholder for the string 'probabilities'
            # if there is no entry in the d_graph[source] dict, then add a new entry with key "probabilities" which is a dictionary itself
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            #Loop over all neighbors of the source node
            for current_node in self.graph.neighbors(source):

                # This is basically the same step as the one introduced above for the source node
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                #Initialization of lists
                unnormalized_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

                # Save neighbors
                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors

            # Calculate first_travel weights for source
            first_travel_weights = []

            for destination in self.graph.neighbors(source):
                first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))

            first_travel_weights = np.array(first_travel_weights)
            d_graph[source][self.FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()

    def _precompute_probabilities_experimental(self):

        A = nx.adjacency_matrix(self.graph, nodelist=self.graph.nodes())
        node_labels_to_int = dict(zip(self.graph.nodes(), range(self.graph.number_of_nodes())))

        if self.chunksize:
            probs = []
            chunk_generator = (list(self.graph)[i:i+self.chunksize] for i in range(0,len(list(self.graph)),self.chunksize))
            Parallel(n_jobs=self.workers)(delayed(get_probabilities_chunked)(chunk, chunkid, self.temp_folder, A, node_labels_to_int, self.PROBABILITIES_KEY, self.p, self.q, quiet = self.quiet) for chunkid, chunk in tqdm(enumerate(chunk_generator), total=int(len(list(self.graph))/self.chunksize)))
            files = glob.glob(os.path.join(self.temp_folder, '*.pkl'))
            for f in files:
                r = pickle.load( open( f, "rb" ) )
                probs += r    
                os.remove(f) 
            logger.info('finish probabilities')

            first_travels = []
            chunk_generator = (list(self.graph)[i:i+self.chunksize] for i in range(0,len(list(self.graph)),self.chunksize))
            Parallel(n_jobs=self.workers)(delayed(get_first_travel_chunked)(chunk, chunkid, self.temp_folder, A, node_labels_to_int, self.FIRST_TRAVEL_KEY, self.p, self.q) for chunkid, chunk in tqdm(enumerate(chunk_generator), total=int(len(list(self.graph))/self.chunksize)))
            files = glob.glob(os.path.join(self.temp_folder, '*.pkl'))
            for f in files:
                r = pickle.load( open( f, "rb" ) )
                first_travels += r    
                os.remove(f) 
            logger.info('finish first travel probabilties')


            neighbors = []
            chunk_generator = (list(self.graph)[i:i+self.chunksize] for i in range(0,len(list(self.graph)),self.chunksize))
            Parallel(n_jobs=self.workers)(delayed(get_neighbors_chunked)(chunk, chunkid, self.temp_folder, A, node_labels_to_int) for chunkid, chunk in tqdm(enumerate(chunk_generator), total=int(len(list(self.graph))/self.chunksize)))
            files = glob.glob(os.path.join(self.temp_folder, '*.pkl'))
            for f in files:
                r = pickle.load( open( f, "rb" ) )
                neighbors += r    
                os.remove(f) 
            
            logger.info('finish neighbors')



        else:
            # Work node per node
            probs = Parallel(n_jobs=self.workers)(delayed(get_probabilities)(node, A, node_labels_to_int, self.PROBABILITIES_KEY, self.p, self.q) for node in tqdm(self.graph.nodes()))
            first_travels = Parallel(n_jobs=self.workers)(delayed(get_first_travel)(node, A, node_labels_to_int, self.FIRST_TRAVEL_KEY, self.p, self.q) for node in tqdm(self.graph.nodes()))
            neighbors = Parallel(n_jobs=self.workers)(delayed(get_neighbors)(node, A, node_labels_to_int) for node in tqdm(self.graph.nodes()))

        
        
        print("finished probs")

        d_graph = self.d_graph

        prob_dict = dict(probs)
        first_travel_dict = dict(first_travels)
        neighbor_dict = dict(neighbors)

        self.neighbor_dict = neighbor_dict
        d_graph.update(first_travel_dict)
        for k,v in d_graph.items():
            v.update(prob_dict[k])
            v.update(neighbor_dict[k])

        print("finished precompute")

    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks

    def fit(self, **skip_gram_params) -> gensim.models.Word2Vec:
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params['size'] = self.dimensions

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)
