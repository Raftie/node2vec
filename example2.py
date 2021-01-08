import networkx as nx
import numpy as np
from node2vec import Node2Vec

# FILES
EMBEDDING_FILENAME = './embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'

# Create a graph
graph = nx.fast_gnp_random_graph(n=100000, p=0.00002)
print(nx.number_of_nodes(graph))
print(nx.number_of_edges(graph))
print(np.mean([x for i,x in nx.degree(graph)]))

# Precompute probabilities and generate walks
#n2v1 = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, experimental=False)
n2v2 = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=10, workers=4, experimental=True, chunksize=2500, temp_folder='/Users/raf/Dropbox/My Mac (mbpraf.local)/Documents/packages/node2vec_develop/temp', quiet=True)

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
#node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

# Embed
model1 = n2v1.fit(window=10, min_count=1, batch_words=4)
model2 = n2v2.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
model3 = n2v1.fit(window=10, min_count=1, batch_words=4)
# Look for most similar nodes
print(model1.wv.most_similar('2'))  # Output node names are always strings
print(model2.wv.most_similar('2'))  # Output node names are always strings
print(model3.wv.most_similar('2'))

