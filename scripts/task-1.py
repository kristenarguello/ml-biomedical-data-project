import itertools
import lzma
import random

import igraph
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn
from Bio import Align
from Bio.Align import substitution_matrices

# decompress the sequences
with lzma.open("data/sequences.txt.xz") as f:
    sequences = f.read()

# save the sequences to a file
with open("processed/sequences.txt", "wb") as f:
    f.write(sequences)

sequences = sequences.decode().split("\n")
sequences = sequences[:-1]


aligner = Align.PairwiseAligner()
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5

# using the BLOSUM62 matrix
substitution_matrices.load()
blosum_matrix = substitution_matrices.load("BLOSUM62")
aligner.substitution_matrix = blosum_matrix

# different lengths of the sequences
aligner.mode = "local"


# matrix with 0s initialized to store the distances
total_sequences = len(sequences)
distance_matrix = np.zeros((total_sequences, total_sequences))


# calculate the pairwise distance matrix
for i in range(total_sequences):
    for j in range(i, total_sequences):
        if i == j:
            distance_matrix[i][j] = 0
        else:
            score = aligner.score(sequences[i], sequences[j])

            # normalize the score by the length of the longest sequence
            normalization_factor = max(len(sequences[i]), len(sequences[j]))
            score = score / normalization_factor

            distance_matrix[i][j] = score
            distance_matrix[j][i] = score


# normalize the usual way to get 0 - 1 values
# min_val = np.min(distance_matrix[np.nonzero(distance_matrix)])
# max_val = np.max(distance_matrix)
# distance_matrix = (distance_matrix - min_val) / (max_val - min_val)

# see the distribution of the distances
df = pandas.DataFrame(distance_matrix.flatten())
desc = df.describe()
max_val = desc.loc["max"][0]

# create a graph from the distance matrix based on a threshold
threshold = 0.5 * max_val
c = range(len(distance_matrix))
edges = [
    (a, b) for a, b in itertools.combinations(c, 2) if distance_matrix[a][b] > threshold
]


# plot the graph
graph = igraph.Graph(n=total_sequences, edges=edges)

# community detection
clusters = graph.community_infomap()
rainbow = seaborn.color_palette("husl", len(clusters))
colours = [rainbow[clusters.membership[v]] for v in graph.vs.indices]
graph.vs["color"] = colours

graph.es["color"] = "#99999933"
# change colour of the internal community edges
colours = graph.vs["color"]
for e in graph.es:
    if colours[e.source] == colours[e.target]:
        e.update_attributes({"color": colours[e.source]})

# scale node size by degree
degrees = [d * 0.2 + 5 for d in graph.degree()]
# degrees = list(set(graph.degree()))
# qcut = pandas.qcut(degrees, 8, labels=False)
# print(qcut)
# he will talk about this on canvas

random.seed(42)
igraph.plot(
    graph,
    "results/1_network.png",
    layout=graph.layout("fr"),
    bbox=(1000, 1000),
    vertex_size=degrees,
)


# topological analysis table
values = []
values.append(("Diameter", graph.diameter()))
values.append(("Girth", graph.girth()))
values.append(("Radius", graph.radius()))
values.append(("Average Path Length", graph.average_path_length()))

values.append(("Density", graph.density()))
degrees = np.mean(graph.degree())
values.append(("Average Degree", degrees))
values.append(("Assortivity Degree", graph.assortativity_degree()))

# clustering coefficient
values.append(("Transitivity Average", graph.transitivity_avglocal_undirected()))
values.append(("Transitivity Undirected", graph.transitivity_undirected()))

# Create a figure and axis
fig, ax = plt.subplots()

# Hide the axis
ax.axis("off")

# Create a table
table = ax.table(
    cellText=values,
    colLabels=["Network Topological Attribute ", "Value"],
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.auto_set_column_width(col=[0, 1])
fig.savefig("results/1_topological_analysis.png", bbox_inches="tight", dpi=300)
