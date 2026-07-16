import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import random

rng = np.random.default_rng(232)

def question_1_1():
    g = ig.Graph.Read_Edgelist("facebook_combined.txt", directed=False)
    print("Number of Nodes = ", g.vcount())
    print("Number of Edges = ", g.ecount())
    return

#def question_1_2():
    