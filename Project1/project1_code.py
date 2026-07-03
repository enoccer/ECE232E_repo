import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(232)

def question_1a():
	g = ig.Graph.Erdos_Renyi(n=8, p=0.3, directed=False, loops=False)

	print(g.summary())
	print("edges:", g.get_edgelist())
	print("degrees:", g.degree())
	
	return