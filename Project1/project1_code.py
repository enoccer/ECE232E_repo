import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(232)

def question_1a():
	probs = [0.002, 0.006, 0.012, 0.045, 0.1]
	n = 900
	for p in probs:
		g = ig.Graph.Erdos_Renyi(n=n, p=p, directed=False, loops=False)

		deg = np.asarray(g.degree())
		plt.figure(figsize=(7, 4))
		plt.hist(deg, bins=np.arange(deg.max() + 2) - 0.5, density=True,
			edgecolor="white", color="#2B7C9B")
		plt.xlabel("degree")
		plt.ylabel("empirical probability")
		plt.title(f"One ER degree distribution sample (p={p})")
		plt.show()
		
		print("empirical mean:", deg.mean())
		print("theory mean:", (n - 1) * p)
		print("empirical variance:", deg.var())
		print("theory variance:", (n - 1) * p * (1 - p))
		
	return
	
def er_summary(n, p):
    g = ig.Graph.Erdos_Renyi(n=n, p=p, directed=False, loops=False)
    comps = g.connected_components()
    sizes = np.asarray(comps.sizes())
    gcc = comps.giant()
    return {
        "connected": g.is_connected(),
        "gcc_fraction": sizes.max() / n,
        "gcc_diameter": gcc.diameter(),
    }

def question_1b():
	ps = [0.002, 0.006, 0.012, 0.045, 0.1]
	n = 900
	trials = 20
	
	for p in ps:
		rows = [er_summary(n, p) for i in range(trials)]
		prob_connected = np.mean([row["connected"] for row in rows])
		avg_gcc_fraction = np.mean([row["gcc_fraction"] for row in rows])
		print(f"p={p:0.3f} P(conn)~{prob_connected:0.2f} avg GCC frac~{avg_gcc_fraction:0.2f}")
	
	return






