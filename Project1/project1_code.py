import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import random

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
	summaries = {}
	
	for p in ps:
		rows = [er_summary(n, p) for i in range(trials)]
		prob_connected = np.mean([row["connected"] for row in rows])
		avg_gcc_fraction = np.mean([row["gcc_fraction"] for row in rows])
		avg_gcc_diam = np.mean([row["gcc_diameter"] for row in rows])
		print(f"p={p:0.3f} P(conn)~{prob_connected:0.2f}, " + 
			  f"avg GCC frac~{avg_gcc_fraction:0.2f}, " + 
			  f"avg GCC diam~{avg_gcc_diam:0.2f}")
		# Cache for outside loop
		summaries[p] = rows
	
	# Find an instance of a disconnected ER network
	found = False
	for p, summary in summaries.items():
		for run in range(trials):
			if summary[run]["connected"] == False:
				diam = summary[run]["gcc_diameter"]
				print(f"GCC diameter={diam} for a disconnected network (p={p}) instance")
				return
	return
	
def threshold_condition_index(x_vals, y_vals, threshold):
	"""Finds first index by which threshold condition is satisfied."""
	
	condition = y_vals > threshold
	if np.any(condition):
		# np.argmax() returns the index of the FIRST 'True' value
		first_index = np.argmax(condition)
		
		x_val = x_vals[first_index]
		y_val = y_vals[first_index]
		
		print(f"First X value above Y > {threshold}: {x_val:.5f}")
		print(f"Actual Y value at that point: {y_val:.5f}")
		return x_val, y_val
	else:
		print(f"The Y value never reaches the threshold of {threshold}")
		return None, None

def question_1c():
	n = 900
	ps = np.linspace(0, 0.005, 25)
	n_networks = 100
	gcc_sizes = []
	for p in ps:
		networks = [er_summary(n, p) for i in range(n_networks)]
		avg_norm_gcc_size = np.mean([row["gcc_fraction"] for row in networks])
		gcc_sizes.append(avg_norm_gcc_size)
	
	# Scatter
	plt.scatter(ps, gcc_sizes, label='Data points')
	
	# Trend line
	x_data, y_data = ps, gcc_sizes
	coefficients = np.polyfit(x_data, y_data, deg=4)
	poly_function = np.poly1d(coefficients)
	x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
	y_fitted_curve = poly_function(x_smooth)
	
	plt.plot(x_smooth, y_fitted_curve, 
			 color='limegreen', linestyle='--', linewidth=2, label='Trend line')
	
	plt.title('Normalized GCC size v. probability')
	plt.xlabel('Probability p')
	plt.ylabel('Normalized GCC size')
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.legend()
	plt.show()
	
	# For answering parts i and ii
	gcc_vals = np.array(gcc_sizes)
	p_i, gcc_i = threshold_condition_index(ps, gcc_vals, 0.005)
	p_ii, gcc_ii = threshold_condition_index(ps, gcc_vals, 0.98)
	print(f"\nPart (i) p = {p_i:.5f}, gcc_size = {gcc_i:.5f}")
	print(f"Part (ii) p = {p_ii:.5f}, gcc_size = {gcc_ii:.5f}")
	
def simulate_across_node_sizes(nodes, c, n_networks):
	"""
	Randomly generates `n_networks` using the average node value `c`.
	The random network generation occurs across a range of node sizes `nodes`.
	"""
	
	gcc_sizes = []
	for n in nodes:
		p = c/n  # Edge-formation probability
		networks = [er_summary(n, p) for i in range(n_networks)]
		avg_norm_gcc_size = np.mean([row["gcc_fraction"] for row in networks])
		gcc_sizes.append(avg_norm_gcc_size)
		
	return gcc_sizes
	
def question_1d(c):
	"""c is the average degree of nodes."""
	nodes = np.linspace(100, 1e4, 25).astype(int)
	n_networks = 100
	gcc_sizes = simulate_across_node_sizes(nodes, c, n_networks)

	# Scatter
	plt.scatter(nodes, gcc_sizes, label='Data points')
	
	plt.title(f"Normalized GCC size v. nodes (c={c})")
	plt.xlabel('Number of nodes n')
	plt.ylabel('Normalized GCC size')
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.legend()
	plt.show()
	
def log_fit(x_data, y_data, x_start, x_end):
	"""Generates a logarithmic fit."""
	
	# Fit a line to: y = a * ln(x) + b
	a, b = np.polyfit(np.log(x_data), y_data, 1)
	
	# Make a smooth curve line to plot
	x_smooth = np.linspace(x_start, x_end, 200)
	y_fitted = a * np.log(x_smooth) + b
	
	return x_smooth, y_fitted
	
def question_1d_logfit(c):
	"""c is the average degree of nodes."""
	n_start, n_end = 100, 1e4
	nodes = np.linspace(n_start, n_end, 25).astype(int)
	n_networks = 50
	
	# Random network simulations
	gcc_sizes = simulate_across_node_sizes(nodes, c, n_networks)

	# Scatter
	plt.scatter(nodes, gcc_sizes, label='Simulation data')
	
	# Logarithm fit
	x_data, y_data = nodes, gcc_sizes
	x_smooth, y_fitted = log_fit(x_data, y_data, n_start, n_end)
	
	# Plot fit
	plt.plot(x_smooth, y_fitted, color='navy', linestyle='--', linewidth=2.5,
             label='Fitted Line')
	
	plt.title(f"Normalized GCC size v. nodes (c={c})")
	plt.xlabel('Number of nodes n')
	plt.ylabel('Normalized GCC size')
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.legend()
	plt.show()

def question_1d_part_iv():
	c_vals = [1.15, 1.25, 1.35]
	n_start, n_end = 100, 1e4
	nodes = np.linspace(n_start, n_end, 25).astype(int)
	n_networks = 100
	plt.figure(figsize=(8, 5))
	
	for c in c_vals:
		print(f"Generating random networks for c={c}")
		# Random network simulations
		gcc_sizes = simulate_across_node_sizes(nodes, c, n_networks)
		
		# Logarithm fit
		x_data, y_data = nodes, gcc_sizes
		x_smooth, y_fitted = log_fit(x_data, y_data, n_start, n_end)

		# Plot fit
		plt.plot(x_smooth, y_fitted, linestyle='--', linewidth=2.5,
				 label=f'Fitted Line (c={c})')

	plt.title(f"Normalized GCC size v. nodes (c={c})")
	plt.xlabel('Number of nodes n')
	plt.ylabel('Normalized GCC size')
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.legend()
	plt.show()


###################################
#                                 #
#   2. Random Walk on Networks    #
#                                 #
###################################

def random_walk(g, num_steps, start_node=None, transition_matrix=None, rng=None):
    """
    Simulate a random walk and return the path of visited nodes.

    Suggested return value:
    - A NumPy array or list containing the starting node and each subsequent node.
    """
    
    if rng is None:
        rng = np.random.default_rng()
    
    n_nodes = g.vcount()
    if start_node is None:
        start_node = rng.choice(n_nodes)
        
    # Initialize path with starting node
    path = [start_node]
    current_node = start_node

    # Random walk
    for _ in range(num_steps):
        neighbors = g.neighbors(current_node)
        
        # If the node has neighbors, pick one randomly and move
        if neighbors:
            current_node = rng.choice(neighbors)
            path.append(current_node)
            
        else:
        	# If walker hits dead end, it's stuck at current node for remaining steps
        	path.append(current_node)

    return path

def estimate_visit_probabilities(path, n_nodes):
    """
    Estimate node visit probabilities from a random-walk path.

    Hints:
    - np.bincount can count visits to integer-labeled nodes.
    - The probabilities should sum to 1.
    """
    
    # Count how many visits to every node in graph
    # minlength ensures output array has slot for every node (0 to n-1)
    visit_counts = np.bincount(path, minlength=n_nodes)
    visit_probs = visit_counts / len(path)
    return visit_probs
    
def plot_visit_probabilities(visit_probs, n_nodes, n_steps):
	
	# Plot probabilities
	plt.figure(figsize=(10, 5))
	node_indices = np.arange(n_nodes)
	plt.plot(node_indices, visit_probs, color='darkblue', alpha=0.7, label='Visit Probability')
	plt.fill_between(node_indices, visit_probs, color='skyblue', alpha=0.4)
	
	# Formatting
	plt.title(f'Empirical Node Visit Probabilities via Random Walk ({n_steps} Steps)', 
	          fontsize=13, fontweight='bold')
	plt.xlabel('Node Index', fontsize=11)
	plt.ylabel('Probability', fontsize=11)
	plt.xlim(0, n_nodes)
	plt.ylim(bottom=0) # Probabilities cannot be negative
	plt.grid(True, linestyle='--', alpha=0.3)
	plt.legend()
	
	plt.show()

def question2_3a():
	
	# Directed random network
	n = 900
	m = 4
	g1 = ig.Graph.Barabasi(n=n, m=m, directed=True)
	g2 = ig.Graph.Barabasi(n=n, m=m, directed=True)
	
	# Shuffle indices of nodes
	shuffled_indices_g1 = random.sample(range(n), n)
	shuffled_indices_g2 = random.sample(range(n), n)
	g1_shuffled = g1.permute_vertices(shuffled_indices_g1)
	g2_shuffled = g2.permute_vertices(shuffled_indices_g2)
	
	# Merge networks by adding the second graph's edges to the first graph
	g2_shuffled_edges = g2_shuffled.get_edgelist()
	g1_shuffled.add_edges(g2_shuffled_edges)
	
	print(f"Graph 1 edges: {g1.ecount()}")
	print(f"Graph 2 edges: {g2.ecount()}")
	print(f"Total nodes: {g1_shuffled.vcount()}")
	print(f"Total merged edges: {g1_shuffled.ecount()}")
	
	# Random walk
	num_steps = 5000
	path = random_walk(g1_shuffled, num_steps=num_steps, rng=rng)

	# Visit probabilities to every node in graph
	visit_probs = estimate_visit_probabilities(path, n)
	
	# Plot probabilities
	plot_visit_probabilities(visit_probs, n, num_steps)

	# See the relationship between node in-degree and visit probability
	# Get the in-degree of every node
	in_degrees = np.array(g1_shuffled.indegree())
	
	# Find the node with the highest in-degree and highest visit probability
	print(f"Node with highest in-degree: {np.argmax(in_degrees)}")
	print(f"Node with highest visit probability: {np.argmax(visit_probs)}")



















