import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import random

rng = np.random.default_rng(232)
random.seed(42)

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
	p_max = 0.005
	ps = np.linspace(0, p_max, 25)
	n_networks = 100
	gcc_sizes = []
	for p in ps:
		networks = [er_summary(n, p) for i in range(n_networks)]
		avg_norm_gcc_size = np.mean([row["gcc_fraction"] for row in networks])
		gcc_sizes.append(avg_norm_gcc_size)
	
	# Plots
	fig, ax = plt.subplots()
	ax.scatter(ps, gcc_sizes, label=f'Average of {n_networks} runs')
	ax.plot(ps, gcc_sizes, color='blue', linestyle='--', linewidth=2, label='Empirical')
	
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

	# Plot
	fig, ax = plt.subplots()
	ax.scatter(nodes, gcc_sizes, label=f'Average of {n_networks} runs')
	ax.plot(nodes, gcc_sizes, color='blue', linestyle='--', linewidth=2, label='Empirical')
	
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


def pa_model_summary(n, m):
    g_pa = ig.Graph.Barabasi(n, m, directed=False)
    print(g_pa.summary())
    print("connected:", g_pa.is_connected())
    print("max degree:", max(g_pa.degree()))
    print("mean degree:", np.mean(g_pa.degree()))
    print("degree variance:", np.var(g_pa.degree()))

    assortativity = g_pa.assortativity_degree(directed=False)
    communities = g_pa.community_fastgreedy().as_clustering()
    print(
        "edges:", g_pa.ecount(),
        ", assortativity:", assortativity,
        ", modularity:", round(communities.modularity, 3),
    )

def pa_degree_distribution_plot(n, m):
    g_pa = ig.Graph.Barabasi(n, m, directed=False)
    deg = np.asarray(g_pa.degree())
    k, counts = np.unique(deg, return_counts=True)
    prob = counts / counts.sum()

    mask = (k >= 1 ) & (counts > 0)
    slope, intercept = np.polyfit(np.log(k[mask]), np.log(prob[mask]), 1)
    print("Nodes: ", n, ", m = ", m)
    print("rough log-log slope:", slope)

    plt.figure(figsize=(7, 4))
    plt.loglog(k, prob, "o", label="empirical")
    plt.loglog(k, np.exp(intercept) * k**slope, label="power law")
    plt.legend()
    plt.xlabel("degree")
    plt.ylabel("empirical probability")
    plt.title("One PA degree distribution sample")
    plt.show()

def sample_neighbor_degrees(g, samples =5000, rng=None):
    rng = np.random.default_rng(rng) if rng is None else rng
    seen = []
    for i in range(samples):
        v = int(rng.choice(g.vcount()))
        nbrs = g.neighbors(v)
        if len(nbrs) > 0:
            u = int(rng.choice(nbrs))
            seen.append(g.degree(u))
    return np.asarray(seen)

def average_degree_over_time(n=1050, m=1, trials=50):
    total = np.zeros(n)
    for i in range(trials):
        g = ig.Graph.Barabasi(n=n, m=m, directed=False)
        deg = np.asarray(g.degree())
        deg_by_age = deg[::-1]
        total += deg_by_age
    return total / trials

def network_plot(g):
    try:
        communities = g.community_fastgreedy().as_clustering()
        print("Used Fastgreedy")
    except Exception as e:
        print(f"Fastgreedy failed ({e}), Using Walktrap")
        communities = g.community_walktrap().as_clustering()
    modularity = communities.modularity
    print("Modularity: ", round(modularity, 3))  
    fig, ax = plt.subplots(figsize=(6, 6))
    ig.plot(communities, target=ax, vertex_size=15, vertex_color="lightblue", edge_width=0.5, mark_groups=True)
    plt.show()

def attachment_probabilities(degrees, ages, alpha, beta, a, b, c, d):
    degrees = np.asarray(degrees, dtype=float)
    ages = np.asarray(ages, dtype=float)
    weights = (c * degrees**alpha + a) * (d * ages**beta + b)
    return weights / weights.sum()

def pa_age_network(n, m, alpha, beta, a, b, c, d):
    #initialize
    degree = np.zeros(n, dtype = float)
    edges = []

    degree[0] += 1
    degree[1] += 1
    edges.append((0,1))

    for i in range(2, n):
        ages = (i - np.arange(i)).astype(float)
        ki = degree[:i]

        prob = attachment_probabilities(ki, ages, alpha = 1, beta = -1, a = 1, b = 0, c = 1, d = 1)

        targets = rng.choice(i, size = m, replace = False, p = prob)
        for target in targets:
            edges.append((i, target))
            degree[i] += 1
            degree[target] += 1

    return ig.Graph(n = n, edges = edges, directed = False)

def question_2a():
    is_connected = []
    for i in range(0, 20):
        g = ig.Graph.Barabasi(n=1050, m=1, directed=False)
        is_connected.append(g.is_connected())
    print(is_connected)
    print("Yes, this network is always connected")  
    return

def question_2b():
    pa_model_summary(1050, 1) 
    print("Assortativity is defined as a measure of the tendency for nodes to connect to other nodes with similar properties")
    print("Assortativity is positive if similar nodes tend to connect to each other, and negative otherwise")
    return

def question_2c():
    pa_model_summary(10500, 1) 
    return

def question_2d():
    pa_degree_distribution_plot(1050, 1)
    pa_degree_distribution_plot(10500, 1)
    return
    
def question_2e(n):
    g_pa = ig.Graph.Barabasi(n, m=1, directed=False)
    neighbor_degrees = sample_neighbor_degrees(g_pa, rng=rng)
    k, counts = np.unique(neighbor_degrees, return_counts=True)
    prob = counts / counts.sum()
    
    mask = (k >= 1 ) & (counts > 0)
    slope, intercept = np.polyfit(np.log(k[mask]), np.log(prob[mask]), 1)
    print("rough log-log slope:", slope)
    
    plt.figure(figsize=(7, 4))
    plt.loglog(k, prob, "o", label="empirical")
    plt.loglog(k, np.exp(intercept) * k**slope, label="power law")
    plt.legend()
    plt.xlabel("degree")
    plt.ylabel("empirical probability")
    plt.title("One PA degree distribution sample")
    plt.show()
    return

def question_2f():
    avg_deg = average_degree_over_time()
    plt.plot(avg_deg)
    plt.xlabel("Birth time")
    plt.ylabel("Average degree")
    plt.title("Average degree by birth time")
    plt.show()
    return

def question_2g(m):
    is_connected = []
    for i in range(0, 20):
        g = ig.Graph.Barabasi(n=1050, m=m, directed=False)
        is_connected.append(g.is_connected())
    print(is_connected)
    print("Yes, this network is always connected")  

    nodes = [1050, 10500]
    for node in nodes: 
        pa_model_summary(node, m) 
        pa_degree_distribution_plot(node, m)
        g_pa = ig.Graph.Barabasi(node, m=m, directed=False)
        neighbor_degrees = sample_neighbor_degrees(g_pa, rng=rng)
        k, counts = np.unique(neighbor_degrees, return_counts=True)
        prob = counts / counts.sum()
    
        mask = (k >= 1 ) & (counts > 0)
        slope, intercept = np.polyfit(np.log(k[mask]), np.log(prob[mask]), 1)
        print("rough log-log slope:", slope)
    
        plt.figure(figsize=(7, 4))
        plt.loglog(k, prob, "o", label="empirical")
        plt.loglog(k, np.exp(intercept) * k**slope, label="power law")
        plt.legend()
        plt.xlabel("degree")
        plt.ylabel("empirical probability")
        plt.title("One PA degree distribution sample")
        plt.show()

        avg_deg = average_degree_over_time(n=node, m=m, trials=50)
        plt.plot(avg_deg)
        plt.xlabel("Birth time")
        plt.ylabel("Average degree")
        plt.title("Average degree by birth time")
        plt.show()
    return

def question_2h():
    g_pa = ig.Graph.Barabasi(n=1050, m=1, directed=False)
    deg_seq = np.array(g_pa.degree())
    new_g = ig.Graph.Degree_Sequence(out = deg_seq, method="configuration")
    print("Original Network")
    network_plot(g_pa)
    print(g_pa.summary())
    print("connected:", g_pa.is_connected())
    print("max degree:", max(g_pa.degree()))
    print("mean degree:", np.mean(g_pa.degree()))
    print("degree variance:", np.var(g_pa.degree()))
    assortativity = g_pa.assortativity_degree(directed=False)
    print(
        "edges:", g_pa.ecount(),
        ", assortativity:", assortativity)
    print("\nNew Network")
    network_plot(new_g)
    print(new_g.summary())
    print("connected:", new_g.is_connected())
    print("max degree:", max(new_g.degree()))
    print("mean degree:", np.mean(new_g.degree()))
    print("degree variance:", np.var(new_g.degree()))
    new_assortativity = new_g.assortativity_degree(directed=False)
    print(
        "edges:", new_g.ecount(),
        ", assortativity:", new_assortativity)
    return


def question_3():
    print("Part A")
    g = pa_age_network(n=1050, m=1, alpha=1.0, beta= -1.0, a=1, b=0, c=1, d=1)
    degrees = np.asarray(g.degree())
    k, counts = np.unique(degrees, return_counts=True)
    prob = counts / counts.sum()
    
    mask = (k >= 1 ) & (counts > 0)
    slope, intercept = np.polyfit(np.log(k[mask]), np.log(prob[mask]), 1)
    print("rough log-log slope:", slope)
    print("Power Law Exponent is roughly: ", -slope)
    
    plt.figure(figsize=(7, 4))
    plt.loglog(k, prob, "o", label="empirical")
    plt.loglog(k, np.exp(intercept) * k**slope, label="power law")
    plt.legend()
    plt.xlabel("degree")
    plt.ylabel("empirical probability")
    plt.title("One PA degree distribution sample")
    plt.show()
    #power_law_exponent = ig.power_law_fit(degrees)

    print("Part B")
    communities = g.community_fastgreedy().as_clustering()
    modularity = communities.modularity
    print("Modularity: ", round(modularity, 3))  

###################################
#                                 #
#   2. Random Walk on Networks    #
#                                 #
###################################

def simulate_random_walks(g, num_trials=1000, max_steps=30):
    """
    Returns:
        avg_distance: average shortest-path distance from start at each step
        var_distance: variance of shortest-path distance at each step
        final_degree: degree of the node reached at the final step
    """

    n_nodes = g.vcount()
    distance_tracks = np.zeros((num_trials, max_steps + 1))
    final_degrees = []
    print(f"Simulating {num_trials} random walk trials")

    for trial in range(num_trials):
        start_node = rng.choice(n_nodes)
        current_node = start_node
        distance_tracks[trial, 0] = 0
        shortest_paths = g.distances(source=start_node)[0]
        
        for t in range(1, max_steps + 1):
            neighbors = g.neighbors(current_node)

            if neighbors:
                current_node = rng.choice(neighbors)
            distance_tracks[trial, t] = shortest_paths[current_node]
        final_degrees.append(g.degree(current_node))
        
    avg_distances = np.mean(distance_tracks, axis=0)
    var_distances = np.mean((distance_tracks - avg_distances) ** 2, axis=0)

    return avg_distances, var_distances, final_degrees

def question2_1():
    p = 0.015
    max_steps = 25
    steps = np.arange(max_steps + 1)

    print("\n (a) Generate ER graph with N = 900 \n")

    g_900 = ig.Graph.Erdos_Renyi(n=900, p=p, directed=False, loops=False)

    diameter900 = g_900.diameter()

    print(f"Nodes     : {g_900.vcount()}")
    print(f"Edges     : {g_900.ecount()}")
    print(f"Diameter  : {diameter900}")

    avg900, var900, final_degrees = simulate_random_walks(g_900, num_trials=2000, max_steps=max_steps)

    print("\n (b) Average distance and variance \n")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(steps, avg900, marker='o', linewidth=2)
    plt.title(r"$\langle s(t)\rangle$ ($N=900$)")
    plt.xlabel("Step")
    plt.ylabel("Average Distance")
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(steps, var900, marker='o', linewidth=2, color="red")
    plt.title(r"$\sigma^2(t)$ ($N=900$)")
    plt.xlabel("Step")
    plt.ylabel("Variance")
    plt.grid(True)
    plt.show()

    print("\n (c) Degree distribution comparison \n")

    graph_degrees = g_900.degree()
    max_deg = max(max(graph_degrees), max(final_degrees))
    bins = np.arange(max_deg + 2) - 0.5

    plt.hist(graph_degrees, bins=bins, density=True, alpha=0.5, label="Original Graph")
    plt.hist(final_degrees, bins=bins, density=True, alpha=0.6, label="Random Walk End Nodes")
    plt.title("Degree Distribution Comparison")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Average Graph Degree      : {np.mean(graph_degrees):.3f}")
    print(f"Average End-Node Degree   : {np.mean(final_degrees):.3f}")
    print("The degree distribution of the nodes reached at the end of the random walk is biased toward "
          "higher-degree nodes compared to the degree distribution of the original graph. In an unbiased "
          "random walk on an undirected network, the probability of visiting a node in the long run is "
          "proportional to its degree, because nodes with more incident edges are more likely to be visited.")

    print("\n (d) Repeat for N = 9000 \n")

    g_9000 = ig.Graph.Erdos_Renyi(n=9000, p=p, directed=False, loops=False)
    diameter9000 = g_9000.diameter()

    print(f"Nodes     : {g_9000.vcount()}")
    print(f"Edges     : {g_9000.ecount()}")
    print(f"Diameter  : {diameter9000}")

    avg9000, var9000, _ = simulate_random_walks(g_9000, num_trials=1000, max_steps=max_steps)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(steps, avg900, marker='o', linewidth=2, label="N=900")
    plt.plot(steps, avg9000, marker='s', linewidth=2, label="N=9000")
    plt.title("Average Distance Comparison")
    plt.xlabel("Step")
    plt.ylabel("Average Distance")
    plt.legend()
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(steps, var900, marker='o', linewidth=2, label="N=900")
    plt.plot(steps, var9000, marker='s', linewidth=2, label="N=9000")
    plt.title("Variance Comparison")
    plt.xlabel("Step")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nDiameter Comparison")
    print(f"N = 900  : {diameter900}")
    print(f"N = 9000 : {diameter9000}")

    print("When the network size was increased from 900 to 9000 nodes while keeping the edge probability fixed "
          "at p=0.015, the diameter decreased from 5 to 3. Although the larger graph contains ten times as many "
          "nodes, it is also much denser because the expected degree increases from approximately 13.5 to 135. The "
          "increased connectivity creates many additional shortcuts between nodes, reducing the maximum shortest-path distance in the network.")

def question2_2():
    m = 1
    max_steps = 25
    steps = np.arange(max_steps + 1)

    print("\n (a) Generate PA graph with N = 900, M = 1 \n")

    g_900 = ig.Graph.Barabasi(n=900, m=m, directed=False)

    diameter900 = g_900.diameter()

    print(f"Nodes     : {g_900.vcount()}")
    print(f"Edges     : {g_900.ecount()}")
    print(f"Diameter  : {diameter900}")

    avg900, var900, final_degrees = simulate_random_walks(g_900, num_trials=2000, max_steps=max_steps)

    print("\n (b) Average distance and variance \n")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(steps, avg900, marker='o', linewidth=2)
    plt.title(r"$\langle s(t)\rangle$ ($N=900$)")
    plt.xlabel("Step")
    plt.ylabel("Average Distance")
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(steps, var900, marker='o', linewidth=2, color="red")
    plt.title(r"$\sigma^2(t)$ ($N=900$)")
    plt.xlabel("Step")
    plt.ylabel("Variance")
    plt.grid(True)
    plt.show()

    print("\n (c) Degree distribution comparison \n")

    graph_degrees = g_900.degree()
    max_deg = max(max(graph_degrees), max(final_degrees))
    bins = np.arange(max_deg + 2) - 0.5

    plt.hist(graph_degrees, bins=bins, density=True, alpha=0.5, label="Original Graph")
    plt.hist(final_degrees, bins=bins, density=True, alpha=0.6, label="Random Walk End Nodes")
    plt.title("Degree Distribution Comparison")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Average Graph Degree      : {np.mean(graph_degrees):.3f}")
    print(f"Average End-Node Degree   : {np.mean(final_degrees):.3f}")
    print("With the original having an average degree of 1.998 and the random having 4.839, there is a "
          "large bias to higher degree nodes this is an effect of having a stationary distribution on an "
          "undirected network where the probability of visit is directly proportional to its degree. In "
          "this graph the vast majority of nodes are leaves with a degree of 1 (over 60%), the random walker drops the probability of landing on degree-1 nodes to under 30%.")

    print("\n (d) Repeat for N = 9000 \n")

    g_9000 = ig.Graph.Barabasi(n=9000, m=m, directed=False)
    diameter9000 = g_9000.diameter()

    print(f"Nodes     : {g_9000.vcount()}")
    print(f"Edges     : {g_9000.ecount()}")
    print(f"Diameter  : {diameter9000}")

    avg9000, var9000, _ = simulate_random_walks(g_9000, num_trials=1000, max_steps=max_steps)

    print("\n Repeat for N = 90 \n")

    g_90 = ig.Graph.Barabasi(n=90, m=m, directed=False)
    diameter90 = g_90.diameter()

    print(f"Nodes     : {g_90.vcount()}")
    print(f"Edges     : {g_90.ecount()}")
    print(f"Diameter  : {diameter90}")

    avg90, var90, _ = simulate_random_walks(g_90, num_trials=1000, max_steps=max_steps)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(steps, avg90, marker='^', linewidth=2, label="N=90")
    plt.plot(steps, avg900, marker='o', linewidth=2, label="N=900")
    plt.plot(steps, avg9000, marker='s', linewidth=2, label="N=9000")
    plt.title("Average Distance Comparison")
    plt.xlabel("Step")
    plt.ylabel("Average Distance")
    plt.legend()
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(steps, var90, marker='^', linewidth=2, label="N=90")
    plt.plot(steps, var900, marker='o', linewidth=2, label="N=900")
    plt.plot(steps, var9000, marker='s', linewidth=2, label="N=9000")
    plt.title("Variance Comparison")
    plt.xlabel("Step")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nDiameter Comparison")
    print(f"N = 90  : {diameter90}")
    print(f"N = 900  : {diameter900}")
    print(f"N = 9000 : {diameter9000}")

    print("PA model fixes m = 1, locking the average degree at a sparse baseline of k ~= 2 for all network  "
          "sizes. This network remains sparse as it grows, with the diameter increasing substantially with "
          "the number of nodes, expanding from 12 to 27 as grows 90 to 90000, shifting also the avg "
          "distance and variance. The expanding diameter also causes the random walker significantly more "
          "steps to explore the larger periphery and the distance keeps increasing showing that the system has not yet reached global equilibrium.")

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
	
def page_rank_network(n, m):
	"""Creates a PageRank network."""

	# Directed random network
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
	
	#print(f"Graph 1 edges: {g1.ecount()}")
	#print(f"Graph 2 edges: {g2.ecount()}")
	#print(f"Total nodes: {g1_shuffled.vcount()}")
	#print(f"Total merged edges: {g1_shuffled.ecount()}")
	
	return g1_shuffled
	
def lock_in_rng():
	"""Lock in rng for reproducibility."""
	
	global rng     
	rng = np.random.default_rng(232)     
	random.seed(42)     
	ig.set_random_number_generator(random.Random(42))

def question2_3a():

	# Lock in rng for reproducibility
	lock_in_rng()
	
	# Directed random network
	n = 900
	m = 4
	pagerank = page_rank_network(n, m)
	
	# Random walk
	num_steps = 5000
	path = random_walk(pagerank, num_steps=num_steps, rng=rng)

	# Visit probabilities to every node in graph
	visit_probs = estimate_visit_probabilities(path, n)
	
	# Plot probabilities
	plot_visit_probabilities(visit_probs, n, num_steps)

	# See the relationship between node in-degree and visit probability
	# Get the in-degree of every node
	in_degrees = np.array(pagerank.indegree())
	
	# Find the node with the highest in-degree and highest visit probability
	print(f"Node with highest in-degree: {np.argmax(in_degrees)}")
	print(f"Node with highest visit probability: {np.argmax(visit_probs)}")

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


def page_rank_network(n, m):
    """Creates a PageRank network."""

    # Directed random network
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

    #print(f"Graph 1 edges: {g1.ecount()}")
    #print(f"Graph 2 edges: {g2.ecount()}")
    #print(f"Total nodes: {g1_shuffled.vcount()}")
    #print(f"Total merged edges: {g1_shuffled.ecount()}")

    return g1_shuffled


def lock_in_rng():
    """Lock in rng for reproducibility."""

    global rng
    rng = np.random.default_rng(232)
    random.seed(42)
    ig.set_random_number_generator(random.Random(42))


def question2_3a():

    # Lock in rng for reproducibility
    lock_in_rng()

    # Directed random network
    n = 900
    m = 4
    pagerank = page_rank_network(n, m)

    # Random walk
    num_steps = 5000
    path = random_walk(pagerank, num_steps=num_steps, rng=rng)

    # Visit probabilities to every node in graph
    visit_probs = estimate_visit_probabilities(path, n)

    # Plot probabilities
    plot_visit_probabilities(visit_probs, n, num_steps)

    # See the relationship between node in-degree and visit probability
    # Get the in-degree of every node
    in_degrees = np.array(pagerank.indegree())

    # Find the node with the highest in-degree and highest visit probability
    print(f"Node with highest in-degree: {np.argmax(in_degrees)}")
    print(f"Node with highest visit probability: {np.argmax(visit_probs)}")

def random_walk_with_teleportation(
    g,
    num_steps,
    start_node=None,
    alpha=0.2,
    teleport_probs=None,
    transition_matrix=None,
    rng=None,
):
    """
    Simulate a random walk with teleportation.

    Suggested behavior:
    - With probability alpha, choose the next node from teleport_probs.
    - Otherwise, follow the graph transition probabilities.
    """
    
    if rng is None:
        rng = np.random.default_rng()

    n_nodes = g.vcount()
    if start_node is None:
        start_node = rng.choice(n_nodes)

    if teleport_probs is None:
        # If not given, then uniform probability to each node
        teleport_probs = np.ones(n_nodes) / n_nodes

    # Initialize path with starting node
    path = [start_node]
    current_node = start_node

    # Random walk with teleportation
    for _ in range(num_steps):
        if rng.random() < alpha:
            # Teleport to a completely new node
            current_node = rng.choice(n_nodes, p=teleport_probs)
        else:
            neighbors = g.neighbors(current_node)
            if neighbors:
                # If the current node has neighbors, pick one randomly and move
                current_node = rng.choice(neighbors)
            else:
                # If at a dead end, force a teleportation
                current_node = rng.choice(n_nodes, p=teleport_probs)

        path.append(current_node)
    
    return path

def question2_3b(alpha=0.2, num_steps=5000):

	# Lock in rng for reproducibility
	lock_in_rng()
	
	# Generate PageRank network
	n = 900
	m = 4
	pagerank = page_rank_network(n, m)
	
	# Random walk with teleportation
	path = random_walk_with_teleportation(pagerank, 
										  num_steps, 
										  alpha=alpha,
										  rng=rng)

	# Visit probabilities to every node in graph
	visit_probs = estimate_visit_probabilities(path, n)
	
	# Plot probabilities
	plot_visit_probabilities(visit_probs, n, num_steps)
	
	# See the relationship between node in-degree and visit probability
	in_degrees = np.array(pagerank.indegree())

	# Find the node with the highest in-degree and highest visit probability
	print(f"Node with highest in-degree: {np.argmax(in_degrees)}")
	print(f"Node with highest visit probability: {np.argmax(visit_probs)}")
	print(f"alpha={alpha}, num_steps={num_steps}")

def question2_4a(alpha=0.2, num_steps=5000):

	# Lock in rng for reproducibility
	lock_in_rng()

	# Generate PageRank network
	n = 900
	m = 4
	pagerank_net = page_rank_network(n, m)
	
	# Use PageRank score for every node to generate teleport probabilities
	# Damping factor is the directed walk (1 - alpha)
	pr_scores = pagerank_net.pagerank(directed=True, damping=1-alpha)
	teleport_probs = np.array(pr_scores)
	print(f"Teleport probabilities sum: {np.sum(teleport_probs):.4f}")
	
	# Random walk with teleportation
	path = random_walk_with_teleportation(pagerank_net, 
										  num_steps, 
										  alpha=alpha,
										  teleport_probs=teleport_probs,
										  rng=rng)

	# Visit probabilities to every node in graph
	visit_probs = estimate_visit_probabilities(path, n)
	
	# Plot probabilities
	plot_visit_probabilities(visit_probs, n, num_steps)
	
	# See the relationship between node in-degree and visit probability
	in_degrees = np.array(pagerank_net.indegree())

	# Find the node with the highest in-degree and highest visit probability
	print(f"Node with highest in-degree: {np.argmax(in_degrees)}")
	print(f"Node with highest visit probability: {np.argmax(visit_probs)}")

def find_nodes_with_median_pagerank(pagerank_net, alpha):

	# Compute PageRank scores
	pr_scores = pagerank_net.pagerank(directed=True, damping=1-alpha)
	
	# Sort PageRank scores and their node indices from lowest to highest
	sorted_node_indices = np.argsort(pr_scores)

	# Middle index of our sorted list
	n_nodes = pagerank_net.vcount()
	mid_point = n_nodes // 2
	
	# Median
	if (n_nodes % 2) != 0: # Odd number of nodes
		median_node = sorted_node_indices[mid_point]
		pr_score_median = pr_scores[median_node]
		print(f"Node {median_node} has the exact median PageRank score of: {pr_score_median:.6f}")
		return [median_node]
	
	else: # Even number of nodes
		# Two middle nodes
		node_1 = sorted_node_indices[mid_point - 1]
		node_2 = sorted_node_indices[mid_point]
		print(f"Node {node_1} has a middle PageRank score of: {pr_scores[node_1]:.6f}")
		print(f"Node {node_2} has a middle PageRank score of: {pr_scores[node_2]:.6f}")
		print(f"The median score is: {np.median(pr_scores):.6f}")
		return [node_1, node_2]
        
def question2_4b(alpha=0.2, num_steps=5000):

	# Lock in rng for reproducibility
	lock_in_rng()

	# Generate PageRank network
	n = 900
	m = 4
	pagerank_net = page_rank_network(n, m)
	
	# Median PageRank nodes
	mdn_nodes = find_nodes_with_median_pagerank(pagerank_net, alpha)
	print(f'Median nodes: {mdn_nodes}')
	
	# 1/2 probability for two target nodes
	custom_teleport_probs = np.zeros(n)
	custom_teleport_probs[mdn_nodes[0]] = 0.5
	custom_teleport_probs[mdn_nodes[1]] = 0.5
	
	# Random walk with teleportation
	path = random_walk_with_teleportation(pagerank_net, 
										  num_steps, 
										  alpha=alpha,
										  teleport_probs=custom_teleport_probs,
										  rng=rng)

	# Visit probabilities to every node in graph
	visit_probs = estimate_visit_probabilities(path, n)
	
	# Plot probabilities
	plot_visit_probabilities(visit_probs, n, num_steps)
	
	# See the relationship between node in-degree and visit probability
	in_degrees = np.array(pagerank_net.indegree())

	# Find the node with the highest in-degree and highest visit probability
	print(f"Node with highest in-degree: {np.argmax(in_degrees)}")
	print(f"Node with highest visit probability: {np.argmax(visit_probs)}")
