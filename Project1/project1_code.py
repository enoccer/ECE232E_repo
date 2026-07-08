import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(232)

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
    print("New Network")
    network_plot(new_g)
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
    

    