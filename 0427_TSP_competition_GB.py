import gurobipy as gp
from gurobipy import GRB
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

# Load data
data = np.loadtxt('cities2024.txt')  # Correct path for your environment
distance_matrix = squareform(pdist(data))

# Cluster the cities into 3 groups using K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

# Function to solve TSP for a given cluster
def solve_tsp_for_cluster(cluster_indices, distance_matrix):
    model = gp.Model(f"TSP_cluster_{cluster_indices[0]}")
    n = len(cluster_indices)
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    # Objective: Minimize travel distance
    model.setObjective(gp.quicksum(x[i, j] * distance_matrix[cluster_indices[i], cluster_indices[j]]
                                   for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE)

    # Constraints: Enter and exit each city exactly once
    for i in range(n):
        model.addConstr(sum(x[i, j] for j in range(n) if j != i) == 1)
        model.addConstr(sum(x[j, i] for j in range(n) if j != i) == 1)

    # Subtour elimination constraints
    u = model.addVars(n, vtype=GRB.INTEGER)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)

    model.Params.lazyConstraints = 1
    model.optimize()

    tour = []
    if model.status == GRB.OPTIMAL:
        solution = model.getAttr('X', x)
        for i in range(n):
            for j in range(n):
                if i != j and solution[i, j] > 0.5:
                    tour.append((cluster_indices[i], cluster_indices[j]))

        tour_cost = model.objVal
        return tour, tour_cost
    else:
        return [], None

total_cost = 0
cluster_sizes = []
for t in range(3):
    cluster_indices = np.where(clusters == t)[0]
    cluster_sizes.append(len(cluster_indices))
    tour, tour_cost = solve_tsp_for_cluster(cluster_indices, distance_matrix)
    if tour:
        print(f"Tour for cluster {t+1}: {tour}")
        print(f"Cost for cluster {t+1}: {tour_cost}")
        total_cost += tour_cost

# Calculate the mean cluster size and penalty term
c_mean = np.mean(cluster_sizes)
penalty = 20000 * sum((c_t - c_mean) ** 2 for c_t in cluster_sizes)

# Calculate total objective K
K = total_cost + penalty

print(f"Total cost: {total_cost}")
print(f"Penalty: {penalty}")
print(f"Objective Function Value K: {K}")
