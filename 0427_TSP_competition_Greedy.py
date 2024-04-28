import numpy as np
from sklearn.cluster import KMeans

# Load data
data = np.loadtxt('cities2024.txt')  # Ensure the path is correct

# Function to calculate distance between two points
def distance(x1, x2, y1, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Extracting x and y coordinates
x = data[:, 0]
y = data[:, 1]

# Creating the distance matrix
distance_matrix = []
for i in range(len(x)):
    row = []
    for j in range(len(y)):
        if i == j:
            row.append(0)  # Distance from a city to itself is 0
        else:
            row.append(distance(x[i], x[j], y[i], y[j]))
    distance_matrix.append(row)

distance_matrix_np = np.array(distance_matrix)

# Cluster the cities into 3 groups using K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

# Function to perform a simple greedy TSP approach
def greedy_tsp(indices, distance_matrix):
    n = len(indices)
    start = indices[0]
    tour = [start]
    used = set(tour)
    total_distance = 0

    current = start
    while len(tour) < n:
        next_city = min(((distance_matrix[current, i], i) for i in indices if i not in used), key=lambda x: x[0])
        current = next_city[1]
        tour.append(current)
        used.add(current)
        total_distance += next_city[0]

    # Return to start
    total_distance += distance_matrix[current, start]
    tour.append(start)
    return tour, total_distance

# Calculate the tours and costs
tours = []
costs = []
cluster_sizes = []
for i in range(3):
    cluster_indices = np.where(clusters == i)[0]
    cluster_sizes.append(len(cluster_indices))
    tour, cost = greedy_tsp(cluster_indices, distance_matrix_np)
    tours.append(tour)
    costs.append(cost)

# Calculate the objective function
C = len(data)  # Total number of cities
c_mean = C / 3  # Average number of cities per cluster
objective = sum(costs) + 20000 * sum((c_t - c_mean) ** 2 for c_t in cluster_sizes)

# Print the results
for i, (tour, cost) in enumerate(zip(tours, costs)):
    print(f"Group {i+1} Tour: {tour}")
    print(f"Group {i+1} Cost: {cost}")

print(f"Objective Function Value: {objective}")
