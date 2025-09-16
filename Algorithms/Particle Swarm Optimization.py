import numpy as np
import matplotlib.pyplot as plt

# Define problem data
depot = {"x": 0, "y": 0, "customer_number": 1, "demand": 0}
customers = [
    {"customer_number": 2, "x": -7, "y": 9, "demand": 94},
    {"customer_number": 3, "x": -7, "y": 8, "demand": 12},
    {"customer_number": 4, "x": -15, "y": 16, "demand": 17},
    {"customer_number": 5, "x": -2, "y": 13, "demand": 619},
    {"customer_number": 6, "x": 0, "y": 3, "demand": 61},
    {"customer_number": 7, "x": -9, "y": 6, "demand": 3},
    {"customer_number": 8, "x": 2, "y": 5, "demand": 4},
    {"customer_number": 9, "x": -13, "y": 9, "demand": 13},
    {"customer_number": 10, "x": -15, "y": 10, "demand": 44},
    {"customer_number": 11, "x": -14, "y": 2, "demand": 12},
    {"customer_number": 12, "x": 1, "y": 0, "demand": 35},
    {"customer_number": 13, "x": -40, "y": 24, "demand": 114},
    {"customer_number": 14, "x": -54, "y": -15, "demand": 29},
    {"customer_number": 15, "x": -43, "y": 10, "demand": 76},
    {"customer_number": 16, "x": -73, "y": -2, "demand": 106},
    {"customer_number": 17, "x": -76, "y": 4, "demand": 157},
    {"customer_number": 18, "x": -45, "y": 31, "demand": 43},
    {"customer_number": 19, "x": -29, "y": 36, "demand": 4},
    {"customer_number": 20, "x": -78, "y": 11, "demand": 38},
    {"customer_number": 21, "x": -31, "y": 25, "demand": 212},
    {"customer_number": 22, "x": -67, "y": 17, "demand": 42},
    {"customer_number": 23, "x": -31, "y": 4, "demand": 10},
    {"customer_number": 24, "x": -51, "y": 10, "demand": 19},
    {"customer_number": 25, "x": -30, "y": -9, "demand": 856},
    {"customer_number": 26, "x": -51, "y": 24, "demand": 13},
    {"customer_number": 27, "x": -80, "y": -10, "demand": 67},
    {"customer_number": 28, "x": -34, "y": 4, "demand": 144},
    {"customer_number": 29, "x": -45, "y": 25, "demand": 310},
    {"customer_number": 30, "x": 73, "y": 32, "demand": 85},
    {"customer_number": 31, "x": 78, "y": 59, "demand": 1061},
    {"customer_number": 32, "x": 58, "y": 53, "demand": 344},
    {"customer_number": 33, "x": 57, "y": 55, "demand": 22},
    {"customer_number": 34, "x": 20, "y": 85, "demand": 15},
    {"customer_number": 35, "x": -12, "y": 81, "demand": 219},
    {"customer_number": 36, "x": -8, "y": 86, "demand": 44},
    {"customer_number": 37, "x": 12, "y": 33, "demand": 370},
    {"customer_number": 38, "x": 19, "y": 86, "demand": 74},
    {"customer_number": 39, "x": -11, "y": 44, "demand": 82},
    {"customer_number": 40, "x": -2, "y": 67, "demand": 3},
    {"customer_number": 41, "x": 25, "y": 76, "demand": 39},
    {"customer_number": 42, "x": -29, "y": 44, "demand": 54},
    {"customer_number": 43, "x": 4, "y": 80, "demand": 22},
    {"customer_number": 44, "x": -2, "y": 84, "demand": 171},
    {"customer_number": 45, "x": -8, "y": 54, "demand": 65},
    {"customer_number": 46, "x": -16, "y": 34, "demand": 405},
    {"customer_number": 47, "x": -14, "y": 80, "demand": 19},
    {"customer_number": 48, "x": -17, "y": 32, "demand": 7},
    {"customer_number": 49, "x": 19, "y": 39, "demand": 586},
    {"customer_number": 50, "x": -20, "y": 82, "demand": 15},
    {"customer_number": 51, "x": 17, "y": 42, "demand": 149},
    {"customer_number": 52, "x": -4, "y": 5, "demand": 141},
    {"customer_number": 53, "x": -20, "y": 25, "demand": 9},
    {"customer_number": 54, "x": -12, "y": 14, "demand": 261},
    {"customer_number": 55, "x": -11, "y": 11, "demand": 4},
    {"customer_number": 56, "x": -12, "y": 4, "demand": 5},
    {"customer_number": 57, "x": -1, "y": 21, "demand": 21},
    {"customer_number": 58, "x": -1, "y": 1, "demand": 25},
    {"customer_number": 59, "x": -9, "y": 21, "demand": 86},
    {"customer_number": 60, "x": 9, "y": 3, "demand": 86},
    {"customer_number": 61, "x": 5, "y": 14, "demand": 124},
    {"customer_number": 62, "x": -9, "y": 27, "demand": 123},
    {"customer_number": 63, "x": -20, "y": 11, "demand": 11},
    {"customer_number": 64, "x": 0, "y": 30, "demand": 41},
    {"customer_number": 65, "x": -12, "y": 15, "demand": 279},
    {"customer_number": 66, "x": -3, "y": 17, "demand": 149},
    {"customer_number": 67, "x": -58, "y": -60, "demand": 9},
    {"customer_number": 68, "x": -71, "y": -58, "demand": 65},
    {"customer_number": 69, "x": -32, "y": -34, "demand": 155},
    {"customer_number": 70, "x": -59, "y": -37, "demand": 6},
    {"customer_number": 71, "x": -48, "y": -19, "demand": 83},
    {"customer_number": 72, "x": -71, "y": -49, "demand": 11},
    {"customer_number": 73, "x": 22, "y": -4, "demand": 735},
    {"customer_number": 74, "x": 13, "y": -10, "demand": 4},
    {"customer_number": 75, "x": 22, "y": -17, "demand": 56},
    {"customer_number": 76, "x": 13, "y": 12, "demand": 26},
    {"customer_number": 77, "x": 3, "y": -5, "demand": 5},
    {"customer_number": 78, "x": 27, "y": -4, "demand": 34},
    {"customer_number": 79, "x": 2, "y": -3, "demand": 13},
    {"customer_number": 80, "x": -2, "y": -22, "demand": 1017},
    {"customer_number": 81, "x": 30, "y": 1, "demand": 85},
    {"customer_number": 82, "x": 22, "y": -6, "demand": 10},
    {"customer_number": 83, "x": -27, "y": 62, "demand": 7},
    {"customer_number": 84, "x": -20, "y": 64, "demand": 6},
    {"customer_number": 85, "x": -26, "y": 69, "demand": 524},
    {"customer_number": 86, "x": -20, "y": 66, "demand": 16},
    {"customer_number": 87, "x": -29, "y": 67, "demand": 15},
    {"customer_number": 88, "x": -84, "y": -10, "demand": 117},
    {"customer_number": 89, "x": -69, "y": -22, "demand": 48},
    {"customer_number": 90, "x": -79, "y": -4, "demand": 43},
    {"customer_number": 91, "x": -84, "y": -19, "demand": 64},
    {"customer_number": 92, "x": -63, "y": -12, "demand": 30},
    {"customer_number": 93, "x": -76, "y": -14, "demand": 3},
    {"customer_number": 94, "x": -70, "y": -4, "demand": 21},
    {"customer_number": 95, "x": 30, "y": 39, "demand": 514},
    {"customer_number": 96, "x": 23, "y": 39, "demand": 625},
    {"customer_number": 97, "x": 30, "y": 48, "demand": 7},
    {"customer_number": 98, "x": 36, "y": 21, "demand": 257},
    {"customer_number": 99, "x": 32, "y": 16, "demand": 603},
    {"customer_number": 100, "x": 42, "y": 30, "demand": 4},
    {"customer_number": 101, "x": 62, "y": 16, "demand": 4}
]
n = len(customers)  # number of customers
m =11 # number of vehicles

Q = 1297 # vehicle capacity

# Prepare nodes, positions, and demands
nodes = [depot] + customers
positions = [(node["x"], node["y"]) for node in nodes]
demands = [node["demand"] for node in nodes]

# Precompute distance matrix
dist_matrix = np.zeros((n + 1, n + 1))
for i in range(n + 1):
    for j in range(n + 1):
        dist_matrix[i, j] = np.sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)

def decode_particle(X, m, Q, positions, demands, dist_matrix):
    n = len(demands) - 1  # number of customers

    # Step 1: Construct customer priority list
    customer_values = X[:n]
    U = np.argsort(customer_values)  # Indices 0 to n-1, representing customers 1 to n

    # Step 2: Construct vehicle priority matrix
    vehicle_refs = []
    for j in range(m):
        xref = X[n + j]
        yref = X[n + m + j]
        vehicle_refs.append((xref, yref))

    V = []
    for i in range(n):
        distances = [np.sqrt((positions[i + 1][0] - vref[0])**2 + (positions[i + 1][1] - vref[1])**2)
                     for vref in vehicle_refs]
        vehicle_order = np.argsort(distances)
        V.append(vehicle_order)

    # Step 3: Construct vehicle routes
    routes = [[] for _ in range(m)]  # List of customer indices per vehicle
    loads = [0] * m
    assigned = set()

    for k in U:
        customer = k + 1  # Customer index in nodes (1 to n)
        for vehicle in V[k]:
            route = routes[vehicle]
            demand = demands[customer]
            if loads[vehicle] + demand > Q:
                continue

            # Cheapest insertion
            min_cost = float('inf')
            best_route = None
            for pos in range(len(route) + 1):
                prev = 0 if pos == 0 else route[pos - 1]
                next_node = 0 if pos == len(route) else route[pos]
                cost = dist_matrix[prev, customer] + dist_matrix[customer, next_node] - dist_matrix[prev, next_node]
                if cost < min_cost:
                    min_cost = cost
                    best_route = route[:pos] + [customer] + route[pos:]

            if best_route and loads[vehicle] + demand <= Q:
                routes[vehicle] = best_route
                loads[vehicle] += demand
                assigned.add(customer)
                break

    # Compute total distance
    total_distance = 0
    for route in routes:
        if route:
            distance = dist_matrix[0, route[0]] + dist_matrix[route[-1], 0]
            if len(route) > 1:
                distance += sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
            total_distance += distance

    # Penalize unassigned customers
    if len(assigned) < n:
        total_distance += 10000 * (n - len(assigned))  # Large penalty

    return routes, total_distance

# PSO parameters
I = 100  # Number of particles
T = 1500  # Number of iterations
K = 5    # Number of neighbors
w_max = 0.9
w_min = 0.4
c_p = 0.5
c_g = 0.5
c_l = 1.5
c_n = 1.5
D = n + 2 * m  # Particle dimension

# Initialize particles
particles = []
for _ in range(I):
    X = np.random.uniform(0, 100, D)
    V = np.zeros(D)
    P = X.copy()
    particles.append({"X": X, "V": V, "P": P, "q_P": float('inf'), "L": None, "N": None})

# Initialize global best
gbest = None
q_gbest = float('inf')

# Define ring topology neighbors
neighbors = [[(i + j) % I for j in range(-2, 3)] for i in range(I)]

# GLNPSO main loop
w = w_max
for t in range(T):
    # Evaluate all particles
    for i in range(I):
        X = particles[i]["X"]
        routes, total_distance = decode_particle(X, m, Q, positions, demands, dist_matrix)
        q = total_distance
        if q < particles[i]["q_P"]:
            particles[i]["P"] = X.copy()
            particles[i]["q_P"] = q
        if q < q_gbest:
            gbest = X.copy()
            q_gbest = q

    # Print best distance every 10 iterations
    if t % 2 == 0:
        print(f"Iteration {t + 1}: Best Distance = {q_gbest:.2f}")

    # Update local best
    for i in range(I):
        neighbor_qs = [particles[j]["q_P"] for j in neighbors[i]]
        best_neighbor = neighbors[i][np.argmin(neighbor_qs)]
        particles[i]["L"] = particles[best_neighbor]["P"]

    # Update near neighbor best
    for i in range(I):
        N = np.zeros(D)
        for d in range(D):
            max_fdr = -float('inf')
            best_j = None
            for j in range(I):
                if j == i or particles[j]["q_P"] >= particles[i]["q_P"]:
                    continue
                # Avoid division by zero
                denom = abs(particles[i]["X"][d] - particles[j]["P"][d])
                if denom < 1e-10:
                    continue
                fdr = (particles[i]["q_P"] - particles[j]["q_P"]) / denom
                if fdr > max_fdr:
                    max_fdr = fdr
                    best_j = j
            N[d] = particles[best_j]["P"][d] if best_j is not None else particles[i]["P"][d]
        particles[i]["N"] = N

    # Update velocity and position
    for i in range(I):
        V = particles[i]["V"]
        X = particles[i]["X"]
        P = particles[i]["P"]
        L = particles[i]["L"]
        N = particles[i]["N"]
        r_p = np.random.rand(D)
        r_g = np.random.rand(D)
        r_l = np.random.rand(D)
        r_n = np.random.rand(D)
        V = (w * V + c_p * r_p * (P - X) + c_g * r_g * (gbest - X) +
             c_l * r_l * (L - X) + c_n * r_n * (N - X))
        X = X + V
        particles[i]["V"] = V
        particles[i]["X"] = X

    # Update inertia weight
    w += (w_min - w_max) / T

# Final solution
routes, total_distance = decode_particle(gbest, m, Q, positions, demands, dist_matrix)

# Output results
print(f"Best Total Distance: {total_distance:.2f}")
print("Vehicle Routes:")
for i, route in enumerate(routes):
    if route:
        route_str = " -> ".join(str(cust) for cust in [0] + route + [0])
        print(f"Vehicle {i + 1}: {route_str}")

# Plotting the routes
plt.figure(figsize=(10, 8))

# Plot depot
plt.scatter(depot["x"], depot["y"], c='red', marker='s', s=100, label='Depot')

# Plot customers
customer_x = [customer["x"] for customer in customers]
customer_y = [customer["y"] for customer in customers]
plt.scatter(customer_x, customer_y, c='blue', marker='o', s=50, label='Customers')

# Define colors for different vehicles
colors = ['green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

# Plot routes for each vehicle
for i, route in enumerate(routes):
    if route:
        # Create the full route including depot at start and end
        full_route = [0] + route + [0]
        route_x = [positions[idx][0] for idx in full_route]
        route_y = [positions[idx][1] for idx in full_route]

        # Plot the route lines
        plt.plot(route_x, route_y, c=colors[i % len(colors)], linewidth=2, label=f'Vehicle {i+1}')

        # Add arrows to indicate direction
        for j in range(len(route_x)-1):
            plt.arrow(route_x[j], route_y[j],
                     route_x[j+1] - route_x[j], route_y[j+1] - route_y[j],
                     color=colors[i % len(colors)], length_includes_head=True,
                     head_width=3, head_length=5, alpha=0.7)

# Add labels and legend
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Vehicle Routing Problem - Routes')
plt.legend()
plt.grid(True)
plt.show()
