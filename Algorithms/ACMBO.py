import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, ceil
import random
from sklearn.cluster import KMeans

# Data
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
demand = {c["customer_number"]: c["demand"] for c in customers}
locations = [depot] + customers  # Index 0: depot (1), 1 to 30: customers (2 to 31)
n_customers = len(customers)  # 30
n_population = max(50, min(100, 2 * n_customers))
n_iterations = 1500
n_routes = 11
capacity = 1297


# Precompute distance matrix
dist_matrix = np.zeros((n_customers + 1, n_customers + 1))
for i in range(n_customers + 1):
    for j in range(n_customers + 1):
        dist_matrix[i, j] = sqrt((locations[i]["x"] - locations[j]["x"])**2 + (locations[i]["y"] - locations[j]["y"])**2)

# Customer segmentation
K = ceil(sqrt(n_customers))
coords = np.array([[c["x"], c["y"]] for c in customers])
kmeans = KMeans(n_clusters=K, random_state=0).fit(coords)
customer_segments = [[] for _ in range(K)]
for idx, label in enumerate(kmeans.labels_):
    customer_segments[label].append(idx)

# Archive for elite solutions
elite_archive = []
archive_size = max(5, min(10, n_customers // 10))

# Track enhancement success rates
enhancement_success = {
    'intra': {'count': 0, 'success': 0, 'interval': 2},
    'inter': {'count': 0, 'success': 0, 'interval': 3},
    'destroy_repair': {'count': 0, 'success': 0, 'interval': 10},
    'three_opt': {'count': 0, 'success': 0, 'interval': 3}
}

# Fitness history for dynamic segmentation
fitness_history = []

def calculate_distance(loc1, loc2):
    return sqrt((loc1["x"] - loc2["x"])**2 + (loc1["y"] - loc2["y"])**2)

def decode_solution(solution, n_routes):
    n = n_customers
    # Step 1: Construct customer priority list
    customer_values = solution[:n]
    U = np.argsort(customer_values)  # Indices 0 to n-1

    # Step 2: Construct vehicle priority matrix
    vehicle_refs = []
    for j in range(n_routes):
        xref = solution[n + j]
        yref = solution[n + n_routes + j]
        vehicle_refs.append((xref, yref))

    V = []
    for i in range(n):
        distances = [np.sqrt((locations[i + 1]["x"] - vref[0])**2 + (locations[i + 1]["y"] - vref[1])**2)
                     for vref in vehicle_refs]
        vehicle_order = np.argsort(distances)
        V.append(vehicle_order)

    # Step 3: Construct vehicle routes
    routes = [[] for _ in range(n_routes)]
    loads = [0] * n_routes
    assigned = set()

    for k in U:
        customer = k + 2  # Customer number: 2 to 31
        for vehicle in V[k]:
            route = routes[vehicle]
            customer_demand = demand[customer]
            if loads[vehicle] + customer_demand > capacity:
                continue

            # Cheapest insertion
            min_cost = float('inf')
            best_route = None
            for pos in range(len(route) + 1):
                prev = 1 if pos == 0 else route[pos - 1]
                next_node = 1 if pos == len(route) else route[pos]
                cost = (dist_matrix[prev - 1, customer - 1] +
                        dist_matrix[customer - 1, next_node - 1] -
                        dist_matrix[prev - 1, next_node - 1])
                if cost < min_cost:
                    min_cost = cost
                    best_route = route[:pos] + [customer] + route[pos:]

            if best_route and loads[vehicle] + customer_demand <= capacity:
                routes[vehicle] = best_route
                loads[vehicle] += customer_demand
                assigned.add(k)
                break

    return routes, assigned

def fitness_function(solution, segment_indices=None, weight=0.7):
    routes, assigned = decode_solution(solution, n_routes)
    total_distance = 0
    segment_distance = 0 if segment_indices is not None else None

    for route in routes:
        if route:
            distance = dist_matrix[0, route[0] - 1] + dist_matrix[route[-1] - 1, 0]
            if len(route) > 1:
                distance += sum(dist_matrix[route[i] - 1, route[i + 1] - 1] for i in range(len(route) - 1))
            total_distance += distance
            if segment_indices is not None and any(idx + 2 in route for idx in segment_indices):
                segment_distance += distance

    if len(assigned) < n_customers:
        total_distance += 10000 * (n_customers - len(assigned))

    if segment_indices is not None:
        return weight * segment_distance + (1 - weight) * total_distance, routes
    return total_distance, routes

def initialize_solutions(n_pop, n_vars):
    solutions = np.random.uniform(0, 1, size=(n_pop, n_vars))
    return solutions

def compute_crowding_distance(solutions, segment_indices):
    n = len(solutions)
    distances = np.zeros(n)
    if n <= 2:
        distances.fill(float('inf'))
        return distances
    valid_indices = [idx for idx in segment_indices if 0 <= idx < n_customers]
    if not valid_indices:
        distances.fill(float('inf'))
        return distances
    for idx in valid_indices:
        values = solutions[:, idx]
        sorted_indices = np.argsort(values)
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
        min_val, max_val = values[sorted_indices[0]], values[sorted_indices[-1]]
        if max_val > min_val:
            for i in range(1, n-1):
                distances[sorted_indices[i]] += (values[sorted_indices[i+1]] - values[sorted_indices[i-1]]) / (max_val - min_val)
    return distances

def fitness_sharing(solutions, objectives, segment_indices):
    valid_indices = [idx for idx in segment_indices if 0 <= idx < n_customers]
    if not valid_indices:
        return objectives.copy()
    sigma = 0.5 * sqrt(len(valid_indices))
    crowding_distances = compute_crowding_distance(solutions, valid_indices)
    max_crowding = max(crowding_distances) if max(crowding_distances) > 0 else 1
    shared_objectives = objectives.copy()
    for i in range(len(solutions)):
        niche_count = 1
        for j in range(len(solutions)):
            if i != j:
                distance = np.sqrt(np.sum((solutions[i, valid_indices] - solutions[j, valid_indices])**2))
                niche_count += np.exp(-(distance**2) / (2 * sigma**2))
        niche_count = max(niche_count, 1e-6)
        crowding_factor = crowding_distances[i] / max_crowding
        shared_objectives[i] = objectives[i] / niche_count * (1 - crowding_factor)
    return shared_objectives

def update_cats(cats, mice, objective_cats, segment_indices, elite_archive, iteration, max_iterations):
    n_cats = len(cats)
    n_mice = len(mice)
    new_cats = cats.copy()
    new_objectives = objective_cats.copy()
    T_0 = 100 * sqrt(n_customers)
    T_t = T_0 * (1 - iteration / max_iterations)
    alpha = 0.5
    exploration_factor = 1 + alpha * (iteration / max_iterations)
    mutation_prob = 0.05 + 0.1 * (n_customers / 100)

    if elite_archive:
        elite_fitnesses = [1 / (item[1] + 1e-6) for item in elite_archive]
        total_fitness = sum(elite_fitnesses)
        elite_probs = [f / total_fitness for f in elite_fitnesses]
        elite_solutions = [np.array(item[0], dtype=float) for item in elite_archive]
    else:
        elite_solutions = [cats[np.argmin(objective_cats)]]
        elite_probs = [1.0]

    for j in range(n_cats):
        k = np.random.randint(0, n_mice)
        r = np.random.random()
        I = round(1 + np.random.random())
        new_cat = cats[j].copy()
        best_idx = np.random.choice(len(elite_solutions), p=elite_probs)
        best_solution = elite_solutions[best_idx]
        valid_indices = [i for i in segment_indices if i < n_customers] + list(range(n_customers, len(cats[j])))
        new_cat[valid_indices] += r * (mice[k][valid_indices] - I * cats[j][valid_indices])*exploration_factor
        new_cat = np.clip(new_cat, 0, 1)

        if random.random() < mutation_prob:
            if len(valid_indices) >= 2:
                idx1, idx2 = random.sample(valid_indices, 2)
                new_cat[idx1], new_cat[idx2] = new_cat[idx2], new_cat[idx1]

        new_distance, new_routes = fitness_function(new_cat)
        delta_f = new_distance - objective_cats[j]
        if delta_f < 0 or (T_t > 0 and random.random() < np.exp(-delta_f / T_t)):
            new_cats[j] = new_cat
            new_objectives[j] = new_distance
    return new_cats, new_objectives

def update_mice(mice, population, objective_mice, objective_population):
    n_mice = len(mice)
    new_mice = mice.copy()
    new_objectives = objective_mice.copy()
    mutation_prob = 0.05 + 0.1 * (n_customers / 100)

    for i in range(n_mice):
        l = np.random.randint(0, len(population))
        r = np.random.random()
        I = round(1 + np.random.random())
        sign = 1 if objective_mice[i] > objective_population[l] else -1 if objective_mice[i] < objective_population[l] else 0
        new_mouse = mice[i] + r * (population[l] - I * mice[i]) * sign
        new_mouse = np.clip(new_mouse, 0, 1)

        if random.random() < mutation_prob:
            if len(new_mouse) >= 2:
                idx1, idx2 = random.sample(range(len(new_mouse)), 2)
                new_mouse[idx1], new_mouse[idx2] = new_mouse[idx2], new_mouse[idx1]

        new_distance, _ = fitness_function(new_mouse)
        if new_distance < objective_mice[i]:
            new_mice[i] = new_mouse
            new_objectives[i] = new_distance

    return new_mice, new_objectives

def three_opt(route):
    if len(route) < 4:
        return route
    improved = True
    best_route = route.copy()
    best_distance = calculate_route_distance(best_route)

    while improved:
        improved = False
        n = len(best_route)
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                for k in range(j + 2, n):
                    new_routes = [
                        best_route[:i + 1] + best_route[i + 1:j][::-1] + best_route[j:k] + best_route[k:],
                        best_route[:i + 1] + best_route[i + 1:j] + best_route[j:k][::-1] + best_route[k:],
                        best_route[:i + 1] + best_route[j:k] + best_route[i + 1:j] + best_route[k:],
                        best_route[:i + 1] + best_route[j:k][::-1] + best_route[i + 1:j][::-1] + best_route[k:],
                        best_route[:i + 1] + best_route[j:k][::-1] + best_route[i + 1:j] + best_route[k:],
                        best_route[:i + 1] + best_route[j:k] + best_route[i + 1:j][::-1] + best_route[k:],
                        best_route[:i + 1] + best_route[j:k][::-1] + best_route[i + 1:j][::-1] + best_route[k:]
                    ]
                    for new_route in new_routes:
                        new_distance = calculate_route_distance(new_route)
                        if new_distance < best_distance:
                            best_distance = new_distance
                            best_route = new_route.copy()
                            improved = True
    return best_route

def calculate_route_distance(route):
    if not route:
        return 0
    distance = dist_matrix[0, route[0] - 1] + dist_matrix[route[-1] - 1, 0]
    if len(route) > 1:
        distance += sum(dist_matrix[route[i] - 1, route[i + 1] - 1] for i in range(len(route) - 1))
    return distance

def destroy_solution(routes, k=max(5, int(n_customers / 10))):
    permutation = [c for route in routes for c in route]
    if not permutation or k >= len(permutation):
        return [], permutation
    selected_customer = random.choice(permutation)
    correlations = []
    for j in permutation:
        if j != selected_customer:
            d_ij = dist_matrix[selected_customer - 1, j - 1]
            max_d = max(dist_matrix[a - 1, b - 1] for a in permutation for b in permutation if a != b) or 1
            V_ij = 0 if any(selected_customer in route and j in route for route in routes) else 1
            R_ij = 1 / (d_ij / max_d + V_ij + 1e-6)
            correlations.append((j, R_ij))
    correlations.sort(key=lambda x: x[1], reverse=True)
    eliminated = [selected_customer] + [x[0] for x in correlations[:min(k-1, len(correlations))]]
    destroyed_perm = [x for x in permutation if x not in eliminated]
    return eliminated, destroyed_perm

def repair_solution(destroyed_perm, eliminated, routes, original_solution):
    current_perm = destroyed_perm[:]
    current_routes = [route[:] for route in routes if route]
    remaining = eliminated[:]

    while remaining:
        best_cost = float('inf')
        best_customer = None
        best_route_idx = None
        best_pos = None
        c_costs = []

        for customer in remaining:
            for r_idx, route in enumerate(current_routes):
                route_demand = sum(demand[c] for c in route)
                if route_demand + demand[customer] <= capacity:
                    for pos in range(len(route) + 1):
                        prev = 1 if pos == 0 else route[pos - 1]
                        next_node = 1 if pos == len(route) else route[pos]
                        cost = (dist_matrix[prev - 1, customer - 1] +
                                dist_matrix[customer - 1, next_node - 1] -
                                dist_matrix[prev - 1, next_node - 1])
                        c_costs.append((cost, customer, r_idx, pos))

        if c_costs:
            c_costs.sort()
            if len(c_costs) >= 2:
                s_loss = c_costs[1][0] - c_costs[0][0]
                if c_costs[0][0] < best_cost:
                    best_cost = c_costs[0][0]
                    best_customer = c_costs[0][1]
                    best_route_idx = c_costs[0][2]
                    best_pos = c_costs[0][3]

        if best_customer is not None:
            current_routes[best_route_idx].insert(best_pos, best_customer)
            remaining.remove(best_customer)
            current_perm.append(best_customer)
        else:
            return original_solution, routes

    while len(current_routes) < n_routes:
        current_routes.append([])
    if len(current_routes) > n_routes:
        current_routes = current_routes[:n_routes]

    return current_perm, current_routes

def inter_insert(routes):
    if len(routes) < 2:
        return routes
    routes = [route[:] for route in routes]
    for _ in range(10):
        r1_idx = random.randint(0, len(routes) - 1)
        r2_idx = random.randint(0, len(routes) - 1)
        if r1_idx != r2_idx and routes[r1_idx]:
            customer = random.choice(routes[r1_idx])
            r2_demand = sum(demand[c] for c in routes[r2_idx])
            if r2_demand + demand[customer] <= capacity:
                pos = random.randint(0, len(routes[r2_idx]))
                routes[r2_idx].insert(pos, customer)
                routes[r1_idx].remove(customer)
                if not routes[r1_idx] and len([r for r in routes if r]) >= n_routes:
                    routes[r1_idx] = []
                return routes
    return routes

def inter_change(routes):
    if len(routes) < 2:
        return routes
    routes = [route[:] for route in routes]
    for _ in range(10):
        r1_idx = random.randint(0, len(routes) - 1)
        r2_idx = random.randint(0, len(routes) - 1)
        if r1_idx != r2_idx and routes[r1_idx] and routes[r2_idx]:
            c1 = random.choice(routes[r1_idx])
            c2 = random.choice(routes[r2_idx])
            r1_demand = sum(demand[c] for c in routes[r1_idx]) - demand[c1] + demand[c2]
            r2_demand = sum(demand[c] for c in routes[r2_idx]) - demand[c2] + demand[c1]
            if r1_demand <= capacity and r2_demand <= capacity:
                routes[r1_idx].remove(c1)
                routes[r2_idx].remove(c2)
                routes[r1_idx].append(c2)
                routes[r2_idx].append(c1)
                return routes
    return routes

def intra_insert(route):
    if len(route) < 2:
        return route
    route = route[:]
    idx1, idx2 = random.sample(range(len(route)), 2)
    customer = route.pop(idx1)
    insert_pos = idx2 if idx2 > idx1 else idx2 + 1
    route.insert(insert_pos, customer)
    return route

def intra_change(route):
    if len(route) < 2:
        return route
    route = route[:]
    idx1, idx2 = random.sample(range(len(route)), 2)
    route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def intra_reverse(route):
    if len(route) < 2:
        return route
    route = route[:]
    idx1, idx2 = sorted(random.sample(range(len(route)), 2))
    route[idx1:idx2 + 1] = route[idx1:idx2 + 1][::-1]
    return route

def calculate_total_distance(routes):
    total_distance = 0
    for route in routes:
        if route:
            distance = dist_matrix[0, route[0] - 1] + dist_matrix[route[-1] - 1, 0]
            if len(route) > 1:
                distance += sum(dist_matrix[route[i] - 1, route[i + 1] - 1] for i in range(len(route) - 1))
            total_distance += distance
    return total_distance

def variable_neighborhood_descent(routes, global_best_fitness):
    operators = [intra_insert, intra_change, intra_reverse, three_opt, inter_change, inter_insert]
    best_routes = [route[:] for route in routes]
    best_fitness = calculate_total_distance(best_routes)
    improved = True
    op_idx = 0

    while improved and op_idx < len(operators):
        improved = False
        if op_idx < 4:
            new_routes = [operators[op_idx](route) if len(route) >= (4 if op_idx == 3 else 2) else route for route in best_routes]
        else:
            new_routes = operators[op_idx](best_routes)
        new_perm = [c for route in new_routes for c in route]
        if len(new_perm) == n_customers and set(new_perm) == set(range(2, n_customers + 2)) and len([r for r in new_routes if r]) <= n_routes:
            new_fitness = calculate_total_distance(new_routes)
            if new_fitness < best_fitness:
                best_routes = new_routes
                best_fitness = new_fitness
                improved = True
                op_idx = 0
            else:
                op_idx += 1
        else:
            op_idx += 1
    return best_routes, best_fitness

def cmbo_vrp():
    n_variables = n_customers + 2 * n_routes
    solutions = initialize_solutions(n_population, n_variables)
    objectives = np.array([fitness_function(sol)[0] for sol in solutions])
    best_idx = np.argmin(objectives)
    global_best_solution = solutions[best_idx].copy()
    global_best_fitness = objectives[best_idx]
    global_best_routes = fitness_function(global_best_solution)[1]
    no_improvement_count = 0
    global elite_archive
    global fitness_history
    fitness_history.append(global_best_fitness)
    elite_archive.append((global_best_solution, global_best_fitness, global_best_routes))

    n_cats = n_population // 2
    n_mice = n_population - n_cats
    if n_customers<1000:
      K_base = max(2, int(sqrt(n_customers) / 2))
    elif n_customers>=1000:
      K_base = ceil(sqrt(n_customers) * 1.5)

    K_t = K_base
    customer_segments_t = [[idx for idx in segment if 0 <= idx < n_customers] for segment in customer_segments]
    cat_assignments = []
    cats_per_segment = n_cats // K_t
    remaining_cats = n_cats % K_t
    cat_idx = 0

    for k in range(K_t):
        segment_size = cats_per_segment + (1 if k < remaining_cats else 0)
        cat_assignments.append(list(range(cat_idx, cat_idx + segment_size)))
        cat_idx += segment_size

    for t in range(n_iterations):
        if t % 50 == 0 and t > 50:
            if len(fitness_history) >= 50:
                delta_f_t = (fitness_history[-50] - fitness_history[-1]) / fitness_history[-50] if fitness_history[-50] > 0 else 0
                K_t = max(2, min(ceil(sqrt(n_customers)), K_base + int(5 * (1 - delta_f_t))))
                coords = np.array([[c["x"], c["y"]] for c in customers])
                kmeans = KMeans(n_clusters=K_t, random_state=t).fit(coords)
                customer_segments_t = [[] for _ in range(K_t)]
                for idx, label in enumerate(kmeans.labels_):
                    if 0 <= idx < n_customers:
                        customer_segments_t[label].append(idx)
                cat_assignments = []
                cats_per_segment = n_cats // K_t
                remaining_cats = n_cats % K_t
                cat_idx = 0
                for k in range(K_t):
                    segment_size = cats_per_segment + (1 if k < remaining_cats else 0)
                    cat_assignments.append(list(range(cat_idx, cat_idx + segment_size)))
                    cat_idx += segment_size

        m_t = ceil(n_customers / K_t * (1 - 0.5 * t / n_iterations))
        active_segments = customer_segments_t[:min(K_t, ceil(n_customers / m_t))]

        sorted_indices = np.argsort(objectives)
        sorted_solutions = solutions[sorted_indices]
        sorted_objectives = objectives[sorted_indices]
        mice = sorted_solutions[:n_mice]
        objective_mice = sorted_objectives[:n_mice]

        all_new_cats = []
        all_new_objectives = []
        for k, segment_indices in enumerate(active_segments):
            valid_segment_indices = [idx for idx in segment_indices if 0 <= idx < n_customers]
            if not valid_segment_indices:
                continue
            segment_cats = sorted_solutions[n_mice:][cat_assignments[k]]
            segment_objectives = sorted_objectives[n_mice:][cat_assignments[k]]
            segment_shared = fitness_sharing(segment_cats, segment_objectives, valid_segment_indices)
            segment_sorted_indices = np.argsort(segment_shared)
            segment_cats = segment_cats[segment_sorted_indices]
            segment_objectives = segment_objectives[segment_sorted_indices]
            new_segment_cats, new_segment_objectives = update_cats(
                segment_cats, mice, segment_objectives, valid_segment_indices, elite_archive, t, n_iterations
            )
            all_new_cats.append(new_segment_cats)
            all_new_objectives.append(new_segment_objectives)

        if all_new_cats:
            cats = np.vstack(all_new_cats)
            objective_cats = np.concatenate(all_new_objectives)
            solutions = np.vstack((mice, cats))
            objectives = np.concatenate((objective_mice, objective_cats))

        new_mice, new_objective_mice = update_mice(mice, sorted_solutions, objective_mice, sorted_objectives)
        solutions[:n_mice] = new_mice
        objectives[:n_mice] = new_objective_mice
        destroy_interval = enhancement_success['destroy_repair']['interval']
        

      

        if no_improvement_count % destroy_interval == 0:
            old_fitness = global_best_fitness
            eliminated, destroyed_perm = destroy_solution(global_best_routes, k=5)
            new_perm, new_routes = repair_solution(destroyed_perm, eliminated, global_best_routes, global_best_solution)
            if len(new_perm) == n_customers and set(new_perm) == set(range(2, n_customers + 2)):
                new_fitness = calculate_total_distance(new_routes)
                enhancement_success['destroy_repair']['count'] += 1
                if new_fitness < global_best_fitness:
                    global_best_fitness = new_fitness
                    global_best_routes = new_routes
                    no_improvement_count = 0
                    enhancement_success['destroy_repair']['success'] += 1
                    elite_archive.append((global_best_solution, global_best_fitness, global_best_routes))
                    if len(elite_archive) > archive_size:
                        elite_archive.sort(key=lambda x: x[1])
                        elite_archive.pop(-1)
                else:
                    no_improvement_count += 1


        if no_improvement_count > 40:
            old_fitness = global_best_fitness
            new_routes, new_fitness = variable_neighborhood_descent(global_best_routes, global_best_fitness)
            new_perm = [c for route in new_routes for c in route]
            if len(new_perm) == n_customers and set(new_perm) == set(range(2, n_customers + 2)):
                if new_fitness < global_best_fitness:
                    global_best_fitness = new_fitness
                    global_best_routes = new_routes
                    no_improvement_count = 0
                    elite_archive.append((global_best_solution, global_best_fitness, global_best_routes))
                    if len(elite_archive) > archive_size:
                        elite_archive.sort(key=lambda x: x[1])
                        elite_archive.pop(-1)
                else:
                    no_improvement_count += 1

        if no_improvement_count % 5== 0 and no_improvement_count > 100 and t>500 :
            n_restart_segments = K_t // 2
            restart_segments = random.sample(range(K_t), n_restart_segments)
            for k in restart_segments:
                if customer_segments_t[k]:
                    restart_indices = [n_mice + idx for idx in cat_assignments[k] if n_mice + idx < n_population]
                    if restart_indices and customer_segments_t[k]:
                        n_elite = int(0.2 * len(restart_indices))
                        if n_elite > 0 and elite_archive:
                            elite_indices = random.sample(range(len(elite_archive)), min(n_elite, len(elite_archive)))
                            for i, idx in enumerate(restart_indices[:n_elite]):
                                elite_sol, _, _ = elite_archive[elite_indices[i]]
                                elite_solution = np.random.uniform(0, 1, n_variables)
                                elite_solution[customer_segments_t[k]] = sorted([random.random() for _ in range(len(customer_segments_t[k]))])
                                solutions[idx] = elite_solution
                        solutions[np.ix_(restart_indices[n_elite:], customer_segments_t[k])] = np.random.uniform(
                            0, 1, size=(len(restart_indices[n_elite:]), len(customer_segments_t[k]))
                        )
                        for idx in restart_indices:
                            objectives[idx], _ = fitness_function(solutions[idx])
            no_improvement_count = 0

        best_idx = np.argmin(objectives)
        new_fitness, new_routes = fitness_function(solutions[best_idx])
        if new_fitness < global_best_fitness:
            global_best_solution = solutions[best_idx].copy()
            global_best_fitness = new_fitness
            global_best_routes = new_routes
            no_improvement_count = 0
            elite_archive.append((global_best_solution, global_best_fitness, global_best_routes))
            if len(elite_archive) > archive_size:
                elite_archive.sort(key=lambda x: x[1])
                elite_archive.pop(-1)
        else:
            no_improvement_count += 1
        fitness_history.append(global_best_fitness)
        if len(fitness_history) > 50:
            fitness_history.pop(0)
        print(f"Iteration {t + 1}: Global Fitness (Total Distance) = {global_best_fitness:.2f}")

    return global_best_solution, global_best_fitness, global_best_routes

def plot_routes(routes, filename='vrp_routes.png'):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab20', max(len(routes), 1))

    plt.scatter(depot['x'], depot['y'], c='red', marker='s', s=100, label='Depot')
    plt.text(depot['x'] + 2, depot['y'], f"Depot ({depot['customer_number']})", fontsize=8)

    for customer in customers:
        plt.scatter(customer['x'], customer['y'], c='blue', marker='o', s=50)
        plt.text(customer['x'] + 2, customer['y'], f"{customer['customer_number']}", fontsize=8)

    for i, route in enumerate(routes):
        if route:
            route_points = [depot] + [locations[c - 1] for c in route] + [depot]
            x = [p['x'] for p in route_points]
            y = [p['y'] for p in route_points]
            plt.plot(x, y, color=colors(i), label=f'Route {i + 1}', linewidth=2)

    plt.title(f'Optimal VRP Routes (Total Distance: {best_fitness:.2f})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    best_solution, best_fitness, best_routes = cmbo_vrp()
    print(f"Optimal Fitness (Total Distance): {best_fitness:.2f}")
    print("Optimal Routes:")
    for i, route in enumerate(best_routes, 1):
        route_demand = sum(demand[c] for c in route) if route else 0
        print(f"  Route {i}: Depot -> {route} -> Depot (Demand: {route_demand})")
    plot_routes(best_routes)
