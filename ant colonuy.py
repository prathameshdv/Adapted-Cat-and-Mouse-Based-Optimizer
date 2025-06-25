import math
import random
import numpy as np
from uuid import uuid4

# Input data
depot = {"x": 0, "y": 0, "customer_number": 1, "demand": 0}
customers = [
    {"customer_number": 2, "x": 35, "y": -56, "demand": 50},
    {"customer_number": 3, "x": 72, "y": -58, "demand": 50},
    {"customer_number": 4, "x": 70, "y": -66, "demand": 170},
    {"customer_number": 5, "x": 45, "y": -40, "demand": 297},
    {"customer_number": 6, "x": 39, "y": -40, "demand": 9},
    {"customer_number": 7, "x": 60, "y": -50, "demand": 630},
    {"customer_number": 8, "x": 42, "y": -59, "demand": 179},
    {"customer_number": 9, "x": 31, "y": -46, "demand": 179},
    {"customer_number": 10, "x": 44, "y": -58, "demand": 216},
    {"customer_number": 11, "x": 45, "y": -67, "demand": 4},
    {"customer_number": 12, "x": 69, "y": -46, "demand": 9},
    {"customer_number": 13, "x": 24, "y": 0, "demand": 154},
    {"customer_number": 14, "x": 12, "y": -4, "demand": 117},
    {"customer_number": 15, "x": 1, "y": -21, "demand": 63},
    {"customer_number": 16, "x": 3, "y": 29, "demand": 436},
    {"customer_number": 17, "x": 19, "y": -13, "demand": 905},
    {"customer_number": 18, "x": 13, "y": -14, "demand": 14},
    {"customer_number": 19, "x": 25, "y": 11, "demand": 3},
    {"customer_number": 20, "x": 24, "y": 23, "demand": 10},
    {"customer_number": 21, "x": 3, "y": 7, "demand": 166},
    {"customer_number": 22, "x": 23, "y": 19, "demand": 211},
    {"customer_number": 23, "x": 2, "y": -7, "demand": 8},
    {"customer_number": 24, "x": 5, "y": 23, "demand": 25},
    {"customer_number": 25, "x": 32, "y": 5, "demand": 139},
    {"customer_number": 26, "x": 14, "y": 25, "demand": 213},
    {"customer_number": 27, "x": -16, "y": -4, "demand": 758},
    {"customer_number": 28, "x": 24, "y": 17, "demand": 429},
    {"customer_number": 29, "x": 0, "y": -7, "demand": 5},
    {"customer_number": 30, "x": -74, "y": -22, "demand": 136},
    {"customer_number": 31, "x": -64, "y": -24, "demand": 501},
    {"customer_number": 32, "x": -71, "y": -19, "demand": 93},
    {"customer_number": 33, "x": -91, "y": -15, "demand": 21},
    {"customer_number": 34, "x": -65, "y": -14, "demand": 169},
    {"customer_number": 35, "x": -91, "y": -26, "demand": 22},
    {"customer_number": 36, "x": -76, "y": -7, "demand": 3},
    {"customer_number": 37, "x": -66, "y": -4, "demand": 271},
    {"customer_number": 38, "x": -87, "y": -10, "demand": 433},
    {"customer_number": 39, "x": -73, "y": -8, "demand": 3},
    {"customer_number": 40, "x": -81, "y": -1, "demand": 1079},
    {"customer_number": 41, "x": -82, "y": -24, "demand": 233},
    {"customer_number": 42, "x": -87, "y": -25, "demand": 11},
    {"customer_number": 43, "x": -76, "y": -25, "demand": 10},
    {"customer_number": 44, "x": -75, "y": -6, "demand": 78},
    {"customer_number": 45, "x": -70, "y": -3, "demand": 63},
    {"customer_number": 46, "x": -64, "y": -22, "demand": 4},
    {"customer_number": 47, "x": -66, "y": -5, "demand": 59},
    {"customer_number": 48, "x": -72, "y": -10, "demand": 8},
    {"customer_number": 49, "x": -89, "y": -3, "demand": 34},
    {"customer_number": 50, "x": -86, "y": -3, "demand": 234},
    {"customer_number": 51, "x": -57, "y": -9, "demand": 30},
    {"customer_number": 52, "x": -22, "y": -36, "demand": 40},
    {"customer_number": 53, "x": -44, "y": 19, "demand": 123},
    {"customer_number": 54, "x": -21, "y": 6, "demand": 7},
    {"customer_number": 55, "x": -49, "y": -4, "demand": 33},
    {"customer_number": 56, "x": -68, "y": -7, "demand": 369},
    {"customer_number": 57, "x": -42, "y": 11, "demand": 11},
    {"customer_number": 58, "x": -69, "y": 3, "demand": 23},
    {"customer_number": 59, "x": -49, "y": 9, "demand": 208},
    {"customer_number": 60, "x": -68, "y": -19, "demand": 4},
    {"customer_number": 61, "x": -57, "y": -7, "demand": 8},
    {"customer_number": 62, "x": -61, "y": -34, "demand": 36},
    {"customer_number": 63, "x": -36, "y": 16, "demand": 504},
    {"customer_number": 64, "x": -56, "y": 2, "demand": 16},
    {"customer_number": 65, "x": -67, "y": 0, "demand": 574},
    {"customer_number": 66, "x": -17, "y": -14, "demand": 19},
    {"customer_number": 67, "x": -17, "y": -20, "demand": 235},
    {"customer_number": 68, "x": -28, "y": -26, "demand": 445},
    {"customer_number": 69, "x": -70, "y": -21, "demand": 6},
    {"customer_number": 70, "x": -46, "y": -14, "demand": 43},
    {"customer_number": 71, "x": -52, "y": 36, "demand": 210},
    {"customer_number": 72, "x": -33, "y": 62, "demand": 268},
    {"customer_number": 73, "x": -53, "y": 49, "demand": 410},
    {"customer_number": 74, "x": -39, "y": 59, "demand": 124},
    {"customer_number": 75, "x": 33, "y": 73, "demand": 11},
    {"customer_number": 76, "x": 38, "y": 88, "demand": 1085}
]
# Parameters
NUM_ANTS = 100
MAX_GENERATIONS = 1000
Q = 1000  # Pheromone constant
ALPHA = 2  # Pheromone influence
BETA = 1   # Visibility influence
RHO = 0.8  # Evaporation rate
VEHICLE_CAPACITY =  1445
NC = len(customers)
NV = 10 # Estimated number of vehicles
PMIN = 1/NC
PMAX = 1/NV

# Initialize data
nodes = [depot] + customers
demands = [node['demand'] for node in nodes]
positions = [(node['x'], node['y']) for node in nodes]
n = len(demands) - 1  # Number of customers
m = NV  # Number of vehicles

# Distance matrix
DIST_MATRIX = np.zeros((len(nodes), len(nodes)))
for i in range(len(nodes)):
    for j in range(len(nodes)):
        DIST_MATRIX[i][j] = math.sqrt((nodes[i]['x'] - nodes[j]['x'])**2 + (nodes[i]['y'] - nodes[j]['y'])**2)

# Initialize pheromone matrix
PHEROMONE = np.ones((len(nodes), len(nodes))) * 0.1
SMIN = Q / (2 * sum(DIST_MATRIX[0][1:]))
SMAX = Q / sum(DIST_MATRIX[0][1:])

# 2-opt local search
def two_opt(route, dist_matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                if j-i == 1: continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                old_dist = sum(dist_matrix[route[k]][route[k+1]] for k in range(len(route)-1))
                new_dist = sum(dist_matrix[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))
                if new_dist < old_dist:
                    best = new_route
                    improved = True
        route = best
    return best

# Check capacity constraint
def check_capacity(route):
    total_demand = sum(nodes[i]['demand'] for i in route if i != 0)
    return total_demand <= VEHICLE_CAPACITY

# Calculate route distance
def route_distance(route):
    return sum(DIST_MATRIX[route[i]][route[i+1]] for i in range(len(route)-1))

# Priority-based solution construction
def construct_solution():
    # Generate random priorities (simulating X vector)
    X = np.random.rand(n + 2*m)  # n customer priorities + m xrefs + m yrefs

    # Step 1: Construct customer priority list
    customer_values = X[:n]
    U = np.argsort(customer_values)  # Indices 0 to n-1, representing customers 1 to n

    # Step 2: Construct vehicle priority matrix
    vehicle_refs = []
    for j in range(m):
        xref = X[n + j]
        yref = X[n + m + j]
        vehicle_refs.append((xref * 100, yref * 100))  # Scale to problem space

    V = []
    for i in range(n):
        distances = [math.sqrt((positions[i + 1][0] - vref[0])**2 + (positions[i + 1][1] - vref[1])**2)
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
            if loads[vehicle] + demand > VEHICLE_CAPACITY:
                continue

            # Cheapest insertion
            min_cost = float('inf')
            best_route = None
            for pos in range(len(route) + 1):
                prev = 0 if pos == 0 else route[pos - 1]
                next_node = 0 if pos == len(route) else route[pos]
                cost = DIST_MATRIX[prev, customer] + DIST_MATRIX[customer, next_node]
                if len(route) > 0:
                    cost -= DIST_MATRIX[prev, next_node]
                if cost < min_cost:
                    min_cost = cost
                    best_route = route[:pos] + [customer] + route[pos:]

            if best_route and loads[vehicle] + demand <= VEHICLE_CAPACITY:
                routes[vehicle] = best_route
                loads[vehicle] += demand
                assigned.add(customer)
                break

    # Convert to full routes (depot to depot)
    full_routes = []
    for route in routes:
        if route:
            full_route = [0] + route + [0]
            full_route = two_opt(full_route, DIST_MATRIX)
            if check_capacity(full_route):
                full_routes.append(full_route)
            else:
                full_routes.append([0, 0])  # Empty route if capacity violated
        else:
            full_routes.append([0, 0])

    # Penalize unassigned customers
    total_distance = sum(route_distance(r) for r in full_routes)
    if len(assigned) < n:
        total_distance += 10000 * (n - len(assigned))

    return full_routes, total_distance

# Mutation operation
def mutate_solution(routes, generation):
    pm = PMIN + (PMAX - PMIN) * (generation / MAX_GENERATIONS)
    if random.random() > pm:
        return routes
    
    new_routes = routes.copy()
    if len([r for r in routes if len(r) > 2]) < 2:
        return routes
    
    # Select two non-empty routes
    valid_routes = [i for i, r in enumerate(routes) if len(r) > 2]
    if len(valid_routes) < 2:
        return routes
    r1, r2 = random.sample(valid_routes, 2)
    
    # Select customers to swap
    route1, route2 = new_routes[r1], new_routes[r2]
    c1 = random.randint(1, len(route1)-2)
    c2 = random.randint(1, len(route2)-2)
    
    # Swap customers
    route1[c1], route2[c2] = route2[c2], route1[c1]
    
    # Fix capacity violations
    if not check_capacity(route1) or not check_capacity(route2):
        route1[c1], route2[c2] = route2[c2], route1[c1]  # Revert swap
        return routes
    
    # Apply 2-opt
    new_routes[r1] = two_opt(route1, DIST_MATRIX)
    new_routes[r2] = two_opt(route2, DIST_MATRIX)
    
    return new_routes

# Update pheromones
def update_pheromones(routes):
    global PHEROMONE
    # Evaporation
    PHEROMONE = PHEROMONE * RHO
    
    # Add new pheromones
    total_length = sum(route_distance(r) for r in routes)
    for route in routes:
        route_length = route_distance(route)
        num_customers = len([x for x in route if x != 0])
        if num_customers == 0:
            continue
        for i in range(len(route)-1):
            u, v = route[i], route[i+1]
            global_increment = Q / (len(routes) * total_length)
            local_increment = (route_length * DIST_MATRIX[u][v]) / (num_customers * route_length)
            delta = global_increment + local_increment
            PHEROMONE[u][v] += delta
            PHEROMONE[v][u] += delta
    
    # Apply bounds
    PHEROMONE = np.clip(PHEROMONE, SMIN, SMAX)

# Main IACO algorithm
def iaco_vrp():
    best_solution = None
    best_distance = float('inf')
    
    for generation in range(MAX_GENERATIONS):
        solutions = []
        for _ in range(NUM_ANTS):
            solution, distance = construct_solution()
            solution = mutate_solution(solution, generation)
            solutions.append((solution, distance))
            
            if distance < best_distance:
                best_solution = solution
                best_distance = distance
        
        # Print best distance every 100 iterations
        if (generation + 1) % 2== 0:
            print(f"Iteration {generation + 1}: Best total distance = {best_distance:.2f}")
        
        # Update pheromones for best solution
        update_pheromones(best_solution)
    
    return best_solution, best_distance

# Run algorithm
best_solution, best_distance = iaco_vrp()

# Print final results
print(f"Final best total distance: {best_distance:.2f}")
print("Routes:")
for i, route in enumerate(best_solution):
    customers_in_route = [nodes[j]['customer_number'] for j in route]
    total_demand = sum(nodes[j]['demand'] for j in route if j != 0)
    print(f"Route {i+1}: {customers_in_route}, Demand: {total_demand}")