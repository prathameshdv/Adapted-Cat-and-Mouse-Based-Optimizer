# Adapted Cat and Mouse Based Optimizer (ACMBO) for Capacitated Vehicle Routing Problem (CVRP)

This repository presents the **Adapted Cat and Mouse Based Optimizer (ACMBO)** algorithm, a metaheuristic designed to efficiently solve the **Capacitated Vehicle Routing Problem (CVRP)**. Our work on ACMBO has been accepted for presentation at the IEEE International Conference of Optimization and Algorithms (ICOA) 2025 in Kenitra, Morocco  (https://www.icoa-conf.org/).
## Project Overview

The Capacitated Vehicle Routing Problem (CVRP) is a fundamental challenge in transportation and logistics, critical for optimizing operations like goods distribution and waste collection. Exact algorithms struggle with the computational complexity of large-scale CVRP instances, leading to the widespread use of heuristic and metaheuristic approaches to find near-optimal solutions.

This project introduces ACMBO, an enhancement of the original Cat and Mouse Based Optimizer (CMBO), specifically tailored for the discrete nature of CVRP. ACMBO incorporates several key modifications to effectively explore the solution space and avoid local optima.

## Key Features

* **Discrete Solution Representation:** Employs customer priorities and vehicle reference points to encode feasible routes.
* **Feasibility-Ensuring Decoding Mechanism:** Guarantees adherence to vehicle capacity constraints.
* **Dynamic Customer Segmentation:** Utilizes K-means clustering to adaptively partition the search space, enhancing exploration and mitigating premature convergence.
* **Elite Archive:** Maintains a collection of high-quality solutions to guide the search process.
* **Local Search Operators:** Integrates powerful local search mechanisms, including:
    * **Variable Neighborhood Descent (VND):** Refines routes using operators like Intra-Insert, Intra-Change, Intra-Reverse, 3-Opt, Inter-Insert, and Inter-Change.
    * **Destroy-and-Repair:** A mechanism triggered by stagnation to perturb existing solutions and escape local optima by intelligently removing and reinserting customers.
* **Fitness Sharing:** Promotes diversity within clusters by penalizing similar solutions.

## Performance and Results

ACMBO was rigorously tested on 35 benchmark instances from CVRPLIB and compared against well-established metaheuristic algorithms like Particle Swarm Optimization (PSO) and Ant Colony Optimization (ACO).

* **Superior Performance:** ACMBO consistently outperformed or matched PSO and ACO in 29 out of 35 instances (83%), yielding lower objective values.
* **High Precision:** Achieved optimal Best Known Solutions (BKS) in seven instances and surpassed BKS in two instances (-2.49% and -4.87% deviation).
* **Scalability:** Demonstrated effectiveness across a wide range of problem scales, from small (16 customers) to large (150 customers) instances.

These results highlight ACMBO's robustness, precision, and consistency, making it a highly competitive approach for solving complex CVRP challenges.

## Future Work

Future enhancements and applications of ACMBO include:

* **Solving VRP Variants:** Adapting ACMBO to address other variations of the Vehicle Routing Problem, such as Electric VRP, VRP with time-windows, and multiple-depot VRP.
* **Real-world Applications:** Validating ACMBO's practical applicability by optimizing routes for e-commerce services with real-time data and dynamic constraints.

