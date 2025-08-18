#!/usr/bin/env python3
"""Main entry point for TSP solver with integrated configuration.

Multi-algorithm comparison framework featuring Standard GA (SGA),
Hybrid GA-ACO (HGA-ACO), and Particle Swarm Optimization (PSO).

Author: gumocimo
Date: 18/08/2025
"""

import time
from core import generate_cities, calculate_distance_matrix
from algorithms.sga import StandardGA
from algorithms.hga_aco import HybridGA_ACO
from algorithms.pso import ParticleSwarmOptimization
from visualization import TSPPlotter


# ============= Algorithm Selection Toggles =============
ENABLE_SGA = True # Standard Genetic Algorithm
ENABLE_HGA_ACO = True # Hybrid GA-ACO
ENABLE_PSO = True # Particle Swarm Optimization


# ============= Problem Settings =============
NUM_CITIES = 50 # Number of cities (determines parameter preset: <50, 50-100, >100)
CITY_SEED = 1 # Seed for reproducible city generation
CITY_WIDTH = 100 # Grid width
CITY_HEIGHT = 100 # Grid height


# ============= Display Settings =============
VERBOSE = True
PROGRESS_FREQUENCY = 10


# ======================================================================
# PARAMETER PRESETS FOR DIFFERENT PROBLEM SIZES
# ======================================================================
# The system automatically selects parameters based on NUM_CITIES:
# - Small:  < 50 cities
# - Medium: 50-100 cities
# - Large:  > 100 cities
# ======================================================================

# Small problems (< 50 cities)
SMALL_PROBLEM_PARAMS = {
    # SGA parameters
    "SGA_POP_SIZE": 100,
    "SGA_GENERATIONS": 750,
    "SGA_CROSSOVER_RATE": 0.85,
    "SGA_MUTATION_RATE": 0.15,
    "SGA_ELITISM_SIZE": 5,
    "SGA_TOURNAMENT_K": 3,

    # HGA-ACO parameters
    "HGA_POP_SIZE": 50,
    "HGA_GENERATIONS": 250,
    "HGA_GA_CROSSOVER_RATE": 0.7,
    "HGA_ACO_CONTRIBUTION_RATE": 0.5,
    "HGA_MUTATION_RATE": 0.15,
    "HGA_ELITISM_SIZE": 5,
    "HGA_TOURNAMENT_K": 3,
    "HGA_ALPHA": 1.0,
    "HGA_BETA": 3.0,
    "HGA_EVAPORATION_RATE": 0.3,
    "HGA_Q_PHEROMONE": 100.0,
    "HGA_INITIAL_PHEROMONE": 0.1,
    "HGA_BEST_N_DEPOSIT": 3,

    # PSO parameters
    "PSO_NUM_PARTICLES": 15,
    "PSO_GENERATIONS": 250,
    "PSO_W": 0.4,
    "PSO_C1": 2.0,
    "PSO_C2": 2.0,
    "PSO_USE_LOCAL_SEARCH": True,

    # Visualization
    "LIVE_PLOT_UPDATE_FREQ": 1
}

# Medium problems (50-100 cities)
MEDIUM_PROBLEM_PARAMS = {
    # SGA parameters
    "SGA_POP_SIZE": 100,
    "SGA_GENERATIONS": 1500,
    "SGA_CROSSOVER_RATE": 0.85,
    "SGA_MUTATION_RATE": 0.15,
    "SGA_ELITISM_SIZE": 10,
    "SGA_TOURNAMENT_K": 3,

    # HGA-ACO parameters
    "HGA_POP_SIZE": 100,
    "HGA_GENERATIONS": 500,
    "HGA_GA_CROSSOVER_RATE": 0.7,
    "HGA_ACO_CONTRIBUTION_RATE": 0.5,
    "HGA_MUTATION_RATE": 0.15,
    "HGA_ELITISM_SIZE": 5,
    "HGA_TOURNAMENT_K": 3,
    "HGA_ALPHA": 1.0,
    "HGA_BETA": 3.0,
    "HGA_EVAPORATION_RATE": 0.3,
    "HGA_Q_PHEROMONE": 100.0,
    "HGA_INITIAL_PHEROMONE": 0.1,
    "HGA_BEST_N_DEPOSIT": 5,

    # PSO parameters
    "PSO_NUM_PARTICLES": 25,
    "PSO_GENERATIONS": 500,
    "PSO_W": 0.5,
    "PSO_C1": 2.0,
    "PSO_C2": 2.0,
    "PSO_USE_LOCAL_SEARCH": True,

    # Visualization
    "LIVE_PLOT_UPDATE_FREQ": 5
}

# Large problems (> 100 cities)
LARGE_PROBLEM_PARAMS = {
    # SGA parameters
    "SGA_POP_SIZE": 200,
    "SGA_GENERATIONS": 5000,
    "SGA_CROSSOVER_RATE": 0.85,
    "SGA_MUTATION_RATE": 0.20,
    "SGA_ELITISM_SIZE": 15,
    "SGA_TOURNAMENT_K": 5,

    # HGA-ACO parameters
    "HGA_POP_SIZE": 200,
    "HGA_GENERATIONS": 1000,
    "HGA_GA_CROSSOVER_RATE": 0.65,
    "HGA_ACO_CONTRIBUTION_RATE": 0.6,
    "HGA_MUTATION_RATE": 0.20,
    "HGA_ELITISM_SIZE": 10,
    "HGA_TOURNAMENT_K": 5,
    "HGA_ALPHA": 1.2,
    "HGA_BETA": 2.5,
    "HGA_EVAPORATION_RATE": 0.4,
    "HGA_Q_PHEROMONE": 100.0,
    "HGA_INITIAL_PHEROMONE": 0.05,
    "HGA_BEST_N_DEPOSIT": 10,

    # PSO parameters
    "PSO_NUM_PARTICLES": 30,
    "PSO_GENERATIONS": 1000,
    "PSO_W": 0.6,
    "PSO_C1": 1.8,
    "PSO_C2": 2.2,
    "PSO_USE_LOCAL_SEARCH": True,

    # Visualization
    "LIVE_PLOT_UPDATE_FREQ": 10
}


# ============= Custom Parameters (Optional) =============
# Set to None to use presets, or define custom parameters
CUSTOM_PARAMS = None

# Example custom parameters (uncomment and modify to use):
# CUSTOM_PARAMS = {
#     "SGA_POP_SIZE": 75,
#     "SGA_GENERATIONS": 250,
#     # ... add other parameters as needed
# }


# ============= Helper Functions =============
def get_enabled_algorithms():
    """Get dictionary of enabled algorithms.

    Returns:
        dict: Algorithm names mapped to enabled status
    """
    return {
        "SGA": ENABLE_SGA,
        "HGA-ACO": ENABLE_HGA_ACO,
        "PSO": ENABLE_PSO
    }


def get_problem_params(num_cities):
    """Get appropriate parameter set based on problem size.

    Args:
        num_cities: Number of cities in the problem

    Returns:
        dict: Parameter dictionary for the problem size
    """
    if num_cities < 50:
        return SMALL_PROBLEM_PARAMS
    elif num_cities <= 100:
        return MEDIUM_PROBLEM_PARAMS
    else:
        return LARGE_PROBLEM_PARAMS


def print_parameter_summary(params_dict, num_cities):
    """Print a summary of the parameters being used.

    Args:
        params_dict: Dictionary of all parameters
        num_cities: Number of cities in the problem
    """
    print("\n" + "="*30)
    print(f"PARAMETER CONFIGURATION FOR {num_cities} CITIES")
    print("="*30)

    if num_cities < 50:
        print("Problem Size: SMALL (< 50 cities)")
    elif num_cities <= 100:
        print("Problem Size: MEDIUM (50-100 cities)")
    else:
        print("Problem Size: LARGE (> 100 cities)")

    print("\nKey Parameters:")
    print(f"  Visualization Update: Every {params_dict.get('LIVE_PLOT_UPDATE_FREQ', 10)} generations")

    if ENABLE_SGA:
        print("\nSGA Parameters:")
        print(f"  Population: {params_dict.get('SGA_POP_SIZE', 'N/A')}")
        print(f"  Generations: {params_dict.get('SGA_GENERATIONS', 'N/A')}")
        print(f"  Crossover Rate: {params_dict.get('SGA_CROSSOVER_RATE', 'N/A')}")
        print(f"  Mutation Rate: {params_dict.get('SGA_MUTATION_RATE', 'N/A')}")

    if ENABLE_HGA_ACO:
        print("\nHGA-ACO Parameters:")
        print(f"  Population: {params_dict.get('HGA_POP_SIZE', 'N/A')}")
        print(f"  Generations: {params_dict.get('HGA_GENERATIONS', 'N/A')}")
        print(f"  ACO Contribution: {params_dict.get('HGA_ACO_CONTRIBUTION_RATE', 'N/A')}")
        print(f"  Alpha/Beta: {params_dict.get('HGA_ALPHA', 'N/A')}/{params_dict.get('HGA_BETA', 'N/A')}")

    if ENABLE_PSO:
        print("\nPSO Parameters:")
        print(f"  Particles: {params_dict.get('PSO_NUM_PARTICLES', 'N/A')}")
        print(f"  Generations: {params_dict.get('PSO_GENERATIONS', 'N/A')}")
        print(f"  Inertia (w): {params_dict.get('PSO_W', 'N/A')}")
        print(f"  Local Search: {'Enabled' if params_dict.get('PSO_USE_LOCAL_SEARCH', True) else 'Disabled'}")

    print("="*50 + "\n")


def get_algorithm_parameters(num_cities):
    """Get algorithm parameters based on problem size.

    Args:
        num_cities: Number of cities in the problem

    Returns:
        tuple: (sga_params, hga_params, pso_params, plot_freq)
    """
    # Get appropriate parameter set
    if CUSTOM_PARAMS:
        params = CUSTOM_PARAMS
        print(f"Using custom parameters")
    else:
        params = get_problem_params(num_cities)
        if num_cities < 50:
            print(f"Using parameters for small problems (< 50 cities)")
        elif num_cities <= 100:
            print(f"Using parameters for medium problems (50-100 cities)")
        else:
            print(f"Using parameters for large problems (> 100 cities)")

    # Extract SGA parameters
    sga_params = {
        "population_size": params.get("SGA_POP_SIZE", 100),
        "generations": params.get("SGA_GENERATIONS", 200),
        "crossover_rate": params.get("SGA_CROSSOVER_RATE", 0.85),
        "mutation_rate": params.get("SGA_MUTATION_RATE", 0.15),
        "elitism_size": params.get("SGA_ELITISM_SIZE", 5),
        "tournament_size": params.get("SGA_TOURNAMENT_K", 3)
    }

    # Extract HGA-ACO parameters
    hga_params = {
        "population_size": params.get("HGA_POP_SIZE", 100),
        "generations": params.get("HGA_GENERATIONS", 200),
        "ga_crossover_rate": params.get("HGA_GA_CROSSOVER_RATE", 0.7),
        "aco_contribution_rate": params.get("HGA_ACO_CONTRIBUTION_RATE", 0.5),
        "mutation_rate": params.get("HGA_MUTATION_RATE", 0.15),
        "elitism_size": params.get("HGA_ELITISM_SIZE", 5),
        "tournament_size": params.get("HGA_TOURNAMENT_K", 3),
        "alpha": params.get("HGA_ALPHA", 1.0),
        "beta": params.get("HGA_BETA", 3.0),
        "evaporation_rate": params.get("HGA_EVAPORATION_RATE", 0.3),
        "Q_pheromone": params.get("HGA_Q_PHEROMONE", 100.0),
        "initial_pheromone_val": params.get("HGA_INITIAL_PHEROMONE", 0.1),
        "best_n_deposit": params.get("HGA_BEST_N_DEPOSIT", 5)
    }

    # Extract PSO parameters
    pso_params = {
        "num_particles": params.get("PSO_NUM_PARTICLES", 30),
        "generations": params.get("PSO_GENERATIONS", 200),
        "w": params.get("PSO_W", 0.5),
        "c1": params.get("PSO_C1", 2.0),
        "c2": params.get("PSO_C2", 2.0),
        "use_local_search": params.get("PSO_USE_LOCAL_SEARCH", True)
    }

    # Get plot frequency
    plot_freq = params.get("LIVE_PLOT_UPDATE_FREQ", 10)

    return sga_params, hga_params, pso_params, plot_freq


# ============= Main Execution =============
def main():
    """Run TSP solver with selected algorithms."""
    # Check which algorithms are enabled
    enabled_algorithms = get_enabled_algorithms()
    num_enabled = sum(enabled_algorithms.values())

    if num_enabled == 0:
        print("Error: No algorithms enabled! Enable at least one algorithm in the configuration section.")
        return

    # Generate cities
    cities = generate_cities(
        NUM_CITIES,
        width=CITY_WIDTH,
        height=CITY_HEIGHT,
        seed=CITY_SEED
    )

    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(cities)

    print("Multi-Algorithm TSP Solver")
    print(f"Comparing: {', '.join([algo for algo, enabled in enabled_algorithms.items() if enabled])}")
    print(f"\nGenerated {NUM_CITIES} cities. Distance matrix calculated.")

    # Get algorithm parameters
    sga_params, hga_params, pso_params, plot_freq = get_algorithm_parameters(NUM_CITIES)

    # Print parameter summary
    all_params = CUSTOM_PARAMS if CUSTOM_PARAMS else get_problem_params(NUM_CITIES)
    print_parameter_summary(all_params, NUM_CITIES)

    # Create plotter with enabled algorithms
    tsp_plotter = TSPPlotter(cities, enabled_algorithms)

    # Store results
    results = {}
    exec_times = {}

    # Run SGA if enabled
    if enabled_algorithms["SGA"]:
        sga = StandardGA(cities, distance_matrix)

        start_time = time.time()
        sga_best, sga_history = sga.solve(**sga_params, plotter=tsp_plotter, plot_freq=plot_freq)
        end_time = time.time()

        exec_times["SGA"] = end_time - start_time
        results["SGA"] = sga_best
        print(f"SGA execution time: {exec_times['SGA']:.2f} seconds")

    # Run HGA-ACO if enabled
    if enabled_algorithms["HGA-ACO"]:
        hga = HybridGA_ACO(cities, distance_matrix)

        start_time = time.time()
        hga_best, hga_history = hga.solve(**hga_params, plotter=tsp_plotter, plot_freq=plot_freq)
        end_time = time.time()

        exec_times["HGA-ACO"] = end_time - start_time
        results["HGA-ACO"] = hga_best
        print(f"HGA-ACO execution time: {exec_times['HGA-ACO']:.2f} seconds")

    # Run PSO if enabled
    if enabled_algorithms["PSO"]:
        pso = ParticleSwarmOptimization(cities, distance_matrix)

        start_time = time.time()
        pso_best, pso_history = pso.solve(**pso_params, plotter=tsp_plotter, plot_freq=plot_freq)
        end_time = time.time()

        exec_times["PSO"] = end_time - start_time
        results["PSO"] = pso_best
        print(f"PSO execution time: {exec_times['PSO']:.2f} seconds")

    # Final comparison
    print("\n" + "="*30 + " Final Comparison " + "="*30)
    print(f"Problem: {NUM_CITIES} cities (Seed: {CITY_SEED})")

    # Display results for each algorithm
    for algo, best_ind in results.items():
        print(f"\n{algo}:")
        print(f"  Best Cost: {best_ind.cost:.2f}")
        print(f"  Execution Time: {exec_times[algo]:.2f}s")

    # Find overall best
    best_algo = min(results.keys(), key=lambda a: results[a].cost)
    print(f"\nBest solution found by: {best_algo}")

    # Compare algorithms pairwise
    if num_enabled > 1:
        print("\nPairwise comparisons:")
        algo_list = list(results.keys())
        for i in range(len(algo_list)):
            for j in range(i + 1, len(algo_list)):
                algo1, algo2 = algo_list[i], algo_list[j]
                diff = results[algo1].cost - results[algo2].cost
                if diff > 0:
                    print(f"  {algo2} beat {algo1} by {diff:.2f}")
                elif diff < 0:
                    print(f"  {algo1} beat {algo2} by {-diff:.2f}")
                else:
                    print(f"  {algo1} and {algo2} found equal solutions")

    # Update final plots
    tsp_plotter.display_execution_times(exec_times)
    tsp_plotter.show_final_routes(results)
    tsp_plotter.show_performance_comparison(results, exec_times)
    tsp_plotter.convergence_ax.set_title("Final Fitness Convergence Comparison")

    print("\nCheck the plots for visual comparison.")
    print("Close the plot window to end the script.")
    tsp_plotter.keep_plot_open()


if __name__ == "__main__":
    main()
