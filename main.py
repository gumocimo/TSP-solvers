#!/usr/bin/env python3
"""
Main entry point for TSP GA solver.

Author: gumocimo
Date: 11/08/2025
"""

import time
from ga import generate_cities, calculate_distance_matrix, StandardGA
from visualization import TSPPlotterSGA


# ============= Configuration Parameters =============
# Problem settings
NUM_CITIES = 50   # Number of cities
CITY_SEED = 1     # Seed for reproducible city generation
CITY_WIDTH = 100  # Grid width
CITY_HEIGHT = 100 # Grid height

# GA parameters (defaults, can be overridden based on problem size)
DEFAULT_SGA_POP_SIZE = 100        # Population size
DEFAULT_SGA_GENERATIONS = 1000    # Number of generations
DEFAULT_SGA_CROSSOVER_RATE = 0.85 # Crossover rate
DEFAULT_SGA_MUTATION_RATE = 0.15  # Mutation rate
DEFAULT_SGA_ELITISM_SIZE = 5      # Elitism size
DEFAULT_SGA_TOURNAMENT_K = 3      # Tournament size

# Visualization settings
LIVE_PLOT_UPDATE_FREQ = 1 # Update plot every N generations (0 to disable)

# Display settings
VERBOSE = True
PROGRESS_FREQUENCY = 10 # Print progress every N generations


# ============= Adaptive Parameters Function =============
def get_adaptive_parameters(num_cities):
    """Get parameters adapted to problem size.

    Args:
        num_cities: Number of cities in the problem

    Returns:
        tuple: (sga_params dict, plot_update_freq)
    """
    params = {
        "population_size": DEFAULT_SGA_POP_SIZE,
        "generations": DEFAULT_SGA_GENERATIONS,
        "crossover_rate": DEFAULT_SGA_CROSSOVER_RATE,
        "mutation_rate": DEFAULT_SGA_MUTATION_RATE,
        "elitism_size": DEFAULT_SGA_ELITISM_SIZE,
        "tournament_size": DEFAULT_SGA_TOURNAMENT_K
    }

    # Default plot frequency
    plot_freq = LIVE_PLOT_UPDATE_FREQ

    # Adapt parameters based on problem size
    if num_cities <= 50:
        params.update({
            "generations": 750,
            "population_size": 100
        })
        plot_freq = 1
    elif num_cities <= 100:
        params.update({
            "generations": 1500,
            "population_size": 200,
            "elitism_size": 10
        })
        plot_freq = 5
    else:
        params.update({
            "generations": 5000,
            "population_size": 250,
            "elitism_size": 15
        })
        plot_freq = 10

    return params, plot_freq


# ============= Main Execution =============
def main():
    """Run TSP solver with configured parameters."""
    # Generate cities
    cities = generate_cities(
        NUM_CITIES,
        width=CITY_WIDTH,
        height=CITY_HEIGHT,
        seed=CITY_SEED
    )

    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(cities)
    print(f"Generated {NUM_CITIES} cities. Distance matrix calculated.")

    # Get adaptive parameters
    sga_params, current_live_plot_freq = get_adaptive_parameters(NUM_CITIES)

    # Warn about plotting performance for large problems
    if (NUM_CITIES > 100 and
            current_live_plot_freq > 0 and
            current_live_plot_freq < 10):
        print(f"INFO: Live plot update frequency is {current_live_plot_freq} "
              f"for {NUM_CITIES} cities. This might be slow.")

    # Create plotter
    tsp_plotter = TSPPlotterSGA(cities)

    # Print parameters
    print(f"\nSGA Parameters: {sga_params}")

    # Create and run SGA
    sga = StandardGA(cities, distance_matrix)

    # Time the execution
    start_time = time.time()
    best_individual, cost_history = sga.solve(
        **sga_params,
        plotter=tsp_plotter,
        plot_freq=current_live_plot_freq
    )
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"SGA execution time: {execution_time:.2f} seconds")

    # Print final results
    print("\n" + "=" * 10 + " Final Results " + "=" * 10)
    print(f"Problem: {NUM_CITIES} cities (Seed: {CITY_SEED})")
    print(f"\nStandard GA (SGA):")
    print(f"  Best Cost: {best_individual.cost:.2f}")
    print(f"  Execution Time: {execution_time:.2f}s")

    # Update final plots
    tsp_plotter.display_execution_time(execution_time, cost_history)
    tsp_plotter.show_final_route(best_individual)
    tsp_plotter.convergence_ax.set_title("SGA Final Fitness Convergence")
    tsp_plotter.convergence_ax.legend(loc='upper right')

    print("\nCheck the plots for visual representation.")
    print("Close the plot window to end the script.")
    tsp_plotter.keep_plot_open()


if __name__ == "__main__":
    main()
