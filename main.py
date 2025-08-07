#!/usr/bin/env python3
"""
Main entry point for TSP GA solver.

Author: gumocimo
Date: 07/08/2025
"""

from ga import get_fixed_cities, calculate_distance_matrix, StandardGA

# GA Parameters
POP_SIZE = 50
GENERATIONS = 100
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.15
TOURNAMENT_K = 3

# Display settings
VERBOSE = True


def main():
    """Run TSP solver with configured parameters."""
    # Get cities
    cities = get_fixed_cities()
    num_cities = len(cities)

    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(cities)
    print(f"Problem: {num_cities} fixed cities. Distance matrix calculated.")

    # Create and run SGA
    sga = StandardGA(cities, distance_matrix)
    best_individual = sga.solve(
        population_size=POP_SIZE,
        generations=GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        tournament_size=TOURNAMENT_K
    )

    # Print final results
    print("\n" + "=" * 10 + " SGA Run Complete " + "=" * 10)
    print(f"Final Best Individual found:")
    print(f"  Cost: {best_individual.cost:.2f}")
    print(f"  Tour: {best_individual.tour}")
    print("Execution finished.")


if __name__ == "__main__":
    main()
