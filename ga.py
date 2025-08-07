"""
Genetic Algorithm components for TSP solver.
This module contains all core components and algorithms for solving TSP using GA.

Author: gumocimo
Date: 07/08/2025
"""

import math
import random
import copy
from abc import ABC, abstractmethod


def get_fixed_cities():
    """Return a fixed set of cities for testing.

    Returns:
        list: List of (x, y) coordinate tuples for 10 cities
    """
    return [
        (60, 200), (180, 200), (80, 180), (140, 180), (20, 160),
        (100, 160), (200, 160), (140, 140), (40, 120), (100, 120)
    ]


def euclidean_distance(city1, city2):
    """Compute Euclidean distance between two cities.

    Args:
        city1: Tuple of (x, y) coordinates
        city2: Tuple of (x, y) coordinates

    Returns:
        float: Euclidean distance
    """
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def calculate_distance_matrix(cities):
    """Calculate distance matrix for all city pairs.

    Args:
        cities: List of (x, y) coordinate tuples

    Returns:
        list: 2D list of distances between cities
    """
    num_cities = len(cities)
    # Initialize distance matrix with zeros using nested lists
    dist_matrix = [[0.0 for _ in range(num_cities)] for _ in range(num_cities)]

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dist = euclidean_distance(cities[i], cities[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return dist_matrix


def calculate_tour_cost(tour, distance_matrix):
    """Calculate total cost (distance) for a complete tour.

    Args:
        tour: List of city indices representing the tour
        distance_matrix: 2D list of distances between cities

    Returns:
        float: Total tour distance
    """
    cost = 0.0
    num_cities = len(tour)
    for i in range(num_cities):
        cost += distance_matrix[tour[i]][tour[(i + 1) % num_cities]]
    return cost


# Individual Representation
class Individual:
    """Represents a solution (tour) in the GA population."""

    def __init__(self, tour):
        """Initialize an individual with a tour.

        Args:
            tour: List of city indices representing the tour
        """
        self.tour = list(tour)
        self.cost = float('inf')

    def calculate_cost(self, distance_matrix):
        """Calculate and store the tour cost.

        Args:
            distance_matrix: 2D list of distances between cities

        Returns:
            float: The calculated cost
        """
        self.cost = calculate_tour_cost(self.tour, distance_matrix)
        return self.cost

    def __lt__(self, other):
        """Enable sorting by cost."""
        return self.cost < other.cost

    def __repr__(self):
        """String representation for console output."""
        return f"Cost: {self.cost:.2f}, Tour: {self.tour}"


# Algorithm Base Class
class TSPAlgorithm(ABC):
    """Abstract base class for TSP solving algorithms."""

    def __init__(self, cities, distance_matrix):
        """Initialize algorithm with problem data.

        Args:
            cities: List of city coordinates
            distance_matrix: Pre-calculated distance matrix
        """
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.num_cities = len(cities)
        self.best_individual = None

    @abstractmethod
    def solve(self, **kwargs):
        """Solve the TSP problem.

        Returns:
            Individual: The best solution found
        """
        pass


# Standard Genetic Algorithm
class StandardGA(TSPAlgorithm):
    """Standard Genetic Algorithm for TSP."""

    def __init__(self, cities, distance_matrix):
        """Initialize SGA.

        Args:
            cities: List of city coordinates
            distance_matrix: Pre-calculated distance matrix
        """
        super().__init__(cities, distance_matrix)

    def initialize_population(self, population_size):
        """Create initial random population.

        Args:
            population_size: Number of individuals in population

        Returns:
            list: List of Individual objects
        """
        population = []
        base_tour = list(range(self.num_cities))

        for _ in range(population_size):
            tour = random.sample(base_tour, self.num_cities)
            population.append(Individual(tour))

        return population

    def selection_tournament(self, population, tournament_size):
        """Tournament selection operator.

        Args:
            population: Current population
            tournament_size: Number of individuals in each tournament

        Returns:
            list: Selected parents (mating pool)
        """
        selected_parents = []

        for _ in range(len(population)):
            aspirants = random.sample(population, tournament_size)
            winner = min(aspirants, key=lambda ind: ind.cost)
            selected_parents.append(winner)

        return selected_parents

    def crossover_ordered(self, parent1_ind, parent2_ind):
        """Ordered crossover (OX) operator.

        Args:
            parent1_ind: First parent Individual
            parent2_ind: Second parent Individual

        Returns:
            Individual: Offspring individual
        """
        parent1_tour = parent1_ind.tour
        parent2_tour = parent2_ind.tour
        size = len(parent1_tour)

        # Initialize child tour with placeholders
        child_tour = [-1] * size

        # Select random segment from parent1
        start, end = sorted(random.sample(range(size), 2))
        child_tour[start:end + 1] = parent1_tour[start:end + 1]

        # Fill remaining positions from parent2
        p2_idx = 0
        for i in range(size):
            if child_tour[i] == -1:
                # Find next city from parent2 not already in child
                while parent2_tour[p2_idx] in child_tour[start:end + 1]:
                    p2_idx += 1
                child_tour[i] = parent2_tour[p2_idx]
                p2_idx += 1

        return Individual(child_tour)

    def mutate_swap(self, individual, mutation_prob):
        """Swap mutation operator.

        Args:
            individual: Individual to mutate
            mutation_prob: Probability of mutation
        """
        if random.random() < mutation_prob:
            tour = individual.tour
            idx1, idx2 = random.sample(range(len(tour)), 2)
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]

    def solve(self, population_size, generations, crossover_rate,
              mutation_rate, tournament_size):
        """Run the SGA to solve TSP.

        Args:
            population_size: Size of population
            generations: Number of generations to run
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament for selection

        Returns:
            Individual: Best solution found
        """
        # Initialize population
        population = self.initialize_population(population_size)

        # Evaluate initial population
        for ind in population:
            ind.calculate_cost(self.distance_matrix)

        # Sort and track best
        population.sort()
        self.best_individual = copy.deepcopy(population[0])

        print(f"\nRunning SGA for {self.num_cities} cities")
        print(f"Parameters: Pop_Size={population_size}, Gens={generations}, "
              f"CR={crossover_rate}, MR={mutation_rate}, Tourn_K={tournament_size}")
        print(f"Initial best: {self.best_individual}")

        # Evolution loop
        for gen in range(1, generations + 1):
            new_population = []

            # Create mating pool
            mating_pool = self.selection_tournament(population, tournament_size)

            # Generate offspring
            offspring_idx = 0
            while len(new_population) < population_size:
                # Select parents
                parent1 = mating_pool[offspring_idx % len(mating_pool)]
                offspring_idx += 1
                parent2 = mating_pool[offspring_idx % len(mating_pool)]
                offspring_idx += 1

                # Apply crossover
                if random.random() < crossover_rate:
                    child = self.crossover_ordered(parent1, parent2)
                else:
                    # Clone one parent if no crossover
                    child = copy.deepcopy(random.choice([parent1, parent2]))

                # Apply mutation
                self.mutate_swap(child, mutation_rate)

                # Evaluate offspring
                child.calculate_cost(self.distance_matrix)
                new_population.append(child)

            # Replace population
            population = new_population
            population.sort()

            # Update best if improved
            if population[0].cost < self.best_individual.cost:
                self.best_individual = copy.deepcopy(population[0])

        print(f"SGA Final best: {self.best_individual}")
        return self.best_individual
