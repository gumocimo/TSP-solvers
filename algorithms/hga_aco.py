"""Hybrid Genetic Algorithm - Ant Colony Optimization implementation.

Author: gumocimo
Date: 13/08/2025
"""

import random
import copy
import numpy as np
from algorithms.base import TSPAlgorithm
from core import Individual


class HybridGA_ACO(TSPAlgorithm):
    """Hybrid GA-ACO Algorithm for TSP."""

    def __init__(self, cities, distance_matrix):
        """Initialize HGA-ACO.

        Args:
            cities: NumPy array of city coordinates
            distance_matrix: Pre-calculated distance matrix
        """
        super().__init__(cities, distance_matrix)
        self.pheromone_matrix = None

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

    def initialize_pheromones(self, initial_value):
        """Initialize pheromone matrix.

        Args:
            initial_value: Initial pheromone value for all edges

        Returns:
            np.ndarray: Initialized pheromone matrix
        """
        initial_value = max(initial_value, 1e-6) # Ensure positive
        return np.full((self.num_cities, self.num_cities), initial_value)

    def construct_aco_tour(self, alpha, beta):
        """Construct a tour using ACO principles.

        Args:
            alpha: Pheromone influence factor
            beta: Heuristic (distance) influence factor

        Returns:
            Individual: Constructed tour
        """
        tour = []
        remaining_cities = list(range(self.num_cities))

        # Start at random city
        current_city = random.choice(remaining_cities)
        tour.append(current_city)
        remaining_cities.remove(current_city)

        # Build tour probabilistically
        while remaining_cities:
            probabilities = []
            prob_sum = 0.0

            for next_city in remaining_cities:
                # Calculate probability based on pheromone and distance
                pheromone = self.pheromone_matrix[current_city, next_city]
                distance = self.distance_matrix[current_city, next_city]

                # Heuristic value (inverse of distance)
                heuristic = (1.0 / (distance + 1e-10)) ** beta

                # Probability calculation
                prob = (max(pheromone, 1e-6) ** alpha) * heuristic
                probabilities.append(prob)
                prob_sum += prob

            # Choose next city
            if prob_sum == 0 or not remaining_cities:
                if not remaining_cities:
                    break
                chosen_city = random.choice(remaining_cities)
            else:
                # Roulette wheel selection
                probabilities_norm = np.array(probabilities) / prob_sum
                chosen_city = np.random.choice(remaining_cities, p=probabilities_norm)

            tour.append(chosen_city)
            remaining_cities.remove(chosen_city)
            current_city = chosen_city

        return Individual(tour)

    def update_pheromones(self, population, evaporation_rate, Q, best_n_deposit):
        """Update pheromone matrix based on population.

        Args:
            population: Current population (should be sorted by cost)
            evaporation_rate: Pheromone evaporation rate (rho)
            Q: Pheromone deposit constant
            best_n_deposit: Number of best individuals to deposit pheromones
        """
        # Evaporate pheromones
        self.pheromone_matrix *= (1.0 - evaporation_rate)

        # Deposit pheromones from best individuals
        sorted_pop = sorted(population, key=lambda ind: ind.cost)
        num_depositing = min(best_n_deposit, len(sorted_pop))

        for i in range(num_depositing):
            individual = sorted_pop[i]
            if individual.cost == 0:
                continue # Avoid division by zero

            deposit_amount = Q / individual.cost
            tour = individual.tour

            # Deposit on all edges in tour
            for j in range(len(tour)):
                city1 = tour[j]
                city2 = tour[(j + 1) % len(tour)]
                self.pheromone_matrix[city1, city2] += deposit_amount
                self.pheromone_matrix[city2, city1] += deposit_amount # Symmetric

        # Ensure minimum pheromone level
        min_pheromone = 1e-6
        self.pheromone_matrix[self.pheromone_matrix < min_pheromone] = min_pheromone

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

        child_tour = [-1] * size
        start, end = sorted(random.sample(range(size), 2))
        child_tour[start:end + 1] = parent1_tour[start:end + 1]

        p2_idx = 0
        for i in range(size):
            if child_tour[i] == -1:
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

    def solve(self, population_size, generations, ga_crossover_rate,
              aco_contribution_rate, mutation_rate, elitism_size,
              tournament_size, alpha, beta, evaporation_rate,
              Q_pheromone, initial_pheromone_val, best_n_deposit,
              plotter=None, plot_freq=1):
        """Run the HGA-ACO to solve TSP.

        Args:
            population_size: Size of population
            generations: Number of generations to run
            ga_crossover_rate: Crossover rate for GA portion
            aco_contribution_rate: Proportion of ACO-constructed individuals
            mutation_rate: Probability of mutation
            elitism_size: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
            alpha: Pheromone influence
            beta: Heuristic influence
            evaporation_rate: Pheromone evaporation rate
            Q_pheromone: Pheromone deposit constant
            initial_pheromone_val: Initial pheromone value
            best_n_deposit: Number of best individuals to deposit pheromones
            plotter: Optional plotter instance for visualization
            plot_freq: Update frequency for live plotting

        Returns:
            tuple: (best_individual, cost_history)
        """
        # Initialize population and pheromones
        population = self.initialize_population(population_size)
        self.pheromone_matrix = self.initialize_pheromones(initial_pheromone_val)

        # Evaluate initial population
        for ind in population:
            ind.calculate_cost(self.distance_matrix)

        # Sort and track best
        population.sort()
        self.best_individual = copy.deepcopy(population[0])
        self.cost_history = [self.best_individual.cost]

        algo_name = "HGA-ACO"
        print(f"\n--- Running {algo_name} for {self.num_cities} cities ---")
        print(f"Initial best cost: {self.best_individual.cost:.2f}")

        # Initial plot update
        if plotter:
            plotter.update_live_route_plot(
                self.best_individual.tour, algo_name, 0,
                self.best_individual.cost, plot_freq
            )

        # Evolution loop
        for gen in range(1, generations + 1):
            new_population = []

            # Elitism
            if elitism_size > 0:
                elites = copy.deepcopy(population[:elitism_size])
                new_population.extend(elites)

            # Calculate offspring split
            num_offspring_needed = population_size - len(new_population)
            num_from_aco = int(num_offspring_needed * aco_contribution_rate)
            num_from_ga = num_offspring_needed - num_from_aco

            # ACO-constructed individuals
            for _ in range(num_from_aco):
                child = self.construct_aco_tour(alpha, beta)
                self.mutate_swap(child, mutation_rate)
                child.calculate_cost(self.distance_matrix)
                new_population.append(child)

            # GA-constructed individuals
            if num_from_ga > 0:
                mating_pool = self.selection_tournament(population, tournament_size)
                offspring_idx = 0

                while len(new_population) < population_size:
                    if not mating_pool:
                        break

                    parent1 = mating_pool[offspring_idx % len(mating_pool)]
                    offspring_idx += 1
                    parent2 = mating_pool[offspring_idx % len(mating_pool)]
                    offspring_idx += 1

                    if random.random() < ga_crossover_rate:
                        child = self.crossover_ordered(parent1, parent2)
                    else:
                        child = copy.deepcopy(random.choice([parent1, parent2]))

                    self.mutate_swap(child, mutation_rate)
                    child.calculate_cost(self.distance_matrix)
                    new_population.append(child)

            # Replace population
            population = new_population
            population.sort()

            # Update best
            if population[0].cost < self.best_individual.cost:
                self.best_individual = copy.deepcopy(population[0])

            self.cost_history.append(self.best_individual.cost)

            # Update pheromones
            self.update_pheromones(population, evaporation_rate, Q_pheromone, best_n_deposit)

            # Progress output
            if gen % 10 == 0 or gen == generations:
                print(f"{algo_name} Gen {gen}/{generations} - Best Cost: {self.best_individual.cost:.2f}")

            # Update plots
            if plotter:
                plotter.update_live_route_plot(
                    self.best_individual.tour, algo_name, gen,
                    self.best_individual.cost, plot_freq
                )
                plotter.update_convergence_plot(self.cost_history, algo_name, "green")

        print(f"{algo_name} Final Best Tour: {self.best_individual.tour} with Cost: {self.best_individual.cost:.2f}")

        # Final plot update
        if plotter:
            plotter.update_live_route_plot(
                self.best_individual.tour, algo_name, generations,
                self.best_individual.cost, plot_freq
            )

        return self.best_individual, self.cost_history
