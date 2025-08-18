"""Particle Swarm Optimization implementation for TSP.

Author: gumocimo
Date: 18/08/2025
"""

import random
import copy
import numpy as np
from algorithms.base import TSPAlgorithm
from core import Individual, calculate_tour_cost


class Particle:
    """Represents a particle in the swarm."""

    def __init__(self, tour):
        """Initialize a particle with a tour.

        Args:
            tour: Initial tour (position)
        """
        self.position = list(tour) # Current tour
        self.velocity = [] # List of swaps to perform
        self.best_position = list(tour) # Personal best tour
        self.best_cost = float('inf')
        self.cost = float('inf')

    def update_personal_best(self):
        """Update personal best if current position is better."""
        if self.cost < self.best_cost:
            self.best_position = list(self.position)
            self.best_cost = self.cost


class ParticleSwarmOptimization(TSPAlgorithm):
    """Particle Swarm Optimization for TSP."""

    def __init__(self, cities, distance_matrix):
        """Initialize PSO.

        Args:
            cities: NumPy array of city coordinates
            distance_matrix: Pre-calculated distance matrix
        """
        super().__init__(cities, distance_matrix)
        self.global_best_position = None
        self.global_best_cost = float('inf')

    def initialize_swarm(self, num_particles):
        """Create initial swarm with random tours.

        Args:
            num_particles: Number of particles in swarm

        Returns:
            list: List of Particle objects
        """
        swarm = []
        base_tour = list(range(self.num_cities))

        for _ in range(num_particles):
            tour = random.sample(base_tour, self.num_cities)
            particle = Particle(tour)
            swarm.append(particle)

        return swarm

    def evaluate_particle(self, particle):
        """Evaluate particle's current position (tour).

        Args:
            particle: Particle to evaluate
        """
        # Calculate cost directly
        particle.cost = calculate_tour_cost(particle.position, self.distance_matrix)

    def generate_swap_sequence(self, tour1, tour2):
        """Generate sequence of swaps to transform tour1 to tour2.

        Args:
            tour1: Source tour
            tour2: Target tour

        Returns:
            list: List of (i, j) swap operations
        """
        tour1_copy = list(tour1)
        swaps = []

        # Only generate first few swaps to avoid too many changes
        max_swaps = min(5, len(tour1) // 10)

        for i in range(len(tour1)):
            if len(swaps) >= max_swaps:
                break

            if tour1_copy[i] != tour2[i]:
                # Find where tour2[i] is in tour1_copy
                j = tour1_copy.index(tour2[i])
                # Record swap
                swaps.append((i, j))
                # Perform swap
                tour1_copy[i], tour1_copy[j] = tour1_copy[j], tour1_copy[i]

        return swaps

    def subtract_positions(self, pos1, pos2):
        """Calculate velocity (swap sequence) between two positions.

        Args:
            pos1: Target position
            pos2: Current position

        Returns:
            list: Velocity as swap sequence
        """
        return self.generate_swap_sequence(pos2, pos1)

    def add_velocities(self, vel1, vel2, weight1=1.0, weight2=1.0):
        """Combine two velocities with weights.

        Args:
            vel1: First velocity
            vel2: Second velocity
            weight1: Weight for first velocity
            weight2: Weight for second velocity

        Returns:
            list: Combined velocity
        """
        combined = []

        # Add swaps from vel1 with probability based on weight1
        for swap in vel1:
            if random.random() < weight1:
                combined.append(swap)

        # Add swaps from vel2 with probability based on weight2
        for swap in vel2:
            if random.random() < weight2:
                combined.append(swap)

        return combined

    def apply_velocity(self, position, velocity):
        """Apply velocity (swaps) to position.

        Args:
            position: Current tour
            velocity: List of swaps to apply

        Returns:
            list: New position after applying velocity
        """
        new_position = list(position)

        for (i, j) in velocity:
            if i < len(new_position) and j < len(new_position):
                new_position[i], new_position[j] = new_position[j], new_position[i]

        return new_position

    def two_opt_improvement(self, tour, max_iterations=5):
        """Apply 2-opt local search to improve tour.

        Args:
            tour: Current tour
            max_iterations: Maximum improvement iterations

        Returns:
            list: Improved tour
        """
        improved = True
        best_tour = list(tour)
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour)):
                    if j - i == 1:
                        continue

                    new_tour = best_tour[:]
                    new_tour[i:j] = reversed(new_tour[i:j])

                    # Calculate improvement
                    current_cost = calculate_tour_cost(best_tour, self.distance_matrix)
                    new_cost = calculate_tour_cost(new_tour, self.distance_matrix)

                    if new_cost < current_cost:
                        best_tour = new_tour
                        improved = True
                        break
                if improved:
                    break

        return best_tour

    def update_particle(self, particle, w, c1, c2):
        """Update particle velocity and position.

        Args:
            particle: Particle to update
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        """
        # Calculate cognitive velocity (towards personal best)
        cognitive_vel = self.subtract_positions(
            particle.best_position,
            particle.position
        )

        # Calculate social velocity (towards global best)
        social_vel = self.subtract_positions(
            self.global_best_position,
            particle.position
        )

        # Combine velocities with improved method
        new_velocity = []

        # Keep some inertia from current velocity (reduced randomness)
        num_keep = int(len(particle.velocity) * w)
        if num_keep > 0 and particle.velocity:
            new_velocity.extend(random.sample(particle.velocity, min(num_keep, len(particle.velocity))))

        # Add cognitive component
        r1 = random.random()
        num_cognitive = int(len(cognitive_vel) * c1 * r1)
        if num_cognitive > 0 and cognitive_vel:
            new_velocity.extend(random.sample(cognitive_vel, min(num_cognitive, len(cognitive_vel))))

        # Add social component
        r2 = random.random()
        num_social = int(len(social_vel) * c2 * r2)
        if num_social > 0 and social_vel:
            new_velocity.extend(random.sample(social_vel, min(num_social, len(social_vel))))

        # Remove duplicate swaps
        seen = set()
        unique_velocity = []
        for swap in new_velocity:
            if swap not in seen:
                seen.add(swap)
                unique_velocity.append(swap)

        # Limit velocity size more conservatively
        max_velocity_size = max(2, self.num_cities // 10)
        if len(unique_velocity) > max_velocity_size:
            unique_velocity = unique_velocity[:max_velocity_size]

        # Update velocity and position
        particle.velocity = unique_velocity
        particle.position = self.apply_velocity(particle.position, particle.velocity)

    def solve(self, num_particles, generations, w, c1, c2, use_local_search=True,
              plotter=None, plot_freq=1):
        """Run PSO to solve TSP.

        Args:
            num_particles: Number of particles in swarm
            generations: Number of iterations
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
            use_local_search: Whether to apply 2-opt improvement
            plotter: Optional plotter instance
            plot_freq: Update frequency for live plotting

        Returns:
            tuple: (best_individual, cost_history)
        """
        # Initialize swarm
        swarm = self.initialize_swarm(num_particles)

        # Evaluate initial swarm
        for particle in swarm:
            # Apply local search to initial positions if enabled
            if use_local_search:
                particle.position = self.two_opt_improvement(particle.position, max_iterations=10)

            self.evaluate_particle(particle)
            particle.update_personal_best()

            # Update global best
            if particle.cost < self.global_best_cost:
                self.global_best_cost = particle.cost
                self.global_best_position = list(particle.position)

        # Track best for return
        self.best_individual = Individual(self.global_best_position)
        self.best_individual.cost = self.global_best_cost
        self.cost_history = [self.global_best_cost]

        algo_name = "PSO"
        print(f"\n--- Running {algo_name} for {self.num_cities} cities ---")
        print(f"Initial best cost: {self.global_best_cost:.2f}")
        if use_local_search:
            print("Using 2-opt local search improvement")

        # Initial plot update
        if plotter:
            plotter.update_live_route_plot(
                self.global_best_position, algo_name, 0,
                self.global_best_cost, plot_freq
            )

        # Main PSO loop
        for gen in range(1, generations + 1):
            # Update each particle
            for particle in swarm:
                self.update_particle(particle, w, c1, c2)

                # Apply local search periodically if enabled
                if use_local_search and gen % 10 == 0:
                    particle.position = self.two_opt_improvement(particle.position, max_iterations=3)

                self.evaluate_particle(particle)
                particle.update_personal_best()

                # Update global best
                if particle.cost < self.global_best_cost:
                    self.global_best_cost = particle.cost
                    self.global_best_position = list(particle.position)

            # Update best individual
            self.best_individual = Individual(self.global_best_position)
            self.best_individual.cost = self.global_best_cost
            self.cost_history.append(self.global_best_cost)

            # Progress output
            if gen % 10 == 0 or gen == generations:
                print(f"{algo_name} Gen {gen}/{generations} - Best Cost: {self.global_best_cost:.2f}")

            # Update plots
            if plotter:
                if gen % plot_freq == 0 or gen == generations:
                    plotter.update_live_route_plot(
                        self.global_best_position, algo_name, gen,
                        self.global_best_cost, plot_freq
                    )
                plotter.update_convergence_plot(self.cost_history, algo_name, "orange")

        print(f"{algo_name} Final Best Tour: {self.best_individual.tour} with Cost: {self.best_individual.cost:.2f}")

        # Final plot update
        if plotter:
            plotter.update_live_route_plot(
                self.best_individual.tour, algo_name, generations,
                self.best_individual.cost, plot_freq
            )

        return self.best_individual, self.cost_history
