"""Visualization components for TSP GA.

This module provides real-time plotting utilities for visualizing multiple algorithms' 
progress in solving TSP, including route evolution and fitness convergence.

Author: gumocimo
Date: 13/08/2025
"""

import matplotlib.pyplot as plt
import numpy as np


class TSPPlotter:
    """Plotter for visualizing multiple algorithms on TSP."""

    def __init__(self, cities):
        """Initialize the plotter with city data.

        Args:
            cities: NumPy array of city coordinates
        """
        self.cities = cities
        plt.ion() # Enable interactive mode

        # Create subplots
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 5))
        self.route_ax = self.ax[0]
        self.convergence_ax = self.ax[1]

        # Setup plots
        self._setup_route_plot()
        self._setup_convergence_plot()

        # Track lines for each algorithm
        self.convergence_lines = {}
        self.route_lines = {}

    def _setup_route_plot(self):
        """Setup the route visualization plot."""
        # Plot cities
        self.route_ax.scatter(
            self.cities[:, 0],
            self.cities[:, 1],
            c='red',
            marker='o',
            label='Cities',
            zorder=5
        )

        # Add city labels
        for i, city_coord in enumerate(self.cities):
            self.route_ax.text(
                city_coord[0] + 0.5,
                city_coord[1] + 0.5,
                str(i),
                fontsize=9
            )

        self.route_ax.set_xlabel("X-coordinate")
        self.route_ax.set_ylabel("Y-coordinate")
        self.route_ax.legend(loc='upper right')

    def _setup_convergence_plot(self):
        """Setup the convergence visualization plot."""
        self.convergence_ax.set_xlabel("Generation")
        self.convergence_ax.set_ylabel("Best Cost (Distance)")
        self.convergence_ax.set_title("Fitness Convergence")

    def update_live_route_plot(self, best_tour_indices, algo_name,
                               generation, best_cost, update_freq):
        """Update the route plot with current best tour.

        Args:
            best_tour_indices: List of city indices in best tour
            algo_name: Name of the algorithm
            generation: Current generation number
            best_cost: Cost of the best tour
            update_freq: Update frequency setting
        """
        # Check if update is needed
        if not update_freq or generation % update_freq != 0:
            if generation != 1 and generation != -1: # Always show first and final
                return

        # Remove old route line for this algorithm
        if algo_name in self.route_lines and self.route_lines[algo_name]:
            self.route_lines[algo_name].pop(0).remove()

        # Create tour coordinates
        tour_coords = np.array([
            self.cities[i] for i in best_tour_indices + [best_tour_indices[0]]
        ])

        # Choose color based on algorithm
        line_color = 'blue' if "SGA" in algo_name else 'green'

        # Plot new route
        line = self.route_ax.plot(
            tour_coords[:, 0],
            tour_coords[:, 1],
            color=line_color,
            linestyle='-',
            marker='.',
            label=f"{algo_name} Best Tour"
        )
        self.route_lines[algo_name] = line

        # Update title
        self.route_ax.set_title(
            f"TSP Route - {algo_name} - Gen: {generation}, Cost: {best_cost:.2f}"
        )

        # Update legend
        handles, labels = self.route_ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles)) # Remove duplicates
        self.route_ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.pause(0.01)

    def update_convergence_plot(self, history, algo_name, color_val):
        """Update the convergence plot with cost history.

        Args:
            history: List of best costs per generation
            algo_name: Name of the algorithm
            color_val: Color for the plot line
        """
        generations_axis = list(range(len(history)))

        if algo_name in self.convergence_lines:
            self.convergence_lines[algo_name].set_data(generations_axis, history)
        else:
            line, = self.convergence_ax.plot(
                generations_axis,
                history,
                label=f"{algo_name} Best Cost",
                color=color_val
            )
            self.convergence_lines[algo_name] = line

        self.convergence_ax.relim()
        self.convergence_ax.autoscale_view()
        self.convergence_ax.legend(loc='upper right')
        plt.pause(0.01)

    def show_final_routes(self, sga_best_ind, hga_best_ind):
        """Display the final best routes for both algorithms.

        Args:
            sga_best_ind: Best Individual from SGA
            hga_best_ind: Best Individual from HGA-ACO
        """
        self.route_ax.cla() # Clear for final plot

        # Replot cities
        self.route_ax.scatter(
            self.cities[:, 0],
            self.cities[:, 1],
            c='red',
            marker='o',
            label='Cities',
            zorder=5
        )

        # Add city labels
        for i, city_coord in enumerate(self.cities):
            self.route_ax.text(
                city_coord[0] + 0.5,
                city_coord[1] + 0.5,
                str(i),
                fontsize=9
            )

        # Plot SGA final path
        sga_tour_coords = np.array([
            self.cities[i] for i in sga_best_ind.tour + [sga_best_ind.tour[0]]
        ])
        self.route_ax.plot(
            sga_tour_coords[:, 0],
            sga_tour_coords[:, 1],
            'b--',
            label=f"SGA Final: {sga_best_ind.cost:.2f}",
            linewidth=1.5
        )

        # Plot HGA-ACO final path
        hga_tour_coords = np.array([
            self.cities[i] for i in hga_best_ind.tour + [hga_best_ind.tour[0]]
        ])
        self.route_ax.plot(
            hga_tour_coords[:, 0],
            hga_tour_coords[:, 1],
            'g-',
            label=f"HGA-ACO Final: {hga_best_ind.cost:.2f}",
            linewidth=2
        )

        self.route_ax.set_title("Final Best Tours Comparison")
        self.route_ax.legend(loc='upper right')
        plt.pause(0.01)

    def keep_plot_open(self):
        """Keep the plot window open after execution."""
        plt.ioff()
        plt.show()
