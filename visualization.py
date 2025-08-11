"""
Visualization components for TSP GA.
This module provides real-time plotting utilities for visualizing the GA's
progress in solving TSP, including route evolution and fitness convergence.

Author: gumocimo
Date: 11/08/2025
"""

import matplotlib.pyplot as plt
import numpy as np


class TSPPlotterSGA:
    """Plotter for visualizing SGA progress on TSP."""

    def __init__(self, cities):
        """Initialize the plotter with city data.

        Args:
            cities: NumPy array of city coordinates
        """
        self.cities = cities
        plt.ion() # Enable interactive mode

        # Create subplots
        self.fig, self.ax_array = plt.subplots(1, 2, figsize=(12, 5))
        self.route_ax = self.ax_array[0]
        self.convergence_ax = self.ax_array[1]

        self.fig.subplots_adjust(wspace=0.25)

        # Setup route plot
        self._setup_route_plot()

        # Setup convergence plot
        self._setup_convergence_plot()

        # Initialize line objects
        self.sga_route_line = None
        self.sga_convergence_line = None

    def _setup_route_plot(self):
        """Setup the route visualization plot."""
        self.route_ax.set_title("SGA Best Route Evolution")
        self.route_ax.set_xlabel("X-coordinate")
        self.route_ax.set_ylabel("Y-coordinate")

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

        self.route_ax.legend(loc='upper right')

    def _setup_convergence_plot(self):
        """Setup the convergence visualization plot."""
        self.convergence_ax.set_title("SGA Fitness Convergence")
        self.convergence_ax.set_xlabel("Generation")
        self.convergence_ax.set_ylabel("Best Cost (Distance)")

    def update_live_route_plot(self, best_tour_indices, generation, best_cost, update_freq):
        """Update the route plot with current best tour.

        Args:
            best_tour_indices: List of city indices in best tour
            generation: Current generation number (-1 for final)
            best_cost: Cost of the best tour
            update_freq: Update frequency setting
        """
        # Check if update is needed
        is_update_time = (
                update_freq > 0 and (
                generation % update_freq == 0 or
                generation == -1 or
                generation == 0
        )
        )

        if not is_update_time and generation > 0:
            return

        # Remove old route line if exists
        if self.sga_route_line:
            try:
                self.sga_route_line.pop(0).remove()
            except (AttributeError, IndexError, ValueError):
                self.sga_route_line = None

        # Create tour coordinates (close the loop)
        tour_coords = np.array([
            self.cities[i] for i in best_tour_indices + [best_tour_indices[0]]
        ])

        # Plot new route
        line = self.route_ax.plot(
            tour_coords[:, 0],
            tour_coords[:, 1],
            color='blue',
            linestyle='-',
            marker='.',
            label="Current Best"
        )
        self.sga_route_line = line

        # Update title
        gen_display = "Final" if generation == -1 else str(generation)
        self.route_ax.set_title(
            f"SGA Route - Gen: {gen_display}, Cost: {best_cost:.2f}"
        )

        # Update legend
        handles, labels = self.route_ax.get_legend_handles_labels()
        by_label = {"Cities": handles[labels.index("Cities")]}
        if self.sga_route_line:
            by_label["Current Best"] = self.sga_route_line[0]
        self.route_ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.pause(0.01)

    def update_convergence_plot(self, history, exec_time=None):
        """Update the convergence plot with cost history.

        Args:
            history: List of best costs per generation
            exec_time: Total execution time (optional)
        """
        generations_axis = list(range(len(history)))
        label = "SGA Best Cost"

        if exec_time is not None:
            label += f" (Time: {exec_time:.2f}s)"

        if self.sga_convergence_line:
            self.sga_convergence_line.set_data(generations_axis, history)
            self.sga_convergence_line.set_label(label)
        else:
            self.sga_convergence_line, = self.convergence_ax.plot(
                generations_axis,
                history,
                label=label,
                color="blue"
            )

        self.convergence_ax.relim()
        self.convergence_ax.autoscale_view()
        self.convergence_ax.legend(loc='upper right')
        plt.pause(0.01)

    def display_execution_time(self, sga_time, sga_history):
        """Update convergence plot with execution time.

        Args:
            sga_time: Total execution time
            sga_history: Cost history list
        """
        self.update_convergence_plot(sga_history, exec_time=sga_time)

    def show_final_route(self, sga_best_ind):
        """Display the final best route.

        Args:
            sga_best_ind: Best Individual found
        """
        self.route_ax.cla()
        self.route_ax.set_title(f"SGA Final Route - Cost: {sga_best_ind.cost:.2f}")

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

        # Plot final tour
        tour_coords = np.array([
            self.cities[i] for i in sga_best_ind.tour + [sga_best_ind.tour[0]]
        ])
        self.route_ax.plot(
            tour_coords[:, 0],
            tour_coords[:, 1],
            'b-',
            label=f"SGA Final Path"
        )

        self.route_ax.legend(loc='upper right')
        plt.pause(0.1)

    def keep_plot_open(self):
        """Keep the plot window open after execution."""
        self.fig.suptitle(
            "TSP Solver: Standard Genetic Algorithm (SGA)",
            fontsize=16,
            y=0.99
        )
        plt.ioff() # Turn off interactive mode
        plt.show() # Block until window is closed
