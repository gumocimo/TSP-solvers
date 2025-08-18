"""Dynamic visualization components for TSP GA.

This module provides adaptive real-time plotting utilities that dynamically
adjust layout based on the number of enabled algorithms (2 or 3).

Author: gumocimo
Date: 18/08/2025
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class TSPPlotter:
    """Dynamic plotter for visualizing multiple algorithms on TSP."""

    def __init__(self, cities, enabled_algorithms):
        """Initialize the plotter with city data and enabled algorithms.

        Args:
            cities: NumPy array of city coordinates
            enabled_algorithms: Dict of algorithm names and their enabled status
        """
        self.cities = cities
        self.enabled_algorithms = enabled_algorithms
        self.num_algorithms = sum(enabled_algorithms.values())

        # Initialize tracking objects first
        self.convergence_lines = {}
        self.route_lines = {}
        self.pheromone_image = None
        self.pheromone_colorbar = None

        plt.ion() # Enable interactive mode

        # Create dynamic subplot layout
        self._create_dynamic_layout()

        # Setup plots based on enabled algorithms
        self._setup_plots()

    def _create_dynamic_layout(self):
        """Create subplot layout based on number of algorithms."""
        if self.num_algorithms == 2:
            # 2x2 layout for 2 algorithms
            self.fig, self.ax_array = plt.subplots(2, 2, figsize=(10, 8))
            self.fig.subplots_adjust(hspace=0.3, wspace=0.25)
        elif self.num_algorithms == 3:
            # 2x3 layout for 3 algorithms
            self.fig, self.ax_array = plt.subplots(2, 3, figsize=(14, 8))
            self.fig.subplots_adjust(hspace=0.3, wspace=0.2)
        else:
            raise ValueError(f"Unsupported number of algorithms: {self.num_algorithms}")

    def _setup_plots(self):
        """Setup plots based on enabled algorithms."""
        # Assign axes to algorithms
        self.route_axes = {}
        self.convergence_ax = None
        self.special_axes = {} # For pheromone or other special plots

        if self.num_algorithms == 2:
            # 2x2 layout
            ax_idx = 0
            for algo, enabled in self.enabled_algorithms.items():
                if enabled:
                    self.route_axes[algo] = self.ax_array.flat[ax_idx]
                    ax_idx += 1
            self.convergence_ax = self.ax_array.flat[2]

            # Special plot (pheromone for HGA-ACO or performance comparison)
            if self.enabled_algorithms.get("HGA-ACO", False):
                self.special_axes["pheromone"] = self.ax_array.flat[3]
            else:
                # Use for performance comparison
                self.special_axes["performance"] = self.ax_array.flat[3]

        elif self.num_algorithms == 3:
            # 2x3 layout - top row for routes, bottom for convergence and special
            route_idx = 0
            for algo, enabled in self.enabled_algorithms.items():
                if enabled:
                    self.route_axes[algo] = self.ax_array[0, route_idx]
                    route_idx += 1

            self.convergence_ax = self.ax_array[1, 0]

            # Special plots
            if self.enabled_algorithms.get("HGA-ACO", False):
                self.special_axes["pheromone"] = self.ax_array[1, 1]

            # Performance comparison in bottom right
            self.special_axes["performance"] = self.ax_array[1, 2]

        # Setup individual plots
        self._setup_route_plots()
        self._setup_convergence_plot()
        if "pheromone" in self.special_axes:
            self._setup_pheromone_plot()
        if "performance" in self.special_axes:
            self._setup_performance_plot()

    def _setup_route_plots(self):
        """Setup route visualization plots for enabled algorithms."""
        for algo, ax in self.route_axes.items():
            ax.set_title(f"{algo} Best Route Evolution")
            ax.set_xlabel("X-coordinate")
            ax.set_ylabel("Y-coordinate")

            # Plot cities
            ax.scatter(
                self.cities[:, 0],
                self.cities[:, 1],
                c='red',
                marker='o',
                label='Cities',
                zorder=5
            )

            # Add city labels
            for i, city_coord in enumerate(self.cities):
                ax.text(
                    city_coord[0] + 0.5,
                    city_coord[1] + 0.5,
                    str(i),
                    fontsize=9
                )

            ax.legend(loc='upper right')
            self.route_lines[algo] = None

    def _setup_convergence_plot(self):
        """Setup the convergence visualization plot."""
        self.convergence_ax.set_title("Fitness Convergence Comparison")
        self.convergence_ax.set_xlabel("Generation")
        self.convergence_ax.set_ylabel("Best Cost (Distance)")

    def _setup_pheromone_plot(self):
        """Setup the pheromone heatmap plot."""
        ax = self.special_axes["pheromone"]
        ax.set_title("HGA-ACO Pheromone Matrix")
        ax.set_xlabel("City Index")
        ax.set_ylabel("City Index")

    def _setup_performance_plot(self):
        """Setup the performance comparison plot."""
        ax = self.special_axes["performance"]
        ax.set_title("Performance Comparison")
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Values")

    def get_algorithm_color(self, algo_name):
        """Get color for algorithm visualization."""
        colors = {"SGA": "blue", "HGA-ACO": "green", "PSO": "orange"}
        return colors.get(algo_name, "black")

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
        if algo_name not in self.route_axes:
            return

        target_ax = self.route_axes[algo_name]

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

        # Remove old route line
        if self.route_lines.get(algo_name):
            try:
                self.route_lines[algo_name].pop(0).remove()
            except (AttributeError, IndexError, ValueError):
                self.route_lines[algo_name] = None

        # Create tour coordinates
        tour_coords = np.array([
            self.cities[i] for i in best_tour_indices + [best_tour_indices[0]]
        ])

        # Get algorithm color
        line_color = self.get_algorithm_color(algo_name)

        # Plot new route
        line = target_ax.plot(
            tour_coords[:, 0],
            tour_coords[:, 1],
            color=line_color,
            linestyle='-',
            marker='.',
            label="Current Best"
        )
        self.route_lines[algo_name] = line

        # Update title
        gen_display = "Final" if generation == -1 else str(generation)
        target_ax.set_title(
            f"{algo_name} Route - Gen: {gen_display}, Cost: {best_cost:.2f}"
        )

        # Update legend
        handles, labels = target_ax.get_legend_handles_labels()
        by_label = {"Cities": handles[labels.index("Cities")]}
        if self.route_lines[algo_name]:
            by_label["Current Best"] = self.route_lines[algo_name][0]
        target_ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.pause(0.01)

    def update_convergence_plot(self, history, algo_name, color_val=None):
        """Update the convergence plot with cost history."""
        if color_val is None:
            color_val = self.get_algorithm_color(algo_name)

        generations_axis = list(range(len(history)))
        label_prefix = f"{algo_name} Best Cost"
        current_label = label_prefix

        # Check if time was already added
        existing_legend = self.convergence_ax.get_legend()
        if existing_legend:
            for text in existing_legend.get_texts():
                if text.get_text().startswith(label_prefix) and "(Time:" in text.get_text():
                    current_label = text.get_text()
                    break

        if algo_name in self.convergence_lines:
            self.convergence_lines[algo_name].set_data(generations_axis, history)
            self.convergence_lines[algo_name].set_label(current_label)
        else:
            line, = self.convergence_ax.plot(
                generations_axis,
                history,
                label=current_label,
                color=color_val
            )
            self.convergence_lines[algo_name] = line

        self.convergence_ax.relim()
        self.convergence_ax.autoscale_view()
        self.convergence_ax.legend(loc='upper right')
        plt.pause(0.01)

    def update_pheromone_heatmap(self, pheromone_matrix, generation, update_freq):
        """Update the pheromone heatmap visualization."""
        if "pheromone" not in self.special_axes:
            return

        ax = self.special_axes["pheromone"]

        # Check if update is needed
        if not (update_freq > 0 and generation % update_freq == 0):
            if generation != -1:
                return

        if self.pheromone_image is None:
            # Create initial heatmap
            self.pheromone_image = ax.imshow(
                pheromone_matrix,
                cmap='viridis',
                aspect='auto',
                interpolation='nearest'
            )
            self.pheromone_colorbar = self.fig.colorbar(
                self.pheromone_image,
                ax=ax,
                orientation='vertical'
            )
        else:
            # Update existing heatmap
            self.pheromone_image.set_data(pheromone_matrix)
            self.pheromone_image.set_clim(
                vmin=np.min(pheromone_matrix),
                vmax=np.max(pheromone_matrix)
            )

        gen_display = "Final" if generation == -1 else str(generation)
        ax.set_title(f"HGA Pheromones - Gen: {gen_display}")
        plt.pause(0.01)

    def display_execution_times(self, exec_times):
        """Update convergence plot with execution times.

        Args:
            exec_times: Dict of algorithm names to execution times
        """
        for algo, time in exec_times.items():
            if algo in self.convergence_lines:
                label = f"{algo} Best Cost (Time: {time:.2f}s)"
                self.convergence_lines[algo].set_label(label)

        self.convergence_ax.legend(loc='upper right')
        plt.pause(0.01)

    def show_final_routes(self, best_individuals):
        """Display the final best routes for all algorithms.

        Args:
            best_individuals: Dict of algorithm names to best individuals
        """
        for algo, best_ind in best_individuals.items():
            if algo not in self.route_axes:
                continue

            ax = self.route_axes[algo]
            ax.cla()
            ax.set_title(f"{algo} Final Route - Cost: {best_ind.cost:.2f}")

            # Replot cities
            ax.scatter(
                self.cities[:, 0],
                self.cities[:, 1],
                c='red',
                marker='o',
                label='Cities',
                zorder=5
            )

            # Add city labels
            for i, city_coord in enumerate(self.cities):
                ax.text(
                    city_coord[0] + 0.5,
                    city_coord[1] + 0.5,
                    str(i),
                    fontsize=9
                )

            # Plot final tour
            tour_coords = np.array([
                self.cities[i] for i in best_ind.tour + [best_ind.tour[0]]
            ])
            color = self.get_algorithm_color(algo)
            ax.plot(
                tour_coords[:, 0],
                tour_coords[:, 1],
                color=color,
                linestyle='-',
                label=f"{algo} Final Path"
            )
            ax.legend(loc='upper right')

        plt.pause(0.1)

    def show_performance_comparison(self, results, exec_times):
        """Display performance comparison metrics.

        Args:
            results: Dict of algorithm names to best individuals
            exec_times: Dict of algorithm names to execution times
        """
        if "performance" not in self.special_axes:
            return

        ax = self.special_axes["performance"]
        ax.cla()

        # Prepare data
        algorithms = list(results.keys())
        costs = [results[algo].cost for algo in algorithms]
        times = [exec_times[algo] for algo in algorithms]

        # Normalize costs and times for comparison
        min_cost = min(costs)
        normalized_costs = [min_cost / cost * 100 for cost in costs] # Higher is better

        max_time = max(times) if max(times) > 0 else 1
        efficiency = [(max_time - t) / max_time * 100 for t in times] # Higher is better

        # Create grouped bar chart
        x = np.arange(len(algorithms))
        width = 0.35

        bars1 = ax.bar(x - width / 2, normalized_costs, width, label='Solution Quality %',
                       color=[self.get_algorithm_color(algo) for algo in algorithms], alpha=0.8)
        bars2 = ax.bar(x + width / 2, efficiency, width, label='Time Efficiency %',
                       color=[self.get_algorithm_color(algo) for algo in algorithms], alpha=0.5)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # Add actual values as text below
        for i, algo in enumerate(algorithms):
            ax.text(i, -15, f'Cost: {costs[i]:.1f}', ha='center', fontsize=8, rotation=0)
            ax.text(i, -25, f'Time: {times[i]:.1f}s', ha='center', fontsize=8, rotation=0)

        ax.set_title('Performance Comparison (Higher is Better)')
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Performance Score (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.legend()
        ax.set_ylim(-30, 110)
        ax.grid(axis='y', alpha=0.3)

        plt.pause(0.01)

    def keep_plot_open(self):
        """Keep the plot window open after execution."""
        active_algos = [algo for algo, enabled in self.enabled_algorithms.items() if enabled]
        title = "TSP Solver Comparison: " + " vs ".join(active_algos)
        self.fig.suptitle(title, fontsize=16, y=0.99)
        plt.ioff()
        plt.show()
