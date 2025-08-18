# TSP Genetic Algorithm

Multi-algorithm comparison framework for solving the Traveling Salesman Problem (TSP) featuring Standard Genetic Algorithm (SGA), Hybrid GA-ACO (HGA-ACO), and Particle Swarm Optimization (PSO).

## New Features in V4
- **Particle Swarm Optimization (PSO)**: Swarm intelligence algorithm with 2-opt local search
- **Algorithm toggle system**: Select any combination of algorithms to compare
- **Dynamic visualization**: Plotter adapts layout based on enabled algorithms (2x2 for 2 algorithms, 2x3 for 3)
- **Flexible comparisons**: Compare any 2 or all 3 algorithms simultaneously
- **Enhanced configuration**: Size-based parameter presets with custom override option
- **Performance comparison chart**: Visual performance metrics for all algorithms

## Available Algorithm Combinations
1. **SGA vs HGA-ACO** (classic comparison)
2. **SGA vs PSO** (GA vs swarm intelligence)
3. **HGA-ACO vs PSO** (hybrid vs pure swarm)
4. **All three** (comprehensive comparison)

### Standard GA (SGA)
- Tournament selection
- Ordered crossover
- Swap mutation
- Elitism preservation

### Hybrid GA-ACO (HGA-ACO)
- Combines GA with Ant Colony Optimization
- Pheromone-guided construction
- Mixed population approach
- Pheromone matrix visualization

### Particle Swarm Optimization (PSO)
- Swarm of particles exploring solution space
- Personal best (pBest) and global best (gBest) tracking
- Velocity-based position updates adapted for discrete TSP
- Social and cognitive learning components
- Optional 2-opt local search improvement

## Project Structure
```
tsp_ga/
├── main.py              # Entry point with integrated configuration
├── core.py              # Core components (cities, distance, individual)
├── visualization.py     # Dynamic plotting with adaptive layout
├── algorithms/          # Algorithm implementations module
│   ├── __init__.py
│   ├── base.py         # Abstract base class for TSP algorithms
│   ├── sga.py          # Standard Genetic Algorithm
│   ├── hga_aco.py      # Hybrid GA-ACO Algorithm
│   └── pso.py          # Particle Swarm Optimization
├── requirements.txt
└── README.md
```


## Configuration
Edit the configuration directly in `main.py`:

### Algorithm Selection
```python
ENABLE_SGA = True      # Standard Genetic Algorithm
ENABLE_HGA_ACO = True  # Hybrid GA-ACO
ENABLE_PSO = True      # Particle Swarm Optimization
```

### Problem Settings
- `NUM_CITIES`: Number of cities (default: 50)
- `CITY_SEED`: Random seed for reproducibility (default: 1)
- `CITY_WIDTH`: Grid width (default: 100)
- `CITY_HEIGHT`: Grid height (default: 100)

### Automatic Parameter Selection
The system automatically selects optimal parameters based on problem size

### Custom Parameters
Override automatic presets by defining `CUSTOM_PARAMS`:
```python
CUSTOM_PARAMS = {
    "SGA_POP_SIZE": 75,
    "SGA_GENERATIONS": 250,
    "SGA_CROSSOVER_RATE": 0.9,
    # ... add other parameters as needed
}
```

## Parameter Quick Reference

| Parameter | Small (<50) | Medium (50-100) | Large (>100) |
|-----------|------------|-----------------|--------------|
| **SGA** |
| 1. Population | 100 | 100 | 200 |
| 2. Generations | 750 | 1500 | 5000 |
| **HGA-ACO** |
| 1. Population | 50 | 100 | 200 |
| 2. Generations | 250 | 500 | 1000 |
| 3. ACO Rate | 0.5 | 0.5 | 0.6 |
| **PSO** |
| 1. Particles | 15 | 25 | 30 |
| 2. Generations | 250 | 500 | 1000 |
| 3. Inertia (w) | 0.4 | 0.5 | 0.6 |
| **Visualization** |
| Update Freq | 1 | 5 | 10 |

## Visualization Features

### Dynamic Layout
- **2x2 Layout** (2 algorithms):
  - Top row: Algorithm route plots
  - Bottom left: Convergence comparison
  - Bottom right: Pheromone heatmap (if HGA-ACO) or Performance comparison

- **2x3 Layout** (3 algorithms):
  - Top row: All algorithm route plots
  - Bottom left: Convergence comparison
  - Bottom middle: Pheromone heatmap (if HGA-ACO enabled)
  - Bottom right: Performance comparison

### Performance Metrics
- Solution Quality Score (normalized)
- Time Efficiency Score
- Actual cost and execution time display
- Color-coded bar charts for easy comparison
