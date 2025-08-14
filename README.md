# TSP Genetic Algorithm

Solving the Traveling Salesman Problem (TSP) by comparing a Standard Genetic Algorithm (SGA) with a Hybrid Genetic Algorithm - Ant Colony Optimization (HGA-ACO).

## New Features in V3
- **Hybrid GA-ACO Algorithm**: Combines genetic operators with ant colony optimization
- **Pheromone-based guidance**: ACO pheromone trails guide solution construction
- **Algorithm comparison**: Side-by-side comparison of SGA vs HGA-ACO
- **Dual algorithm visualization**: Separate route plots for each algorithm
- **Performance metrics**: Detailed comparison of solution quality and execution time
- **Modular algorithm structure**: Algorithms organized in a separate module

## Features
- Two algorithm implementations:
  - **SGA**: Standard Genetic Algorithm with elitism
  - **HGA-ACO**: Hybrid approach combining GA and ACO strategies
- Real-time visualization of both algorithms
- Comprehensive performance comparison
- Configurable parameters for both algorithms
- Reproducible results with seeded random generation
- Adaptive parameter scaling based on problem size

## Project Structure
```
TSP-solvers/
├── main.py           # Entry point with configuration and comparison logic
├── core.py           # Core components (cities, distance, individual)
├── visualization.py  # Plotting utilities for algorithm comparison
├── algorithms/       # Algorithm implementations module
│   ├── __init__.py
│   ├── base.py       # Abstract base class for TSP algorithms
│   ├── sga.py        # Standard Genetic Algorithm
│   └── hga_aco.py    # Hybrid GA-ACO Algorithm
├── requirements.txt
└── README.md
```

## Code Organization
- **main.py**: Contains main execution logic and configuration
  - Problem settings (cities, grid dimensions, seed)
  - Algorithm parameters for both SGA and HGA-ACO
  - Adaptive parameter scaling based on problem size
  - Algorithm execution and comparison logic
  - Results reporting and performance metrics

- **core.py**: Consolidated core components
  - City management (fixed and random generation)
  - Distance calculations using NumPy
  - Individual class for solution representation

- **visualization.py**: Plotting utilities
  - TSPPlotter class for dual-algorithm visualization
  - Live route plotting for both algorithms
  - Convergence comparison graphs
  - Final route comparison display

- **algorithms/** module:
  - **base.py**: Abstract TSPAlgorithm base class
  - **sga.py**: Standard GA with tournament selection, ordered crossover, and swap mutation
  - **hga_aco.py**: Hybrid algorithm combining:
    - GA operators (crossover, mutation, selection)
    - ACO pheromone-based tour construction
    - Pheromone update mechanisms

   
## Configuration
Edit the configuration parameters directly in `main.py`:

### Problem Settings
- `NUM_CITIES`: Number of cities to generate (default: 50)
- `CITY_SEED`: Random seed for reproducibility (default: 1)
- `CITY_WIDTH`: Grid width for city generation (default: 100)
- `CITY_HEIGHT`: Grid height for city generation (default: 100)

### SGA Parameters
- `DEFAULT_SGA_POP_SIZE`: Population size (default: 100)
- `DEFAULT_SGA_GENERATIONS`: Number of generations (default: 1000)
- `DEFAULT_SGA_CROSSOVER_RATE`: Crossover probability (default: 0.85)
- `DEFAULT_SGA_MUTATION_RATE`: Mutation probability (default: 0.15)
- `DEFAULT_SGA_ELITISM_SIZE`: Number of elites (default: 5)
- `DEFAULT_SGA_TOURNAMENT_K`: Tournament size (default: 3)

### HGA-ACO Parameters
- `DEFAULT_HGA_POP_SIZE`: Population size (default: 100)
- `DEFAULT_HGA_GENERATIONS`: Number of generations (default: 250)
- `DEFAULT_HGA_GA_CROSSOVER_RATE`: GA crossover rate (default: 0.7)
- `DEFAULT_HGA_ACO_CONTRIBUTION_RATE`: ACO individual proportion (default: 0.5)
- `DEFAULT_HGA_MUTATION_RATE`: Mutation probability (default: 0.15)
- `DEFAULT_HGA_ELITISM_SIZE`: Number of elites (default: 5)
- `DEFAULT_HGA_TOURNAMENT_K`: Tournament size (default: 3)

### ACO-specific Parameters
- `DEFAULT_HGA_ALPHA`: Pheromone influence (default: 1.0)
- `DEFAULT_HGA_BETA`: Heuristic influence (default: 3.0)
- `DEFAULT_HGA_EVAPORATION_RATE`: Pheromone evaporation (default: 0.3)
- `DEFAULT_HGA_Q_PHEROMONE`: Pheromone deposit constant (default: 100.0)
- `DEFAULT_HGA_INITIAL_PHEROMONE`: Initial pheromone level (default: 0.1)
- `DEFAULT_HGA_BEST_N_DEPOSIT`: Number of best ants depositing (default: 5)

### Visualization Settings
- `LIVE_PLOT_UPDATE_FREQ`: Update frequency for live plotting (default: 1)
