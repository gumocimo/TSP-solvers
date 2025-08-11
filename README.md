# TSP Genetic Algorithm

An implementation of a Genetic Algorithm (GA) for solving the Traveling Salesman Problem (TSP) with real-time visualization capabilities.

## New Features in V2
- **Live plotting**: Real-time visualization of route evolution during GA execution
- **Dual plot display**: 
  - Best route visualization with cities and paths
  - Fitness convergence graph showing cost improvement over generations
- **Configurable update frequency**: Control plotting performance for different problem sizes
- **Final route display**: Clear visualization of the best solution found
- **Interactive matplotlib interface**: Pan, zoom, and examine results in detail

## Features
- Standard Genetic Algorithm (SGA) with elitism
- Random city generation in configurable grid
- Tournament selection operator
- Ordered crossover (OX) operator
- Swap mutation operator
- Real-time visualization of algorithm progress
- Adaptive parameter scaling based on problem size
- Execution time measurement

## Project Structure
```
tsp_ga/
├── main.py          # Entry point with configuration and adaptive parameters
├── ga.py            # GA components and algorithms with NumPy optimizations
├── visualization.py # Real-time plotting utilities for TSP visualization
├── requirements.txt
└── README.md
```

## Code Organization
- **main.py**: Contains the main execution logic and configuration
  - Problem settings (number of cities, grid dimensions, seed)
  - SGA default parameters
  - Visualization settings with adaptive update frequency
  - Adaptive parameter function for problem size scaling
  - Main function with timing, plotting orchestration, and results reporting
  
- **ga.py**: Consolidated GA components including:
  - City management (fixed and random generation with NumPy)
  - Distance calculations using NumPy arrays
  - Individual class with improved representation
  - Abstract TSP algorithm base class with cost history tracking
  - Standard GA with elitism support and visualization hooks
  
- **visualization.py**: Real-time plotting utilities including:
  - TSPPlotterSGA class for dual-plot visualization
  - Live route plot showing best tour evolution
  - Convergence plot showing fitness improvement
  - Interactive matplotlib interface
  - Final route and results display

## Configuration
Edit the configuration parameters directly in `main.py`:

### Problem Settings
- `NUM_CITIES`: Number of cities to generate (default: 50)
- `CITY_SEED`: Random seed for reproducible results (default: 1)
- `CITY_WIDTH`: Grid width for city generation (default: 100)
- `CITY_HEIGHT`: Grid height for city generation (default: 100)

### SGA Parameters
- `DEFAULT_SGA_POP_SIZE`: Population size (default: 100)
- `DEFAULT_SGA_GENERATIONS`: Number of generations (default: 1000)
- `DEFAULT_SGA_CROSSOVER_RATE`: Crossover probability (default: 0.85)
- `DEFAULT_SGA_MUTATION_RATE`: Mutation probability (default: 0.15)
- `DEFAULT_SGA_ELITISM_SIZE`: Number of elites to preserve (default: 5)
- `DEFAULT_SGA_TOURNAMENT_K`: Tournament selection size (default: 3)

### Visualization Settings
- `LIVE_PLOT_UPDATE_FREQ`: Update plot every N generations (default: 1, set to 0 to disable)

### Display Settings
- `VERBOSE`: Enable verbose output (default: True)
- `PROGRESS_FREQUENCY`: Print progress every N generations (default: 10)

## Adaptive Parameters
The algorithm automatically adjusts parameters based on problem size:
- **Small problems (≤50 cities)**: 
  - 750 generations, 100 population
  - Plot update every generation
- **Medium problems (≤100 cities)**: 
  - 1500 generations, 200 population, 10 elites
  - Plot update every 5 generations
- **Large problems (>100 cities)**: 
  - 5000 generations, 250 population, 15 elites
  - Plot update every 10 generations

## Dependencies
- numpy>=1.20.0
- matplotlib>=3.3.0
