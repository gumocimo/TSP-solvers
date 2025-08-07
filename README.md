# TSP Genetic Algorithm

A simple implementation of a Genetic Algorithm (GA) for solving the Traveling Salesman Problem (TSP).

## Features
- Basic Standard Genetic Algorithm (SGA) implementation
- Fixed set of 10 cities for testing
- Tournament selection
- Ordered crossover operator
- Swap mutation operator
- Console output showing progress and results

## Project Structure
```
TSP-solvers/
├── main.py # Entry point with configuration
├── ga.py # GA components and algorithms
└── README.md
```

## Configuration
Edit the configuration parameters directly in `main.py`:
- `POP_SIZE`: Population size (default: 50)
- `GENERATIONS`: Number of generations (default: 100)
- `CROSSOVER_RATE`: Crossover probability (default: 0.85)
- `MUTATION_RATE`: Mutation probability (default: 0.15)
- `TOURNAMENT_K`: Tournament selection size (default: 3)
