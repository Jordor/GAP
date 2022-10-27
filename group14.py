import Reporter
import numpy as np

#TBD
TIME_LIMIT = 100000
K_TOURNAMENT = 5
POP_SIZE = 20
MUTATION_RATE = 0.4

class Path:
    def __init__(self):
        self.cycle
        self.fitness = -1
    #example: cycle: [4 5 6 3 1 2 0]

def randomPath(path_size: int) -> Path:
    path = np.random.permutation(path_size)
    return path

def calculate_fitness(path: Path) -> float:
    # update fitness value
    pass

def mutate_path(path: Path, mutation_rate: float) -> Path:
    #
    pass

def crossover_parents(p1: Path, p2: Path) -> Path:
    pass

def mutate_population(pop: np.ndarray, mutation_rate: float):
    mutated_pop = []
    
    for i in range(0, len(pop)):
        mutatedInd = mutate_path(pop[i], mutation_rate)
        mutated_pop.append(mutatedInd)
    return np.array(mutated_pop)


def initialize_population(pop_size: int, path_size: int) -> np.ndarray:
    pop = []
    for i in range(pop_size):
        pop.append(randomPath(path_size))
    return np.array(pop)

def selection_k_tournament(initial_pop: np.ndarray, desired_size: int) -> np.ndarray:
    new_pop = np.array
    for x in range (desired_size):
        parents = random.choices(initial_pop, k= K_TOURNAMENT)
        parents = sorted(parents, key=lambda agent: agent.fitness, reverse=True)
        new_pop.append(parents[0])
    return np.array()

def select_2_parents(pop: np.ndarray):
    #return p1, p2
    pass


def variation(population: np.ndarray) -> None:
    mutated_population = mutate_population(population, MUTATION_RATE)

    offspring = []
    for i in range(POP_SIZE):
        #how to choose 2 parents?,for now it's random
        p1, p2 = select_2_parents(mutate_population)
        offspring.append(crossover_parents(p1, p2))
    return np.concatenate(mutated_population, offspring)

def eliminate(intermediate_pop: np.ndarray, desired_size: int) -> np.ndarray:
    pass


class group14:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
    # The evolutionary algorithm ’s main loop

    def optimize(self, filename):
        # Read distance matrix from file .
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        # Your code here .
        yourConvergenceTestsHere = True

        #initualize the population
        n_city = distanceMatrix.shape[0]
        population = initialize_population(n_city)

        while (yourConvergenceTestsHere):
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1, 2, 3, 4, 5])

            population = selection_k_tournament(population, POP_SIZE)
            intemediate_pop = variation(population)
            population = eliminate(intemediate_pop, POP_SIZE)

            # Call the reporter with :
            # - the mean objective function value of the population
            # - the best objective function value of the population
            # - a 1D numpy array in the cycle notation containing the best solution
            # with city numbering starting from 0
            timeLeft = self.reporter.report(
                meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
        # Your code here .
        return 0
