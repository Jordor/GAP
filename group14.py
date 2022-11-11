import random
import Reporter
import numpy as np
import time

########################################################################################################################

TIME_LIMIT = 5 * 60  # seconds
K_TOURNAMENT = 16  # Number of candidates in the tournament
POP_SIZE = 400  # Population size
MUTATION_RATE = 0.10  # Percent mutation rate 1 = 100%
NUM_MUT = 20  # maximum number of mutations
MIN_MUT = 1  # minimum number of mutations. best is 1


########################################################################################################################

class Path:
    def __init__(self):
        self.cycle = []
        self.fitness = -1

    def getcycle(self):
        return self.cycle

    def setcycle(self, cycle):
        self.cycle = cycle
        self.fitness = -1

    def setfitness(self, fit):
        self.fitness = fit

    def getfitness(self):
        return self.fitness


########################################################################################################################

class CSVdata:
    def __init__(self):
        self.distances = np.ndarray([1, 2, 3, 4, 5])

    def load_distances(self, csv_file):
        file = open(csv_file)
        self.distances = np.loadtxt(file, delimiter=",")
        file.close()

    def getdistance(self, city_a, city_b) -> float:
        return self.distances[city_a][city_b]

    def numcities(self) -> int:
        return self.distances.shape[0]


########################################################################################################################

# Methods for the algorithm

def randomPath(path_size: int) -> Path:
    path = Path()
    c = np.random.permutation(path_size)
    path.setcycle(c)
    return path


def calculate_fitness(path: Path, csvdata: CSVdata):
    if path.getfitness() == -1:
        C = path.getcycle()
        L = len(C) - 1
        D = csvdata.getdistance(C[-1], C[0])
        if D > 100000:
            D = 100000
        for i in range(L):
            s2 = csvdata.getdistance(C[i], C[i + 1])
            if s2 > 100000:
                s2 = 100000
            D += s2
        path.setfitness(D)


def initialize_population(pop_size: int, path_size: int) -> np.ndarray:
    pop = []
    for i in range(pop_size):
        pop.append(randomPath(path_size))
    return np.array(pop)


def mutate_path(path: Path, num_mutations: int) -> Path:
    C = path.cycle.copy()
    L = len(C)
    R1 = np.array([], dtype=int)
    R2 = np.array([], dtype=int)
    N = num_mutations
    i = 0
    while i < N:
        ran = random.randint(0, L - 1)
        while (ran in R1) or (ran in R2):
            ran = random.randint(0, L - 1)
        R1 = np.append(R1, ran)
        i += 1
    i = 0
    while i < N:
        ran = random.randint(0, L - 1)
        while (ran in R1) or (ran in R2):
            ran = random.randint(0, L - 1)
        R2 = np.append(R2, ran)
        i += 1
    for i in range(N):
        c1 = C[int(R1[i])]
        c2 = C[int(R2[i])]
        C[int(R1[i])] = c2
        C[int(R2[i])] = c1
    m_path = Path()
    m_path.setcycle(C)
    return m_path


def mutate_population(pop: np.ndarray, mutation_rate: float):
    L = len(pop[0].cycle) - 1
    new_pop = []
    for p in pop:
        if random.random() < mutation_rate:
            new_pop.append(mutate_path(p, random.randint(MIN_MUT, NUM_MUT)))
        else:
            new_pop.append(p)
    return np.array(new_pop)

def ordered_crossover(parent1: Path, parent2: Path):
    child_p1 = []
    path_a = int(random.random() * len(parent1.cycle))
    path_b = int(random.random() * len(parent1.cycle))
    startGene = min(path_a, path_b)
    endGene = max(path_a, path_b)
    for i in range(startGene, endGene):
        child_p1.append(parent1.cycle[i])
    child_p2 = [item for item in parent2.cycle if item not in child_p1]
    child = Path()
    child.setcycle(np.array(child_p2 + child_p1))
    return child


def selection_k_tournament(population: np.ndarray, csvdata, K, popsize) -> np.ndarray:
    small_population = []

    for i in range(popsize):  # Do this until popsize is reached
        selected = random.choices(population, k=K)  # Randomly select K individuals
        for p in selected:
            calculate_fitness(p, csvdata)
        selected = sorted(selected, key=lambda agent: agent.fitness, reverse=False)
        small_population.append(selected[0])  # add the best to the returned population

    return np.array(small_population)  # Return the population


def recombination(population: np.ndarray, csv) -> np.ndarray:
    offspring = []
    for j in range(5):
        for i in range(POP_SIZE):
            p1 = population[i]
            p2 = population[random.randint(0, POP_SIZE - 1)]
            offspring.append(ordered_crossover(p1, p2))

    return np.array(offspring)


def elimination(intermediate_pop: np.ndarray, desired_size: int, csvdata) -> np.ndarray:
    for p in intermediate_pop:
        calculate_fitness(p, csvdata)
    sorted_pop = np.array(sorted(intermediate_pop, key=lambda agent: agent.fitness, reverse=False))
    s = sorted_pop[:int(desired_size / 2)]
    s2 = random.choices(sorted_pop[int(desired_size/2+1):], k=int(desired_size / 2 + 1))
    return np.append(s, s2)


class group14:
    def __init__(self):
        self.file = ''
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.CSV = CSVdata()

    def optimize(self, filename):
        # Read distance matrix from file
        self.file = filename
        self.CSV.load_distances(self.file)

        simtime = 0
        meanObjective = 0.0
        bestObjective = 0.0
        bestSolution = np.array([1, 2, 3, 4, 5])
        N_GENS = 0
        N_stuck = 0
        max_gens = 1000
        max_stuck = 200
        lastBest = 0.0

        n_city = self.CSV.numcities()
        # initialize the population
        population = initialize_population(POP_SIZE * 5, n_city)

        while (simtime <= TIME_LIMIT):
            start = time.time()

            # Selection: parents for cross-over
            parents_to_crossover = selection_k_tournament(population, self.CSV, K_TOURNAMENT, POP_SIZE)
            # Variation operators: crossover parents and discard previous population
            selected_pop = recombination(parents_to_crossover, self.CSV)
            # Variation operators: mutation
            mutated_pop = mutate_population(selected_pop, MUTATION_RATE)
            # Elimination operator
            population = elimination(mutated_pop, POP_SIZE, self.CSV)

            currentBest = population[0].fitness
            fits = 0.0
            if currentBest < bestObjective or bestObjective == 0.0:
                bestObjective = currentBest
                bestSolution = population[0].getcycle()
            for p in population:
                fits += p.getfitness()

            meanObjective = fits / POP_SIZE

            end = time.time()
            simtime += end - start

            timeLeft = self.reporter.report(
                meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                print('Reached time limit')
                break

            N_GENS += 1
            if N_GENS > max_gens:
                print('Reached max gens')
                break
            if bestObjective != lastBest:
                N_stuck = 0
                lastBest = bestObjective
            else:
                N_stuck += 1
            if N_stuck > max_stuck:
                print('Reached max stuck gens')
                break

        return 0


########################################################################################################################
# Launch the optimization

g = group14()
g.optimize("./data/tour50.csv")

########################################################################################################################
import random
import Reporter
import numpy as np
import time

########################################################################################################################

TIME_LIMIT = 5 * 60  # seconds
K_TOURNAMENT = 16  # Number of candidates in the tournament
POP_SIZE = 400  # Population size
MUTATION_RATE = 0.10  # Percent mutation rate 1 = 100%
NUM_MUT = 20  # maximum number of mutations
MIN_MUT = 1  # minimum number of mutations. best is 1


########################################################################################################################

class Path:
    def __init__(self):
        self.cycle = []
        self.fitness = -1

    def getcycle(self):
        return self.cycle

    def setcycle(self, cycle):
        self.cycle = cycle
        self.fitness = -1

    def setfitness(self, fit):
        self.fitness = fit

    def getfitness(self):
        return self.fitness


########################################################################################################################

class CSVdata:
    def __init__(self):
        self.distances = np.ndarray([1, 2, 3, 4, 5])

    def load_distances(self, csv_file):
        file = open(csv_file)
        self.distances = np.loadtxt(file, delimiter=",")
        file.close()

    def getdistance(self, city_a, city_b) -> float:
        return self.distances[city_a][city_b]

    def numcities(self) -> int:
        return self.distances.shape[0]


########################################################################################################################

# Methods for the algorithm

def randomPath(path_size: int) -> Path:
    path = Path()
    c = np.random.permutation(path_size)
    path.setcycle(c)
    return path


def calculate_fitness(path: Path, csvdata: CSVdata):
    if path.getfitness() == -1:
        C = path.getcycle()
        L = len(C) - 1
        D = csvdata.getdistance(C[-1], C[0])
        if (D > 100000):
            D = 100000
        for i in range(L):
            s2 = csvdata.getdistance(C[i], C[i + 1])
            if (s2 > 100000):
                s2 = 100000
            D += s2
        path.setfitness(D)


def initialize_population(pop_size: int, path_size: int) -> np.ndarray:
    pop = []
    for i in range(pop_size):
        pop.append(randomPath(path_size))
    return np.array(pop)


def mutate_path(path: Path, num_mutations: int) -> Path:
    C = path.cycle.copy()
    L = len(C)
    R1 = np.array([], dtype=int)
    R2 = np.array([], dtype=int)
    N = num_mutations
    i = 0
    while i < N:
        ran = random.randint(0, L - 1)
        while (ran in R1) or (ran in R2):
            ran = random.randint(0, L - 1)
        R1 = np.append(R1, ran)
        i += 1
    i = 0
    while i < N:
        ran = random.randint(0, L - 1)
        while (ran in R1) or (ran in R2):
            ran = random.randint(0, L - 1)
        R2 = np.append(R2, ran)
        i += 1
    for i in range(N):
        c1 = C[int(R1[i])]
        c2 = C[int(R2[i])]
        C[int(R1[i])] = c2
        C[int(R2[i])] = c1
    m_path = Path()
    m_path.setcycle(C)
    return m_path


def mutate_population(pop: np.ndarray, mutation_rate: float):
    L = len(pop[0].cycle) - 1
    new_pop = []
    for p in pop:
        if random.random() < mutation_rate:
            new_pop.append(mutate_path(p, random.randint(MIN_MUT, NUM_MUT)))
        else:
            new_pop.append(p)
    return np.array(new_pop)

def ordered_crossover(parent1: Path, parent2: Path):
    child_p1 = []
    path_a = int(random.random() * len(parent1.cycle))
    path_b = int(random.random() * len(parent1.cycle))
    startGene = min(path_a, path_b)
    endGene = max(path_a, path_b)
    for i in range(startGene, endGene):
        child_p1.append(parent1.cycle[i])
    child_p2 = [item for item in parent2.cycle if item not in child_p1]
    child = Path()
    child.setcycle(np.array(child_p2 + child_p1))
    return child


def selection_k_tournament(population: np.ndarray, csvdata, K, popsize) -> np.ndarray:
    small_population = []

    for i in range(popsize):  # Do this until popsize is reached
        selected = random.choices(population, k=K)  # Randomly select K individuals
        for p in selected:
            calculate_fitness(p, csvdata)
        selected = sorted(selected, key=lambda agent: agent.fitness, reverse=False)
        small_population.append(selected[0])  # add the best to the returned population

    return np.array(small_population)  # Return the population


def recombination(population: np.ndarray, csv) -> np.ndarray:
    offspring = []
    for j in range(5):
        for i in range(POP_SIZE):
            p1 = population[i]
            p2 = population[random.randint(0, POP_SIZE - 1)]
            offspring.append(ordered_crossover(p1, p2))

    return np.array(offspring)


def elimination(intermediate_pop: np.ndarray, desired_size: int, csvdata) -> np.ndarray:
    for p in intermediate_pop:
        calculate_fitness(p, csvdata)
    sorted_pop = np.array(sorted(intermediate_pop, key=lambda agent: agent.fitness, reverse=False))
    s = sorted_pop[:int(desired_size / 2)]
    s2 = random.choices(sorted_pop[int(desired_size/2+1):], k=int(desired_size / 2 + 1))
    return np.append(s, s2)


class group14:
    def __init__(self):
        self.file = ''
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.CSV = CSVdata()

    def optimize(self, filename):
        # Read distance matrix from file
        self.file = filename
        self.CSV.load_distances(self.file)

        simtime = 0
        meanObjective = 0.0
        bestObjective = 0.0
        bestSolution = np.array([1, 2, 3, 4, 5])
        N_GENS = 0
        N_stuck = 0
        max_gens = 1000
        max_stuck = 200
        lastBest = 0.0

        n_city = self.CSV.numcities()
        # initialize the population
        population = initialize_population(POP_SIZE * 5, n_city)

        while (simtime <= TIME_LIMIT):
            start = time.time()

            # Selection: parents for cross-over
            parents_to_crossover = selection_k_tournament(population, self.CSV, K_TOURNAMENT, POP_SIZE)
            # Variation operators: crossover parents and discard previous population
            selected_pop = recombination(parents_to_crossover, self.CSV)
            # Variation operators: mutation
            mutated_pop = mutate_population(selected_pop, MUTATION_RATE)
            # Elimination operator
            population = elimination(mutated_pop, POP_SIZE, self.CSV)

            currentBest = population[0].fitness
            fits = 0.0
            if currentBest < bestObjective or bestObjective == 0.0:
                bestObjective = currentBest
                bestSolution = population[0].getcycle()
            for p in population:
                fits += p.getfitness()

            meanObjective = fits / POP_SIZE

            end = time.time()
            simtime += end - start

            timeLeft = self.reporter.report(
                meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                print('Reached time limit')
                break

            N_GENS += 1
            if N_GENS > max_gens:
                print('Reached max gens')
                break
            if bestObjective != lastBest:
                N_stuck = 0
                lastBest = bestObjective
            else:
                N_stuck += 1
            if N_stuck > max_stuck:
                print('Reached max stuck gens')
                break

        return 0


########################################################################################################################
# Launch the optimization

g = group14()
g.optimize("./data/tour50.csv")

########################################################################################################################
