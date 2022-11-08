import random
import Reporter
import time
import numpy as np

# TBD
TIME_LIMIT = 5 * 60  # in seconds? otherwise calculate accordingly
K_TOURNAMENT = 5  # Number of candidates in the tournament
POP_SIZE = 200  # Population size
MUTATION_RATE = 0.05  # percent mutation rate try, 5


class Path:
    def __init__(self):
        self.cycle = []
        self.fitness = None

    def getcycle(self):
        return self.cycle

    def setcycle(self, cycle):
        self.cycle = cycle
        self.fitness = -1

    def setfitness(self, fit):
        self.fitness = fit

    def getfitness(self):
        return self.fitness

    # example: cycle: [4 5 6 3 1 2 0]


class CSVdata:

    def __init__(self):
        """
        1- open csv file
        2- import the distance matrix as distances[][]
        3- implement a method for retrieving the distances according to city index
        """
        self.distances = np.ndarray([1, 2, 3, 4, 5])

    def load_distances(self, csv_file):
        file = open(csv_file)
        self.distances = np.loadtxt(file, delimiter=",")
        file.close()

    def getdistance(self, city_a, city_b) -> float:
        return self.distances[city_a][city_b]

    def numcities(self) -> int:
        return self.distances.shape[0]


def randomPath(path_size: int) -> Path:
    """
    initializes the cycle of a new path which it returns

    :param path_size: size of the permutation of numbers from 1 to pathsize
    :return: a new Path object containing a random cycle of cities
    """

    path = Path()
    c = np.random.permutation(path_size)
    path.setcycle(c)
    return path


def calculate_fitness(path: Path, csvdata: CSVdata):
    """
    :param path: Path object containing a particular city cycle
    :param csvdata: CVSdata object containing the information of the city distances
    :return: a float corresponding to the total distance travelled according to the Path's particular cycle

    1- get distance between two adjacent cities
    2- add the value to D
    3- repeat for all following cities
    4- do not forget the distance between the first and the last in the cycle
    """

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


def mutate_path(path: Path, num_mutations: int) -> Path:
    """
    :param path: path to be mutated
    :param num_mutations: number of mutations to apply
    :return: mutated version of the path

    1- generate two random numbers in the range of 0:len(cycle)
    1.5 - repeat if they are the same
    2- swap the cities in that path that correspond to those indices
    3- repeat for num_mutations
    """

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


def crossover_parents(p1: Path, p2: Path) -> Path:
    """
    :param p1: Path object of the first parent
    :param p2: Path object of the second parent
    :return: Offspring (one)

    We should add a new parameter defining how big the crossover is in number of cities

    1- obtain cycles from p1 and p2
    2- select a random crossover index
    3- pick a number of cities from that point to the right from p1
    4- remove those cities from p2
    5- insert p1 cities in the same order and the same position but in p2
    """

    c1 = p1.getcycle()
    c2 = p2.getcycle()
    L = len(c1)
    nn = 10
    r = random.randint(0, L - nn - 1)
    nc = c2[:]  # copy c2 for new cycle
    cross = c1[r:r + nn]

    for i in cross:
        # remove from nc
        index = np.where(nc == i)
        nc = np.delete(nc, index)

    nc = np.append(nc, cross)
    p = Path()
    p.setcycle(nc)

    return p


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
    child.setcycle(np.array(child_p1 + child_p2))
    return child


def mutate_population(pop: np.ndarray, mutation_rate: float):
    """
    :param pop: current population
    :param mutation_rate: integer % of pop to mutate
    :param num_mutations: number of mutations to apply
    :return: population after mutation

    1- copy population
    2- calculate number of mutated paths to make with len(pop)*mutation_rate/100
    3- generate that many number of randoms 0:len(pop)
    4- generate repeated randoms again
    5- apply mutation to those indexes to copied population
    6- return mutated population
    """
    L = len(pop[0].cycle) -1

    new_pop = []
    for p in pop:
        if random.random() < mutation_rate:
            # i = random.randint(0, L)
            # j = random.randint(0, L)
            # if i != j:
            #    p.cycle[i], p.cycle[j] = p.cycle[j], p.cycle[i]
            new_pop.append(mutate_path(p, random.randint(1, 6)))
        else:
            new_pop.append(p)
    return np.array(new_pop)


def initialize_population(pop_size: int, path_size: int) -> np.ndarray:
    pop = []
    for i in range(pop_size):
        pop.append(randomPath(path_size))
    return np.array(pop)


def selection_k_tournament(initial_pop: np.ndarray, csvdata) -> Path:
    parents = random.choices(initial_pop, k=K_TOURNAMENT)

    for p in parents:
        calculate_fitness(p, csvdata)

    parents = sorted(parents, key=lambda agent: agent.fitness, reverse=False)
    return parents[0]


def select_2_parents(pop: np.ndarray, csv):
    #parents = random.choices(pop, k=2)  # completely random for now # we can change it later
    return selection_k_tournament(pop, csv), selection_k_tournament(pop, csv)


def variation(population: np.ndarray, csv) -> np.ndarray:
    #mutated_population = mutate_population(population, MUTATION_RATE)

    offspring = []
    for i in range(3 * POP_SIZE):
        p1, p2 = select_2_parents(population, csv)
        offspring.append(ordered_crossover(p1, p2))

    # out = np.append(population, offspring)

    # return out
    return offspring


def eliminate(intermediate_pop: np.ndarray, desired_size: int, csvdata) -> np.ndarray:

    for p in intermediate_pop:
        calculate_fitness(p, csvdata)
    pop = []
    for i in range(0, desired_size):
        pop.append(selection_k_tournament(intermediate_pop, csvdata))

    return np.array(sorted(pop, key=lambda agent: agent.fitness, reverse=False))


class group14:
    def __init__(self):
        self.file = ''
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.CSV = CSVdata()  # this object can return distances

    # The evolutionary algorithm ’s main loop

    def optimize(self, filename):
        # Read distance matrix from file. Jordi: I used a separate object for that
        self.file = filename
        self.CSV.load_distances(self.file)

        simtime = 0

        # initialize the population
        n_city = self.CSV.numcities()

        population = initialize_population(POP_SIZE, n_city)
        for p in population:
            calculate_fitness(p, self.CSV)

        meanObjective = 0.0
        bestObjective = 0.0
        bestSolution = np.array([1, 2, 3, 4, 5])


        while (simtime < TIME_LIMIT):

            start = time.time()

            #population = selection_k_tournament(population, POP_SIZE, self.CSV)
            intermediate_pop = variation(population, self.CSV)
            intermediate_pop = mutate_population(intermediate_pop, MUTATION_RATE)
            population = eliminate(intermediate_pop, POP_SIZE, self.CSV)

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

            print('\n'*2)
            print('Current best is = ', currentBest)
            print('Global best is = ', bestObjective)
            print('mean objective is', meanObjective)
            print('Best Solution ', bestSolution)

            # Call the reporter with :
            # - the mean objective function value of the population
            # - the best objective function value of the population
            # - a 1D numpy array in the cycle notation containing the best solution
            # with city numbering starting from 0
            timeLeft = self.reporter.report(
                meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break


        return 0

g = group14()

g.optimize("./data/tour50.csv")
