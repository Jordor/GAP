import random

import Reporter
import numpy as np

# TBD
TIME_LIMIT = 5 * 60  # in seconds?
K_TOURNAMENT = 5  # Number of candidates in the tournament
POP_SIZE = 100  # Population size
MUTATION_RATE = 5  # percent mutation rate
NUM_MUTATIONS = 5


class Path:
    def __init__(self):
        self.cycle
        self.fitness = -1

    def getcycle(self):
        return self.cycle

    def setcycle(self, cycle):
        self.cycle = cycle

    # example: cycle: [4 5 6 3 1 2 0]


class CSVdata:

    def __init__(self):
        """
        1- open csv file
        2- import the distance matrix as distances[][]
        3- implement a method for retrieving the distances according to city index
        """
        self.distances = []

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
    C = np.random.permutation(path_size)
    path.setcycle(C)
    return path


def calculate_fitness(path: Path, csvdata: CSVdata) -> float:
    """
    :param path: Path object containing a particular city cycle
    :param csvdata: CVSdata object containing the information of the city distances
    :return: a float corresponding to the total distance travelled according to the Path's particular cycle

    1- get distance between two adjacent cities
    2- add the value to D
    3- repeat for all following cities
    4- do not forget the distance between the first and the last in the cycle
    """

    C = Path.getcycle()
    L = len(C)
    D = 0

    for i in range(L):
        if i < L:
            s = csvdata.getdistance(C[i], C[i + 1])
        else:
            s = csvdata.getdistance(C[i], C[0])
        D += s

    return D


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

    C = Path.cycle
    R1 = []
    R2 = []
    N = num_mutations

    while len(R1) < N:
        ran = random.randint()
        while (ran in R1) or (ran in R2):
            ran = random.randint()
        R1.append(ran)

    while len(R2) < N:
        ran = random.randint()
        while (ran in R1) or (ran in R2):
            ran = random.randint()
        R2.append(ran)

    for i in range(N):
        c1 = C[R1[i]]
        c2 = C[R2[i]]
        C[R1[i]] = c2
        C[R2[i]] = c1

    m_path = Path()
    m_path.setcycle(C)

    return m_path


def crossover_parents(p1: Path, p2: Path) -> Path:
    pass


def mutate_population(pop: np.ndarray, mutation_rate: int, num_mutations: int):
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

    mutated_pop = pop.copy()
    L = len(mutated_pop)
    tomutate = int(L * mutation_rate / 100)
    randnumbers = []

    for i in range(tomutate - 1):
        r = random.randint(0, L)
        while r in randnumbers:
            r = random.randint(0, L)
        randnumbers.append(r)

    for index in randnumbers:
        mutated_pop[index] = mutate_path(pop[index], num_mutations)

    return mutated_pop


def initialize_population(pop_size: int, path_size: int) -> np.ndarray:
    pop = []
    for i in range(pop_size):
        pop.append(randomPath(path_size))
    return np.array(pop)


def selection_k_tournament(initial_pop: np.ndarray, desired_size: int) -> np.ndarray:
    new_pop = np.array
    for x in range(desired_size):
        parents = random.choices(initial_pop, k=K_TOURNAMENT)
        parents = sorted(parents, key=lambda agent: agent.fitness, reverse=True)
        new_pop.append(parents[0])
    return np.array()


def select_2_parents(pop: np.ndarray):
    # return p1, p2
    pass


def variation(population: np.ndarray) -> None:
    mutated_population = mutate_population(population, MUTATION_RATE, NUM_MUTATIONS)

    offspring = []
    for i in range(POP_SIZE):
        # how to choose 2 parents?,for now it's random
        p1, p2 = select_2_parents(mutate_population)
        offspring.append(crossover_parents(p1, p2))
    return np.concatenate(mutated_population, offspring)


def eliminate(intermediate_pop: np.ndarray, desired_size: int) -> np.ndarray:
    pass


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

        # Your code here .
        yourConvergenceTestsHere = True

        # initialize the population
        n_city = CSVdata.numcities()
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
