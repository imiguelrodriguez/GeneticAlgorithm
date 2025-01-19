from IPython.core.pylabtools import figsize

from genetic_algorithm import GeneticAlgorithm
from simulated_annealing import SimulatedAnnealing


def load_dataset(filepath: str) -> list[list[tuple[int, int]]]:
    """
    Load a dataset from a text file and convert it into the jobs format.

    :param filepath: Path to the dataset file.
    :type filepath: str
    :return: A list of jobs, where each job is a list of (machine, duration) tuples.
    :rtype: list[list[tuple[int, int]]]
    """
    with open(filepath, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]

    for i, line in enumerate(lines):
        if line[0].isdigit():
            num_jobs, num_machines = map(int, line.split())
            start_line = i + 1
            break
    else:
        raise ValueError("Could not find the line with number of jobs and machines.")

    jobs = []
    for i in range(num_jobs):
        tokens = list(map(int, lines[start_line + i].split()))
        job = [(tokens[j], tokens[j + 1]) for j in range(0, len(tokens), 2)]
        jobs.append(job)

    return jobs


def run_ga_algorithm(jobs, population_size=10, selection='rank', crossover='two_point', mutation='independent', iterations=1500,
                  std_threshold=2, figsize=(12,7), visualize=False):
    """
    Run the Genetic Algorithm with given parameters.

    :param jobs: Dataset of jobs.
    :param population_size: Population size for the algorithm.
    :param selection: Selection method.
    :param crossover: Crossover method.
    :param mutation: Mutation method.
    :param iterations: Number of iterations to avoid infinite loops.
    :param visualize: Boolean flag to plot results.
    """
    ga = GeneticAlgorithm(
        jobs=jobs,
        population_size=population_size,
        selection_method=selection,
        crossover_method=crossover,
        mutation_method=mutation, iterations=iterations,
        std_threshold=std_threshold, figsize=figsize
    )
    ga.main_loop()
    best_individuals = ga.elitism(num_top_individuals=10)

    print("🏆 Best Individuals")
    for i, ind in enumerate(best_individuals):
        print(f"Individual {i + 1}: Chromosome: {ind.chromosome}, Fitness: {ind.fitness}")
    print(len(ga.population))

    if visualize:
        ga.plot_gantt(best_individuals[0].chromosome, jobs)

def run_sa_algorithm(jobs, initial_temperature=100, cooling_rate=0.85, lower_t=0.01, visualize=True):
    """
        Run the Genetic Algorithm with given parameters.

        :param jobs: Dataset of jobs.
        :param initial_temperature: Initial temperature of the SA algorithm.
        :param cooling_rate: Rate at which the temperature will decrease at each step.
        :param crossover: Crossover method.
        :param mutation: Mutation method.
        :param iterations: Number of iterations to avoid infinite loops.
        :param visualize: Boolean flag to plot results.
        """
    sa = SimulatedAnnealing(
        jobs=jobs, initial_temperature=initial_temperature, cooling_rate=cooling_rate, lower_t=lower_t
    )
    best = sa.optimize()

    print("🏆 Best fitness")
    print(best)

    if visualize:
        sa.plot_fitness()