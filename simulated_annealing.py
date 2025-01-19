import math
import random
import matplotlib.pyplot as plt
from individual import Individual


class SimulatedAnnealing:
    """
    Simulated Annealing optimization for the Job Shop Scheduling Problem (JSSP).

    This class implements a Simulated Annealing algorithm to optimize the makespan
    of schedules for JSSP by iteratively refining solutions while balancing exploration
    and exploitation through a temperature-based acceptance criterion.
    """

    def __init__(self, jobs, initial_temperature=100, cooling_rate=0.95, lower_t=0.001):
        """
        Initialize the Simulated Annealing optimization for JSSP.

        :param jobs: List of jobs with their tasks (machine, duration).
        :type jobs: list[list[tuple[int, int]]]
        :param initial_temperature: Starting temperature for the SA algorithm.
        :type initial_temperature: float
        :param cooling_rate: Rate at which the temperature decreases (0 < cooling_rate < 1).
        :type cooling_rate: float
        :param lower_t: Minimum temperature threshold to stop the algorithm.
        :type lower_t: float
        """
        self.jobs = jobs
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.lower_t = lower_t

        # Generate an initial solution
        self.current_solution = Individual(self.generate_chromosome(jobs))
        self.fitness(self.current_solution, jobs)

        # Keep track of the best solution found
        self.best_solution = self.current_solution
        self.best_fitness = self.current_solution.fitness
        self.fitness_history = [self.best_fitness]

    def generate_chromosome(self, jobs):
        """
        Generate a valid chromosome for the JSSP.

        A chromosome is a sequence of (job, operation) pairs, ensuring that all
        operations are included exactly once in a valid order.

        :param jobs: List of jobs with tasks (machine, duration).
        :type jobs: list[list[tuple[int, int]]]
        :return: A valid chromosome.
        :rtype: list[tuple[int, int]]
        """
        chromosome = []
        job_counts = [0] * len(jobs)
        valid = False
        while not valid:
            while len(chromosome) < sum(len(job) for job in jobs):
                job = random.choice(range(len(jobs)))
                if job_counts[job] < len(jobs[job]):
                    chromosome.append((job, job_counts[job]))
                    job_counts[job] += 1
            if self.check_validity(chromosome, jobs):
                valid = True
        return chromosome

    def check_validity(self, chromosome, jobs):
        """
        Check if a chromosome is valid for the Job Shop Scheduling Problem.

        Validity is ensured by checking that all operations are covered exactly
        once in the correct order for each job.

        :param chromosome: List of (job, operation) pairs representing the chromosome.
        :type chromosome: list[tuple[int, int]]
        :param jobs: List of jobs with their tasks represented as (machine, duration) pairs.
        :type jobs: list[list[tuple[int, int]]]
        :return: True if the chromosome is valid, False otherwise.
        :rtype: bool
        """
        num_jobs = len(jobs)
        job_counts = [0] * num_jobs

        for job, operation in chromosome:
            if job < 0 or job >= num_jobs:
                return False
            if operation < 0 or operation >= len(jobs[job]):
                return False
            if operation != job_counts[job]:
                return False
            job_counts[job] += 1

        if job_counts != [len(job) for job in jobs]:
            return False

        return True

    def fitness(self, individual, jobs):
        """
        Calculate the fitness (makespan) of an individual.

        Fitness is computed as the makespan, which is the maximum time required
        to complete all jobs on their assigned machines.

        :param individual: The individual whose fitness is being calculated.
        :type individual: Individual
        :param jobs: List of jobs with their tasks (machine, duration).
        :type jobs: list[list[tuple[int, int]]]
        """
        chromosome = individual.chromosome
        num_machines = max(machine for job in jobs for machine, _ in job) + 1
        machine_times = [0] * num_machines
        job_times = [0] * len(jobs)

        for job, operation in chromosome:
            machine, duration = jobs[job][operation]
            start_time = max(machine_times[machine], job_times[job])
            end_time = start_time + duration
            machine_times[machine] = end_time
            job_times[job] = end_time

        makespan = max(machine_times)
        individual.fitness = makespan

    def mutate(self, chromosome):
        """
        Mutate a chromosome by randomly swapping two genes.

        :param chromosome: Chromosome to mutate.
        :type chromosome: list[tuple[int, int]]
        :return: Mutated chromosome.
        :rtype: list[tuple[int, int]]
        """
        mutated_chromosome = chromosome[:]
        i, j = random.sample(range(len(mutated_chromosome)), 2)
        mutated_chromosome[i], mutated_chromosome[j] = mutated_chromosome[j], mutated_chromosome[i]
        return mutated_chromosome

    def acceptance_probability(self, current_fitness, new_fitness):
        """
        Calculate the acceptance probability for a new solution.

        :param current_fitness: Fitness of the current solution.
        :type current_fitness: float
        :param new_fitness: Fitness of the new solution.
        :type new_fitness: float
        :return: Acceptance probability for the new solution.
        :rtype: float
        """
        if new_fitness < current_fitness:
            return 1.0
        else:
            return math.exp((current_fitness - new_fitness) / self.temperature)

    def optimize(self):
        """
        Perform the Simulated Annealing optimization.

        Iteratively refines the current solution by generating new solutions,
        accepting them probabilistically based on temperature, and updating the
        temperature after each step. Stops when the temperature falls below
        the minimum threshold.

        :return: The best solution found during the optimization.
        :rtype: Individual
        """
        while self.temperature > self.lower_t:
            new_chromosome = self.mutate(self.current_solution.chromosome)
            new_solution = Individual(new_chromosome)
            self.fitness(new_solution, self.jobs)

            if random.random() < self.acceptance_probability(
                    self.current_solution.fitness, new_solution.fitness):
                self.current_solution = new_solution

            if self.current_solution.fitness < self.best_fitness:
                self.best_solution = self.current_solution
                self.best_fitness = self.current_solution.fitness

            self.fitness_history.append(self.best_fitness)
            self.temperature *= self.cooling_rate

        return self.best_solution

    def plot_fitness(self):
        """
        Plot the fitness (makespan) progression over the optimization process.

        This visualization provides insights into how the algorithm converges
        towards the best solution.
        """
        plt.plot(self.fitness_history)
        plt.xlabel("Temperature Steps")
        plt.ylabel("Best Fitness (Makespan)")
        plt.title("Simulated Annealing Optimization Progress")
        plt.show()
