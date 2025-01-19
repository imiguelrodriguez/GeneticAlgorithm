import math
import random
import matplotlib.pyplot as plt
from individual import Individual


class SimulatedAnnealing:
    def __init__(self, jobs, initial_temperature, cooling_rate, iterations):
        """
        Initialize the Simulated Annealing optimization for JSSP.

        :param jobs: List of jobs with their tasks (machine, duration).
        :type jobs: list[list[tuple[int, int]]]
        :param initial_temperature: Starting temperature for the SA algorithm.
        :type initial_temperature: float
        :param cooling_rate: Rate at which the temperature decreases (0 < cooling_rate < 1).
        :type cooling_rate: float
        :param iterations: Number of iterations per temperature level.
        :type iterations: int
        """
        self.jobs = jobs
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.iterations = iterations

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
        """
        chromosome = []
        job_counts = [0] * len(jobs)
        while len(chromosome) < sum(len(job) for job in jobs):
            job = random.choice(range(len(jobs)))
            if job_counts[job] < len(jobs[job]):
                chromosome.append((job, job_counts[job]))
                job_counts[job] += 1
        return chromosome

    def fitness(self, individual, jobs):
        """
        Calculate the fitness (makespan) of an individual.
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
        """
        mutated_chromosome = chromosome[:]
        i, j = random.sample(range(len(mutated_chromosome)), 2)
        mutated_chromosome[i], mutated_chromosome[j] = mutated_chromosome[j], mutated_chromosome[i]
        return mutated_chromosome

    def acceptance_probability(self, current_fitness, new_fitness):
        """
        Calculate the acceptance probability for a new solution.
        """
        if new_fitness < current_fitness:
            return 1.0
        else:
            return math.exp((current_fitness - new_fitness) / self.temperature)

    def optimize(self):
        """
        Perform the Simulated Annealing optimization.
        """
        while self.temperature > 1:  # Stop when temperature is close to 0

            # Generate a new solution
            new_chromosome = self.mutate(self.current_solution.chromosome)
            new_solution = Individual(new_chromosome)
            self.fitness(new_solution, self.jobs)

            # Decide whether to accept the new solution
            if random.random() < self.acceptance_probability(
                    self.current_solution.fitness, new_solution.fitness):
                self.current_solution = new_solution

            # Update the best solution found
            if self.current_solution.fitness < self.best_fitness:
                self.best_solution = self.current_solution
                self.best_fitness = self.current_solution.fitness

            # Track the best fitness for plotting
            self.fitness_history.append(self.best_fitness)

            # Cool down
            self.temperature *= self.cooling_rate

        return self.best_solution

    def plot_fitness(self):
        """
        Plot the fitness over the iterations.
        """
        plt.plot(self.fitness_history)
        plt.xlabel("Temperature Steps")
        plt.ylabel("Best Fitness (Makespan)")
        plt.title("Simulated Annealing Optimization Progress")
        plt.show()
