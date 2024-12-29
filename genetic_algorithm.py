import random
import matplotlib.pyplot as plt

class Individual:
    """
       Represents an individual in the Genetic Algorithm population.

       :param chromosome: List of genes representing the individual's solution.
       :type chromosome: list[tuple[int, int]]

       **Attributes:**
       - **chromosome** (*list[tuple[int, int]]*): The individual's chromosome.
       - **fitness** (*float*): The fitness score of the individual.
       """
    def __init__(self, chromosome):
        self._chromosome = chromosome
        self._fitness = None

    @property
    def fitness(self):
        """
        Gets the fitness value of the individual.

        :return: Fitness value.
        :rtype: float
        """
        return self._fitness
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        """
        Sets the fitness value of the individual.

        :param value: Fitness value.
        :type value: float
        """
        self._fitness = value

    @property
    def chromosome(self):
        """
        Gets the chromosome of the individual.

        :return: Chromosome.
        :rtype: list[tuple[int, int]]
        """
        return self._chromosome

    @chromosome.setter
    def chromosome(self, value):
        """
        Sets the chromosome of the individual.

        :param value: Chromosome.
        :type value: list[tuple[int, int]]
        """
        self._chromosome = value

class GeneticAlgorithm:
    """
       A Genetic Algorithm implementation for Job Shop Scheduling Problem (JSSP).

       :param jobs: List of jobs, each represented by machine and duration pairs.
       :type jobs: list[list[tuple[int, int]]]
       :param population_size: Number of individuals in the population (default: 10).
       :type population_size: int, optional
       :param selection_method: Selection method ('rank' or 'tournament').
       :type selection_method: str, optional
       :param crossover_method: Crossover method ('one_point' or 'two_point').
       :type crossover_method: str, optional
       :param mutation_method: Mutation method ('one' or 'independent').
       :type mutation_method: str, optional

       **Methods:**
       - **fitness(individual, jobs)**: Calculate fitness for an individual.
       - **repair_chromosome(chromosome, jobs)**: Repair an invalid chromosome.
       - **crossover_onePoint(parent1, parent2, jobs)**: Perform one-point crossover.
       - **crossover_twoPoint(parent1, parent2, jobs)**: Perform two-point crossover.
       - **mutation_one(individual, jobs)**: Apply single mutation.
       - **mutation_independent(individual, jobs)**: Apply independent mutation.
       - **elitism(num_top_individuals)**: Select top individuals by fitness.
       - **generate_population(population_size, jobs)**: Generate initial population.
       - **generate_chromosome(jobs)**: Create a valid chromosome.
       - **check_validity(chromosome, jobs)**: Validate a chromosome.
       - **plot_gantt(chromosome, jobs)**: Plot a Gantt chart for a chromosome.
       - **main_loop(num_generations)**: Execute the Genetic Algorithm loop.
    """
    def __init__(self, jobs, population_size=10,
                 selection_method='rank',
                 crossover_method='one_point',
                 mutation_method='independent'):
        self.jobs = jobs
        self.population = self.generate_population(population_size, jobs)
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        for individual in self.population:
            self.fitness(individual, jobs)

    def select_parents(self):
        """
        Select two parents based on the specified selection method.

        This method delegates the selection to the appropriate method:
        - 'rank': Uses rank-based selection.
        - 'tournament': Uses tournament selection.

        :return: A tuple containing two selected parents from the population.
        :rtype: tuple[Individual, Individual]

        :raises ValueError: If an invalid selection method is specified.
        """
        if self.selection_method == 'rank':
            return self.rank_selection(self.population), self.rank_selection(self.population)
        elif self.selection_method == 'tournament':
            return self.tournament_selection(self.population), self.tournament_selection(self.population)
        else:
            raise ValueError("Invalid selection. Choose between 'rank' or 'tournament'.")

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents based on the specified method.

        This method delegates the crossover operation to the appropriate method:
        - 'one_point': Uses one-point crossover.
        - 'two_point': Uses two-point crossover.

        :param parent1: The first parent individual.
        :type parent1: Individual
        :param parent2: The second parent individual.
        :type parent2: Individual
        :return: A tuple containing two offspring individuals resulting from the crossover.
        :rtype: tuple[Individual, Individual]

        :raises ValueError: If an invalid crossover method is specified.
        """
        if self.crossover_method == 'one_point':
            return self.crossover_onePoint(parent1, parent2, self.jobs)
        elif self.crossover_method == 'two_point':
            return self.crossover_twoPoint(parent1, parent2, self.jobs)
        else:
            raise ValueError("Invalid crossover. Choose between 'one_point' or 'two_point'.")

    def mutate(self, individual):
        """
        Apply mutation to an individual based on the specified mutation method.

        This method delegates the mutation operation to the appropriate method:
        - 'independent': Applies independent mutation.
        - 'one': Applies one-point mutation.

        :param individual: The individual to be mutated.
        :type individual: Individual

        :raises ValueError: If an invalid mutation method is specified.
        """
        if self.mutation_method == 'independent':
            self.mutation_independent(individual, self.jobs)
        elif self.mutation_method == 'one':
            self.mutation_one(individual, self.jobs)
        else:
            raise ValueError("Invalid mutation method. Choose 'independent' or 'one'.")


    def fitness(self, individual, jobs):
        """
        Calculate the fitness (makespan) on an individual and upadtes the fitness attribute of the class

        :param individual: Object of the Individual class.
        :type individual: Individual
        :param jobs: List of jobs with tasks (machine, duration).
        :type jobs: list[list[tuple[int, int]]]
        """
        chromosome = individual.chromosome

        num_machines = max(machine for job in jobs for machine, _ in job) + 1
        machine_times = [0] * num_machines  # Availability time for each machine
        job_times = [0] * len(jobs)  # Availability time for each job

        for job, operation in chromosome:
            machine, duration = jobs[job][operation]
            start_time = max(machine_times[machine], job_times[job])
            end_time = start_time + duration

            # Update machine and job times
            machine_times[machine] = end_time
            job_times[job] = end_time

        makespan = max(machine_times)
        individual.fitness = makespan  # Update the fitness value

    def repair_chromosome(self, chromosome, jobs):
        """
        Repairs a chromosome by removing duplicates and assigning missing operations.

        :param chromosome: List of (job, operation) pairs.
        :type chromosome: list[tuple[int, int]]
        :param jobs: List of jobs with tasks (machine, duration).
        :type jobs: list[list[tuple[int, int]]]
        :return: A repaired chromosome.
        :rtype: list[tuple[int, int]]
        """
        expected_operations = [(job, op) for job in range(len(jobs)) for op in range(len(jobs[job]))]
        seen = set()
        repaired_chromosome = []

        # Delate duplicates
        for gene in chromosome:
            if gene not in seen:
                repaired_chromosome.append(gene)
                seen.add(gene)

        # Add missing operations
        missing_operations = [op for op in expected_operations if op not in seen]
        repaired_chromosome.extend(missing_operations)

        return repaired_chromosome


    def crossover_twoPoint(self, parent1, parent2, jobs, n_iter=1):
        """
        Perform two-point crossover between two parents.

        :param parent1: First parent individual.
        :type parent1: Individual
        :param parent2: Second parent individual.
        :type parent2: Individual
        :param jobs: List of jobs with tasks (machine, duration).
        :type jobs: list[list[tuple[int, int]]]
        :return: Two offspring individuals.
        :rtype: tuple[Individual, Individual]
        """
        chromosome1 = parent1.chromosome
        chromosome2 = parent2.chromosome

        length = len(chromosome1)
        if length != len(chromosome2):
            raise ValueError("The chromosomes of both parents must be the same length")

        valid = False
        iter = 0
        while not valid and iter < n_iter:
            # Select two random points
            point1, point2 = sorted(random.sample(range(length), 2))

            child1_chromosome = chromosome1[:point1] + chromosome2[point1:point2] + chromosome1[point2:]
            child2_chromosome = chromosome2[:point1] + chromosome1[point1:point2] + chromosome2[point2:]

            child1_chromosome = self.repair_chromosome(child1_chromosome, jobs)
            child2_chromosome = self.repair_chromosome(child2_chromosome, jobs)


            valid = self.check_validity(child1_chromosome, jobs) and self.check_validity(child2_chromosome, jobs)
            iter += 1
        if not valid:
            return None, None
        # Create two new individuals
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)

        return child1, child2

    def crossover_onePoint(self, parent1, parent2, jobs, n_iter=1):
        """
        Perform one-point crossover between two parents.

        :param parent1: First parent individual.
        :type parent1: Individual
        :param parent2: Second parent individual.
        :type parent2: Individual
        :param jobs: List of jobs with tasks (machine, duration).
        :type jobs: list[list[tuple[int, int]]]
        :return: Two offspring individuals.
        :rtype: tuple[Individual, Individual]
        """
        chromosome1 = parent1.chromosome
        chromosome2 = parent2.chromosome

        length = len(chromosome1)
        if length != len(chromosome2):
            raise ValueError("The chromosomes of both parents must be the same length")

        valid = False
        iter = 0
        while not valid and iter < n_iter:
            # Select one random point
            point = random.randint(1, length -1 )

            child1_chromosome = chromosome1[:point] + chromosome2[point:]
            child2_chromosome = chromosome2[:point] + chromosome1[point:]

            child1_chromosome = self.repair_chromosome(child1_chromosome, jobs)
            child2_chromosome = self.repair_chromosome(child2_chromosome, jobs)

            valid = self.check_validity(child1_chromosome, jobs) and self.check_validity(child2_chromosome, jobs)
            iter += 1
        if not valid:
            return None, None
        # Create two new individuals
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)

        return child1, child2

    def mutation_one(self, individual, jobs, mutation_rate=0.1):
        """
        Apply single-point mutation to an individual, ensuring the order of operations remains valid.

        This mutation selects a random gene in the chromosome and replaces its operation with a new valid one.

        :param individual: The individual to be mutated.
        :type individual: Individual
        :param jobs: List of jobs with their tasks (machine, duration).
        :type jobs: list[list[tuple[int, int]]]
        :param mutation_rate: Probability of mutation occurring (default: 0.1).
        :type mutation_rate: float

        :raises ValueError: If the chromosome becomes invalid after mutation.
        """
        chromosome = individual.chromosome[:]

        if random.random() < mutation_rate:
            valid = False
            while not valid:
                index = random.randint(0, len(chromosome) - 1)

                job = chromosome[index][0]
                possible_operations = list(range(len(jobs[job])))
                new_operation = random.choice(possible_operations)

                temp_chromosome = chromosome[:]
                temp_chromosome[index] = (job, new_operation)

                if self.check_validity(temp_chromosome, jobs):
                    chromosome = temp_chromosome
                    valid = True

            individual.chromosome=chromosome

    def mutation_independent(self, individual, jobs, mutation_rate=0.1):
        """
        Apply independent mutation to each gene in an individual's chromosome.

        Each gene has a probability (mutation_rate) of being mutated independently. Validity of the chromosome
        is ensured after mutations.

        :param individual: The individual to be mutated.
        :type individual: Individual
        :param jobs: List of jobs with their tasks (machine, duration).
        :type jobs: list[list[tuple[int, int]]]
        :param mutation_rate: Probability of mutation for each gene (default: 0.1).
        :type mutation_rate: float

        :raises ValueError: If the chromosome becomes invalid after mutations.
        """
        chromosome = individual.chromosome[:]
        valid = False

        while not valid:
            mutated_chromosome = chromosome[:]

            for index in range(len(mutated_chromosome)):
                if random.random() < mutation_rate:
                    job = mutated_chromosome[index][0]
                    possible_operations = list(range(len(jobs[job])))

                    current_operation = mutated_chromosome[index][1]
                    possible_operations.remove(current_operation)
                    new_operation = random.choice(possible_operations)

                    mutated_chromosome[index] = (job, new_operation)

            if self.check_validity(mutated_chromosome, jobs):
                valid = True

        individual.chromosome = mutated_chromosome

    def elitism(self, num_top_individuals=2):
        """
        Select the top N individuals with the best fitness from the population.

        :param num_top_individuals: Number of best individuals to select.
        :type num_top_individuals: int
        :return: List of top individuals.
        :rtype: list[Individual]
        """
        # Sort the population based on fitness
        sorted_population = sorted(self.population, key=lambda ind: ind.fitness)

        # Select the top N individuals
        top_individuals = sorted_population[:num_top_individuals]

        return top_individuals

    def generate_population(self, population_size, jobs):
        individuals = []
        while len(individuals) < population_size:
            chromosome = self.generate_chromosome(jobs)
            if self.check_validity(chromosome, jobs):
                individuals.append(Individual(chromosome))
        return individuals

    def generate_chromosome(self, jobs):
        """
        Generate a valid chromosome for the Job Shop Scheduling Problem.

        A chromosome represents a sequence of (job, operation) pairs, ensuring that all tasks are included
        exactly once in a valid order.

        :param jobs: List of jobs, where each job is a list of (machine, duration) pairs.
        :type jobs: list[list[tuple[int, int]]]
        :return: A valid chromosome representing the sequence of job operations.
        :rtype: list[tuple[int, int]]
        """
        num_jobs = len(jobs)
        chromosome = []
        job_counts = [0] * num_jobs
        valid = False
        while not valid:
            while len(chromosome) < sum(len(job) for job in jobs):
                job = random.choice(range(num_jobs))  # Selection of random job
                if job_counts[job] < len(jobs[job]):
                    chromosome.append((job, job_counts[job]))
                    job_counts[job] += 1
            self.plot_gantt(chromosome, jobs)
            valid = True

        return chromosome

    def plot_gantt(self, chromosome, jobs):
        """
        Generate a Gantt chart from a chromosome and job definitions.

        The Gantt chart visually represents job operations on machines over time, showing the start and
        end times of each task on each machine.

        :param chromosome: A sequence of tasks represented as (job, operation) pairs.
        :type chromosome: list[tuple[int, int]]
        :param jobs: List of jobs with their tasks represented as (machine, duration) pairs.
        :type jobs: list[list[tuple[int, int]]]

        :raises ValueError: If the chromosome or jobs data is malformed.
        """
        num_machines = max(machine for job in jobs for machine, _ in job) + 1
        machine_times = [0] * num_machines
        job_times = [0] * len(jobs)

        machine_tasks = {i: [] for i in range(num_machines)}
        task_colors = {0: 'tab:blue', 1: 'tab:red', 2: 'tab:green'}

        for job, operation in chromosome:
            machine, duration = jobs[job][operation]

            # Calculate task start and end times
            start_time = max(machine_times[machine], job_times[job])
            end_time = start_time + duration

            machine_tasks[machine].append((start_time, duration, job))

            # Update machine and job availability times
            machine_times[machine] = end_time
            job_times[job] = end_time
        # Track jobs added to the legend
        jobs_in_legend = set()

        # Draw the Gantt chart
        fig, ax = plt.subplots(figsize=(10, 6))
        for machine, tasks in machine_tasks.items():
            for start, duration, job in tasks:
                label = f"Job {job}" if job not in jobs_in_legend else None
                jobs_in_legend.add(job)

                ax.barh(y=machine, width=duration, left=start, height=0.4,
                        color=task_colors[job], edgecolor='black', label=label)
                ax.text(start + duration / 2, machine, f"job({job},{operation})",
                        ha='center', va='center', color='white', fontsize=8)

        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f"Machine {i}" for i in range(num_machines)])
        ax.set_xlabel("Time")
        ax.set_title("Gantt Chart for Job Shop Scheduling")
        ax.invert_yaxis()  # Les màquines van de dalt cap avall
        plt.legend(loc='upper right')
        plt.show()

    def check_validity(self, chromosome, jobs):
        """
        Check if a chromosome is valid for the Job Shop Scheduling Problem.

        :param chromosome: List of (job, operation) pairs representing the chromosome.
        :type chromosome: list[tuple[int, int]]
        :param jobs: List of jobs with their tasks represented as (machine, duration) pairs.
        :type jobs: list[list[tuple[int, int]]]
        :return: True if the chromosome is valid, False otherwise.
        :rtype: bool

        :raises ValueError: If invalid job or operation indices are detected.
        """
        print(chromosome)
        num_jobs = len(jobs)
        job_counts = [0] * num_jobs

        for job, operation in chromosome:
            # Check if job index is within bounds
            if job < 0 or job >= num_jobs:
                return False

            # Check if operation index is within bounds
            if operation < 0 or operation >= len(jobs[job]):
                return False

            # Check if the operation follows the correct sequence
            if operation != job_counts[job]:
                return False

            job_counts[job] += 1

        # Verify all operations are covered exactly once
        if job_counts != [len(job) for job in jobs]:
            return False

        return True

    def tournament_selection(self, population, k=3):
        """
        Select the best individual from a random subset of the population using tournament selection.

        :param population: List of individuals in the current population.
        :type population: list[Individual]
        :param k: Number of individuals to participate in the tournament (default: 3).
        :type k: int
        :return: The individual with the best fitness from the tournament.
        :rtype: Individual

        :raises ValueError: If k is larger than the population size.
        """
        tournament = random.sample(population, k)
        best_individual = min(tournament, key=lambda ind: ind.fitness)
        return best_individual

    def rank_selection(self, population):
        """
        Select an individual from the population using rank-based selection.

        :param population: List of individuals in the current population.
        :type population: list[Individual]
        :return: An individual selected based on rank probabilities.
        :rtype: Individual
        """
        # Sort the population based on fitness
        sorted_population = sorted(population, key=lambda ind: ind.fitness)

        # Assign ranks
        ranks = list(range(1, len(population) + 1))  # Rang 1 a N

        # Calculate selection probabilities
        total_ranks = sum(ranks)
        probabilities = [rank / total_ranks for rank in ranks]

        # Select one individual based on rank probabilities
        selected = random.choices(sorted_population, weights=probabilities, k=1)[0]

        return selected

    def main_loop(self, num_generations=5):
        """
        Execute the Genetic Algorithm main loop.

        :param num_generations: Number of generations to run the algorithm.
        :type num_generations: int
        """
        for generation in range(num_generations):
            descendants = []
            for _ in range(int(len(self.population) / 2)):
                # Selection
                ind1, ind2 = self.select_parents()

                # Crossover
                ch1, ch2 = self.crossover(ind1, ind2)
                if ch1 is None or ch2 is None:
                    continue

                # Mutation
                self.mutate(ch1)
                self.mutate(ch2)

                descendants.append(ch1)
                descendants.append(ch2)

            # Etilism
            top_individuals = self.elitism()
            self.population = top_individuals + self.population

            for individual in self.population:
                self.fitness(individual, self.jobs)


jobs = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3)]
]
ga = GeneticAlgorithm(
    jobs,
    population_size=10,
    selection_method='tournament',  # 'rank', 'tournament'
    crossover_method='two_point',   # 'one_point' 'two_point'
    mutation_method='independent'           # 'independent' 'one'
)
ga.main_loop()
best_individuals = ga.elitism()
print("🏆 Best Individual")
for i, ind in enumerate(best_individuals):
    print(f"Individual {i + 1}: Chromosome: {ind.chromosome}, Fitness: {ind.fitness}")