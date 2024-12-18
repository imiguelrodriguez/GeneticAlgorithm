import random
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, chromosome):
        self._chromosome = chromosome
        self._fitness = None

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        self._fitness = value

    @property
    def chromosome(self):
        return self._chromosome

    @chromosome.setter
    def chromosome(self, value):
        self._chromosome = value

class GeneticAlgorithm:
    def __init__(self, jobs, population_size=10 ):
        self.population = self.generate_population(population_size, jobs)
        for individual in self.population:
            self.fitness(individual, jobs)

    def fitness(self, individual, jobs):

        """
        Calcula el fitness (makespan) d'un individu i actualitza l'atribut fitness de la classe.

        :param individual: Objecte de la classe Individual.
        :param jobs: Llista de feines amb tasques (màquina, durada).
        """
        chromosome = individual.chromosome  # Obtenim el cromosoma de l'individu

        num_machines = max(machine for job in jobs for machine, _ in job) + 1
        machine_times = [0] * num_machines  # Temps de disponibilitat per cada màquina
        job_times = [0] * len(jobs)  # Temps de disponibilitat per cada feina

        # Calcula el makespan segons l'ordre del cromosoma
        for job, operation in chromosome:
            machine, duration = jobs[job][operation]
            start_time = max(machine_times[machine], job_times[job])
            end_time = start_time + duration

            # Actualitza els temps de la màquina i la feina
            machine_times[machine] = end_time
            job_times[job] = end_time

        # El makespan és el temps màxim entre totes les màquines
        makespan = max(machine_times)
        individual.fitness = makespan  # Actualitza el fitness


    def crossover(self, parent1, parent2, n_iter=1000):
        """
          Implementa el Two-Point Crossover entre dos individus.

          :param parent1: Objecte de la classe Individual (pare 1).
          :param parent2: Objecte de la classe Individual (pare 2).
          :return: Dos nous individus (fills) com a instàncies de la classe Individual.
          """
        # Obtenim els cromosomes dels pares
        chromosome1 = parent1.chromosome
        chromosome2 = parent2.chromosome

        # Assegura que els cromosomes tenen la mateixa longitud
        length = len(chromosome1)
        if length != len(chromosome2):
            raise ValueError("Els cromosomes dels dos pares han de tenir la mateixa longitud")

        valid = False
        iter = 0
        while not valid and iter < n_iter:
            # Selecciona dos punts de creuament aleatoris
            point1, point2 = sorted(random.sample(range(length), 2))

            # Genera els cromosomes dels fills
            child1_chromosome = chromosome1[:point1] + chromosome2[point1:point2] + chromosome1[point2:]
            child2_chromosome = chromosome2[:point1] + chromosome1[point1:point2] + chromosome2[point2:]

            valid = self.check_validity(child1_chromosome, jobs) and self.check_validity(child2_chromosome, jobs)
            iter += 1
        if not valid:
            return None, None
        # Crea dos nous individus
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)

        return child1, child2

    def mutation(self, individual, jobs, mutation_rate=0.1):
        """
          Aplica la mutació a un individu, assegurant que l'ordre de les operacions sigui correcte.

          :param individual: Objecte de la classe Individual.
          :param jobs: Llista de feines [(màquina, durada)].
          :param mutation_rate: Probabilitat de mutació.
          """
        chromosome = individual.chromosome[:]  # Fem una còpia del cromosoma

        if random.random() < mutation_rate:  # Probabilitat de mutació
            valid = False
            while not valid:
                # Seleccionem un índex aleatori dins del cromosoma
                index = random.randint(0, len(chromosome) - 1)

                # Generem una nova operació vàlida
                job = chromosome[index][0]
                possible_operations = list(range(len(jobs[job])))
                new_operation = random.choice(possible_operations)

                # Substituïm temporalment el gen
                temp_chromosome = chromosome[:]
                temp_chromosome[index] = (job, new_operation)

                # Comprovem si el nou cromosoma és vàlid
                if self.check_validity(temp_chromosome, jobs):
                    chromosome = temp_chromosome
                    valid = True

            # Actualitzem el cromosoma de l'individu
            individual.chromosome=chromosome

    def elitism(self, num_top_individuals=2):
        """
           Selects the top N individuals with the best fitness from the population.

           :param num_top_individuals: Number of best individuals to select.
           :return: List of top individuals sorted by fitness.
           """
        # Sort the population based on fitness (assuming lower fitness is better)
        sorted_population = sorted(self.population, key=lambda ind: ind.fitness)

        # Select the top N individuals
        top_individuals = sorted_population[:num_top_individuals]

        return top_individuals


    def generate_population(self, population_size, jobs):
        individuals = []
        for i in range(population_size):
            individuals.append(Individual(self.generate_chromosome(jobs)))
        return individuals

    def generate_chromosome(self, jobs):
        num_jobs = len(jobs)
        chromosome = []
        job_counts = [0] * num_jobs  # Inicialitza el comptador d'operacions per cada feina
        valid = False
        while not valid:
            while len(chromosome) < sum(len(job) for job in jobs):
                job = random.choice(range(num_jobs))  # Selecciona una feina aleatòria
                if job_counts[job] < len(jobs[job]):  # Encara queden operacions per aquesta feina
                    chromosome.append((job, job_counts[job]))
                    job_counts[job] += 1
            self.plot_gantt(chromosome, jobs)
            valid = True


        return chromosome

    def plot_gantt(self, chromosome, jobs):
        """
        Genera un diagrama de Gantt a partir d'un cromosoma i les feines.

        :param chromosome: Seqüència de tasques [(job, operation)].
        :param jobs: Llista de feines amb les seves tasques [(machine, time)].
        """
        # Inicialitza temps de disponibilitat per cada màquina i feina
        num_machines = max(machine for job in jobs for machine, _ in job) + 1
        machine_times = [0] * num_machines
        job_times = [0] * len(jobs)

        # Llistes per emmagatzemar les dades del diagrama
        machine_tasks = {i: [] for i in range(num_machines)}  # Cada màquina tindrà tasques
        task_colors = {0: 'tab:blue', 1: 'tab:red', 2: 'tab:green'}  # Colors per JOBS

        # Processa cada tasca en el cromosoma
        for job, operation in chromosome:
            machine, duration = jobs[job][operation]

            # Calcula el temps d'inici i fi de la tasca
            start_time = max(machine_times[machine], job_times[job])
            end_time = start_time + duration

            # Guarda la tasca per a la màquina
            machine_tasks[machine].append((start_time, duration, job))

            # Actualitza els temps
            machine_times[machine] = end_time
            job_times[job] = end_time
        # Set to track jobs already added to the legend
        jobs_in_legend = set()

        # Dibuixa el diagrama de Gantt
        fig, ax = plt.subplots(figsize=(10, 6))
        for machine, tasks in machine_tasks.items():
            for start, duration, job in tasks:
                # Only add label to the first occurrence of each job
                label = f"Job {job}" if job not in jobs_in_legend else None
                jobs_in_legend.add(job)

                ax.barh(y=machine, width=duration, left=start, height=0.4,
                        color=task_colors[job], edgecolor='black', label=label)
                ax.text(start + duration / 2, machine, f"job({job},{operation})",
                        ha='center', va='center', color='white', fontsize=8)

        # Configuració del gràfic
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

        :param chromosome: List of (job, operation) pairs.
        :param jobs: List of jobs with their tasks [(machine, time)].
        :return: True if the chromosome is valid, False otherwise.
        """
        print(chromosome)
        num_jobs = len(jobs)
        job_counts = [0] * num_jobs  # To track the number of operations completed for each job

        for job, operation in chromosome:
            # 1. Check if the job index is valid
            if job < 0 or job >= num_jobs:
                return False

            # 2. Check if the operation index is valid for the given job
            if operation < 0 or operation >= len(jobs[job]):
                return False

            # 3. Check if the operation is in the correct sequence
            if operation != job_counts[job]:  # Operation is not the next expected one
                return False

            # Update the operation count for this job
            job_counts[job] += 1

        # 4. Check if all operations for all jobs are covered exactly once
        if job_counts != [len(job) for job in jobs]:
            return False

        return True

    def main_loop(self, num_generations=5):
        for generation in range(num_generations):
            descendants = []
            for pair in range(int(len(self.population)/2)):
                # Randomly select two individuals from the population
                ind1, ind2 = random.sample(self.population, 2)
                ch1, ch2 = self.crossover(ind1, ind2)
                if ch1 is None or ch2 is None:
                    continue
                self.mutation(ch1, jobs)
                self.mutation(ch2, jobs)
                descendants.append(ch1)
                descendants.append(ch2)
            descendants = self.elitism()
            self.population = self.population + descendants
            # evaluate fitness
            for individual in self.population:
                self.fitness(individual, jobs)

jobs = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3)]
]
ga = GeneticAlgorithm(jobs)
ga.main_loop()
print(ga.elitism())