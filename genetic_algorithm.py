import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, jobs, population_size=5 ):
        self.population = self.generate_population(population_size, jobs)

    def fitness(self):
        pass

    def crossover(self):
        pass

    def mutation(self):
        pass

    def elitism(self):
        pass

    def generate_population(self, population_size, jobs):
        individuals = []
        for i in range(population_size):
            individuals.append(self.generate_chromosome(jobs))
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
            if self.check_validity(chromosome, jobs):
                print(True)
                self.plot_gantt(chromosome, jobs)
                valid = True

        return chromosome

    def plot_gantt(self,chromosome, jobs):
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

        # Dibuixa el diagrama de Gantt
        fig, ax = plt.subplots(figsize=(10, 6))
        for machine, tasks in machine_tasks.items():
            for start, duration, job in tasks:
                ax.barh(y=machine, width=duration, left=start, height=0.4,
                        color=task_colors[job], edgecolor='black', label=f"Job {job}" if start == 0 else "")
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

    def main_loop(self):

        pass


jobs = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3)]
]
ga = GeneticAlgorithm(jobs)
ga.main_loop()
