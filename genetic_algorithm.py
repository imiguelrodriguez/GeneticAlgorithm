import random

class GeneticAlgorithm:
    def __init__(self, jobs, population_size=20, ):
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
                valid = True
        return chromosome

    def check_validity(self, chromosome, jobs):

        pass

    def main_loop(self):

        pass


jobs = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3)]
]
ga = GeneticAlgorithm(jobs)
ga.main_loop()
