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

    def __str__(self):
        chr = ""
        for job in self._chromosome:
            chr = chr + str(job)
        return chr + ", Fitness: " + str(self.fitness)
