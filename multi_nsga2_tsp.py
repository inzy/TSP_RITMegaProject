"""
objectives: Traveling Salesman Problem (TSP) with multi-objective optimization using the Non-dominated Sorting Genetic Algorithm II (NSGA-II). And optimized for both total distance and total cost.

Explanation:
1.Individual Class:
a.Represents a single solution (tour) with methods to calculate the total distance and cost.

2.Initialization:
a.Generates an initial population of random tours.

3.Non-dominated Sorting:
a.Sorts the population into different fronts based on Pareto dominance.
b.Individuals in the first front are non-dominated by any other individuals.

4.Crowding Distance Calculation:
a.Measures the density of solutions surrounding a particular solution in the objective space to maintain diversity.

5.Selection:
a.Selects individuals based on rank and crowding distance for the next generation.

6.Crossover and Mutation:
a.Implements crossover and mutation to generate new solutions.

7.Evolution:
a.Evolves the population over a number of generations.

8.Run and Plot:
Runs the NSGA-II algorithm and plots the Pareto front.
"""



import math
import random
import matplotlib.pyplot as plt

class SolveTSPUsingNSGAII:
    class Individual:
        def __init__(self, tour, nodes):
            self.tour = tour
            self.nodes = nodes
            self.distance = self.calculate_distance()
            self.cost = self.calculate_cost()
            self.rank = 0
            self.crowding_distance = 0

        def calculate_distance(self):
            distance = 0.0
            for i in range(len(self.tour)):
                distance += math.sqrt(
                    pow(self.nodes[self.tour[i]][0] - self.nodes[self.tour[(i + 1) % len(self.tour)]][0], 2.0) +
                    pow(self.nodes[self.tour[i]][1] - self.nodes[self.tour[(i + 1) % len(self.tour)]][1], 2.0))
            return distance

        def calculate_cost(self):
            return sum([random.uniform(1, 10) for _ in range(len(self.tour))])  # Random cost for demonstration

    def __init__(self, population_size=100, generations=50, nodes=None):
        self.population_size = population_size
        self.generations = generations
        self.nodes = nodes
        self.population = self.initialize_population()
        self.fronts = []

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            tour = random.sample(range(len(self.nodes)), len(self.nodes))
            population.append(self.Individual(tour, self.nodes))
        return population

    def non_dominated_sorting(self):
        fronts = [[]]
        for p in self.population:
            p.dominated_solutions = []
            p.domination_count = 0
            for q in self.population:
                if self.dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self.dominates(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def dominates(self, p, q):
        return (p.distance < q.distance and p.cost <= q.cost) or (p.distance <= q.distance and p.cost < q.cost)

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            for individual in front:
                individual.crowding_distance = 0
            for m in range(2):
                front.sort(key=lambda x: x.distance if m == 0 else x.cost)
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')
                for i in range(1, len(front) - 1):
                    front[i].crowding_distance += (front[i + 1].distance - front[i - 1].distance) / (max(front, key=lambda x: x.distance).distance - min(front, key=lambda x: x.distance).distance) if m == 0 else \
                                                  (front[i + 1].cost - front[i - 1].cost) / (max(front, key=lambda x: x.cost).cost - min(front, key=lambda x: x.cost).cost)

    def selection(self):
        new_population = []
        for front in self.fronts:
            self.calculate_crowding_distance(front)
            front.sort(key=lambda x: (x.rank, -x.crowding_distance))
            new_population.extend(front)
            if len(new_population) >= self.population_size:
                break
        return new_population[:self.population_size]

    def crossover(self, parent1, parent2):
        cut = random.randint(0, len(parent1.tour) - 1)
        child1_tour = parent1.tour[:cut] + [gene for gene in parent2.tour if gene not in parent1.tour[:cut]]
        child2_tour = parent2.tour[:cut] + [gene for gene in parent1.tour if gene not in parent2.tour[:cut]]
        return self.Individual(child1_tour, self.nodes), self.Individual(child2_tour, self.nodes)

    def mutate(self, individual):
        idx1, idx2 = random.sample(range(len(individual.tour)), 2)
        individual.tour[idx1], individual.tour[idx2] = individual.tour[idx2], individual.tour[idx1]
        individual.distance = individual.calculate_distance()
        individual.cost = individual.calculate_cost()

    def evolve(self):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(self.population, 2)
            child1, child2 = self.crossover(parent1, parent2)
            if random.random() < 0.1:
                self.mutate(child1)
            if random.random() < 0.1:
                self.mutate(child2)
            new_population.extend([child1, child2])
        self.population = new_population

    def run(self):
        for generation in range(self.generations):
            print(f'Generation {generation + 1}/{self.generations}')
            self.fronts = self.non_dominated_sorting()
            self.population = self.selection()
            self.evolve()
        print('Finished Optimization')
        for front in self.fronts[0]:
            print(f'Distance: {front.distance}, Cost: {front.cost}, Tour: {front.tour}')

    def plot(self):
        plt.figure(figsize=(10, 6))
        for front in self.fronts:
            distances = [ind.distance for ind in front]
            costs = [ind.cost for ind in front]
            plt.scatter(distances, costs)
        plt.xlabel('Total Distance')
        plt.ylabel('Total Cost')
        plt.title('Pareto Front')
        plt.show()

if __name__ == '__main__':
    _nodes = [(random.uniform(-400, 400), random.uniform(-400, 400)) for _ in range(0, 15)]
    nsga_ii = SolveTSPUsingNSGAII(population_size=100, generations=50, nodes=_nodes)
    nsga_ii.run()
    nsga_ii.plot()
