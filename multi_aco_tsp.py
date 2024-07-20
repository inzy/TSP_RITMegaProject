"""
objectives: minimizing the total distance and minimizing the total cost or time. 
Implemented this using Pareto fronts for multi-objective optimization.

Explanation:
1.Multi-objective Representation:
a.Edge Class: Now includes both weight (distance) and cost.
b.Ant Class: Computes both distance and cost for a tour.

2.Pareto Front:
a._pareto Method: Runs the ACO and updates pheromones based on both distance and cost. Collects solutions and performs non-dominated sorting to maintain the Pareto front.
b._non_dominated_sort Method: Determines the Pareto front from a list of solutions.
c._dominates Method: Checks if one solution dominates another based on distance and cost.

3.Running the Code:
a.Initializes the problem with random nodes.
b.Runs the Pareto ACO mode.
c.Outputs the Pareto front with the sequences of nodes, distances, and costs.
d.Plots each solution in the Pareto front.
"""



import math
import random
from matplotlib import pyplot as plt

class SolveTSPUsingACO:
    class Edge:
        def __init__(self, a, b, weight, cost, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight
            self.cost = cost
            self.pheromone = initial_pheromone

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = num_nodes
            self.edges = edges
            self.tour = None
            self.distance = 0.0
            self.cost = 0.0

        def _select_node(self):
            roulette_wheel = 0.0
            unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
            heuristic_total = 0.0
            for unvisited_node in unvisited_nodes:
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
            for unvisited_node in unvisited_nodes:
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
            random_value = random.uniform(0.0, roulette_wheel)
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes:
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour(self):
            self.tour = [random.randint(0, self.num_nodes - 1)]
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node())
            return self.tour

        def get_distance_and_cost(self):
            self.distance = 0.0
            self.cost = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
                self.cost += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].cost
            return self.distance, self.cost

    def __init__(self, mode='Pareto', colony_size=10, elitist_weight=1.0, min_scaling_factor=0.001, alpha=1.0, beta=3.0,
                 rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=100, nodes=None, labels=None):
        self.mode = mode
        self.colony_size = colony_size
        self.elitist_weight = elitist_weight
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        self.num_nodes = len(nodes)
        self.nodes = nodes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                distance = math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0))
                cost = random.uniform(1, 10)  # Assigning random cost for demonstration
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, distance, cost, initial_pheromone)
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)]
        self.pareto_front = []

    def _add_pheromone(self, tour, distance, cost, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / (distance + cost)
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

    def _pareto(self):
        for step in range(self.steps):
            current_solutions = []
            for ant in self.ants:
                tour = ant.find_tour()
                distance, cost = ant.get_distance_and_cost()
                current_solutions.append((tour, distance, cost))
                self._add_pheromone(tour, distance, cost)
            self.pareto_front = self._non_dominated_sort(current_solutions)
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

    def _non_dominated_sort(self, solutions):
        pareto_front = []
        for solution in solutions:
            if not any(self._dominates(other, solution) for other in solutions):
                pareto_front.append(solution)
        return pareto_front

    def _dominates(self, solution1, solution2):
        return (solution1[1] < solution2[1] and solution1[2] <= solution2[2]) or \
               (solution1[1] <= solution2[1] and solution1[2] < solution2[2])

    def run(self):
        print('Started : {0}'.format(self.mode))
        if self.mode == 'Pareto':
            self._pareto()
        print('Ended : {0}'.format(self.mode))
        for tour, distance, cost in self.pareto_front:
            print('Sequence : <- {0} ->'.format(' - '.join(str(self.labels[i]) for i in tour)))
            print('Total distance : {0}, Total cost : {1}\n'.format(round(distance, 2), round(cost, 2)))

    def plot(self, line_width=1, point_radius=math.sqrt(2.0), annotation_size=8, dpi=120, save=True, name=None):
        for tour, distance, cost in self.pareto_front:
            x = [self.nodes[i][0] for i in tour]
            x.append(x[0])
            y = [self.nodes[i][1] for i in tour]
            y.append(y[0])
            plt.plot(x, y, linewidth=line_width)
            plt.scatter(x, y, s=math.pi * (point_radius ** 2.0))
            plt.title(f"{self.mode} - Distance: {round(distance, 2)}, Cost: {round(cost, 2)}")
            for i in tour:
                plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)
            if save:
                if name is None:
                    name = f'{self.mode}_{round(distance, 2)}_{round(cost, 2)}.png'
                plt.savefig(name, dpi=dpi)
            plt.show()
            plt.gcf().clear()

if __name__ == '__main__':
    _colony_size = 5
    _steps = 50
    _nodes = [(random.uniform(-400, 400), random.uniform(-400, 400)) for _ in range(0, 15)]
    pareto = SolveTSPUsingACO(mode='Pareto', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    pareto.run()
    pareto.plot()
