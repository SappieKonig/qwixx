from copy import deepcopy
from random import choice
import numpy as np
import random
import pickle

from qwixx.game import Color, compete
from qwixx.player import Player


class RandomPlayer(Player):
    def move(self, actions, is_main, scoreboard, scoreboards):
        return choice(actions)


class ImprovedPlayer(Player):
    """
    An improved agent that no longer picks a random move, but who picks a move that leaves the fewest open squares
    """
    def cost(self, scoreboard, actions):
        """
        Measures the cost of a set of actions. Formally, it's the number of squares left empty by playing a move
        """
        # arbitrarily decide that the cost for taking no action is 10
        if len(actions) == 0:
            return 10
        scoreboard = deepcopy(scoreboard)
        c = 0
        for action in actions:
            if len(scoreboard[action.color]) != 0:
                c += abs(scoreboard[action.color][-1] - action.n) - 1
            else:
                if action.color in [Color.RED, Color.YELLOW]:
                    c += abs(action.n - 1) - 1
                else:
                    c += abs(action.n - 13) - 1
        return c / len(actions)

    def move(self, actions, is_main, scoreboard, scoreboards):
        best_action = []
        cost = float('inf')
        if not is_main:
            for action in actions:
                new_c = self.cost(scoreboard, action)
                if new_c < cost:
                    cost = new_c
                    best_action = action

        else:
            for action in actions:
                new_c = self.cost(scoreboard, action)
                if new_c < cost:
                    cost = new_c
                    best_action = action
        return best_action



class GeneticAlgorithmAgent(Player):
    def __init__(self, genome_size=200):
        self.genome_size = genome_size
        self.genome = np.random.rand(genome_size)

    def move(self, actions, is_main, scoreboard, scoreboards):
        state = self.get_state_vector(scoreboard, is_main)
        action_values = [self.evaluate_action(state, action) for action in actions]
        return actions[np.argmax(action_values)]

    def get_state_vector(self, scoreboard, is_main):
        state = []
        for color in Color:
            state.extend([1 if i in scoreboard[color] else 0 for i in range(2, 13)])
        state.extend([scoreboard.crosses, scoreboard.score, int(is_main)])
        return np.array(state)

    def evaluate_action(self, state, action):
        action_vector = self.get_action_vector(action)
        input_vector = np.concatenate([state, action_vector])
        return np.dot(input_vector, self.genome[:len(input_vector)])

    def get_action_vector(self, action):
        vector = np.zeros(8)  # 4 colors * 2 (color, number)
        for i, act in enumerate(action):
            vector[i*2] = list(Color).index(act.color)
            vector[i*2+1] = act.n
        return vector

class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.01, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = [GeneticAlgorithmAgent() for _ in range(population_size)]

    def evolve(self, generations=100, games_per_individual=10):
        for generation in range(generations):
            fitnesses = self.evaluate_population(games_per_individual)
            new_population = []

            # Elitism: keep the best individual
            elite = self.population[np.argmax(fitnesses)]
            new_population.append(elite)

            while len(new_population) < self.population_size:
                parent1, parent2 = self.selection(fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population[:self.population_size]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")

        return elite

    def evaluate_population(self, games_per_individual):
        fitnesses = []
        for individual in self.population:
            wins, draws, losses = compete(individual, RandomPlayer(), games_per_individual)
            fitnesses.append(wins + 0.5 * draws)
        return fitnesses

    def selection(self, fitnesses):
        total_fitness = sum(fitnesses)
        selection_probs = [f/total_fitness for f in fitnesses]
        selected_indices = np.random.choice(len(self.population), 2, p=selection_probs, replace=False)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, parent1.genome_size - 1)
            child1_genome = np.concatenate([parent1.genome[:crossover_point], parent2.genome[crossover_point:]])
            child2_genome = np.concatenate([parent2.genome[:crossover_point], parent1.genome[crossover_point:]])
        else:
            child1_genome = parent1.genome.copy()
            child2_genome = parent2.genome.copy()

        child1 = GeneticAlgorithmAgent()
        child2 = GeneticAlgorithmAgent()
        child1.genome = child1_genome
        child2.genome = child2_genome
        return child1, child2

    def mutate(self, individual):
        mutation_mask = np.random.random(individual.genome_size) < self.mutation_rate
        individual.genome[mutation_mask] += np.random.normal(0, 0.1, size=np.sum(mutation_mask))
        individual.genome = np.clip(individual.genome, 0, 1)

class RandomPlayer(Player):
    def move(self, actions, is_main, scoreboard, scoreboards):
        return random.choice(actions)


class GA_Player(Player):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            self.genome = pickle.load(f)
        self.genome_size = len(self.genome)

    def move(self, actions, is_main, scoreboard, scoreboards):
        state = self.get_state_vector(scoreboard, is_main)
        action_values = [self.evaluate_action(state, action) for action in actions]
        return actions[np.argmax(action_values)]

    def get_state_vector(self, scoreboard, is_main):
        state = []
        for color in Color:
            state.extend([1 if i in scoreboard[color] else 0 for i in range(2, 13)])
        state.extend([scoreboard.crosses, scoreboard.score, int(is_main)])
        return np.array(state)

    def evaluate_action(self, state, action):
        action_vector = self.get_action_vector(action)
        input_vector = np.concatenate([state, action_vector])
        return np.dot(input_vector, self.genome[:len(input_vector)])

    def get_action_vector(self, action):
        vector = np.zeros(8)  # 4 colors * 2 (color, number)
        for i, act in enumerate(action):
            vector[i*2] = list(Color).index(act.color)
            vector[i*2+1] = act.n
        return vector
# Usage
if __name__ == '__main__':


    pop_size = 50
    mutation_rate = 0.01
    crossover_rate = 0.7
    generations = 100
    games_per_individual = 10

    # ga = GeneticAlgorithm(population_size=pop_size, mutation_rate=mutation_rate, crossover_rate=crossover_rate)
    # best_agent = ga.evolve(generations=generations, games_per_individual=games_per_individual)

    best_agent = GA_Player('saved_agents/GA50_0.01_0.7_100_10.pkl')
    # Compete against RandomPlayer
    wins, draws, losses = compete(best_agent, best_agent, n_games=1000)
    print(f"Best agent vs RandomPlayer: Wins: {wins}, Draws: {draws}, Losses: {losses}")

    # Compete against ImprovedPlayer
    wins, draws, losses = compete(best_agent, ImprovedPlayer(), n_games=1000)
    print(f"Best agent vs ImprovedPlayer: Wins: {wins}, Draws: {draws}, Losses: {losses}")

   

    # Save the best agent's genome to a file
    with open(f'saved_agents/GA{pop_size}_{mutation_rate}_{crossover_rate}_{generations}_{games_per_individual}.pkl', 'wb') as f:
        pickle.dump(best_agent.genome, f)

    print("Best agent saved!")


    
