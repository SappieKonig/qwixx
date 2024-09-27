import random
import numpy as np
from copy import deepcopy
from qwixx.player import Player
from qwixx.game import Color, Action, Scoreboard, Qwixx, compete

class GeneticPPOAgent(Player):
    def __init__(self, genome_size=200, learning_rate=0.01, epsilon=0.2):
        self.genome_size = genome_size
        self.genome = np.random.rand(genome_size)
        self.old_genome = self.genome.copy()
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def move(self, actions, is_main, scoreboard, scoreboards):
        state = self.get_state_vector(scoreboard, is_main)
        action_probabilities = self.get_action_probabilities(state, actions)
        return actions[np.argmax(action_probabilities)]

    def get_state_vector(self, scoreboard, is_main):
        state = []
        for color in Color:
            state.extend([1 if i in scoreboard[color] else 0 for i in range(2, 13)])
        state.extend([scoreboard.crosses, scoreboard.score, int(is_main)])
        return np.array(state)

    def get_action_probabilities(self, state, actions):
        action_vectors = [self.get_action_vector(action) for action in actions]
        input_vectors = [np.concatenate([state, action_vector]) for action_vector in action_vectors]
        logits = [np.dot(input_vector, self.genome[:len(input_vector)]) for input_vector in input_vectors]
        return self.softmax(logits)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_action_vector(self, action):
        vector = np.zeros(8)  # 4 colors * 2 (color, number)
        for i, act in enumerate(action):
            vector[i*2] = list(Color).index(act.color)
            vector[i*2+1] = act.n
        return vector

    def update(self, states, actions, actions_lists, advantages):
        for _ in range(5):  # Multiple optimization steps
            for state, action, actions_list, advantage in zip(states, actions, actions_lists, advantages):
                old_prob = self.get_action_probability(state, action, actions_list, use_old=True)
                new_prob = self.get_action_probability(state, action, actions_list)
                ratio = new_prob / (old_prob + 1e-8)
                clipped_ratio = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
                loss = -min(ratio * advantage, clipped_ratio * advantage)
                gradient = self.get_gradient(state, action)
                self.genome[:len(gradient)] -= self.learning_rate * loss * gradient
        self.old_genome = self.genome.copy()


    def get_action_probability(self, state, action, actions_list, use_old=False):
        genome = self.old_genome if use_old else self.genome
        action_vectors = [self.get_action_vector(a) for a in actions_list]
        input_vectors = [np.concatenate([state, av]) for av in action_vectors]
        logits = [np.dot(iv, genome[:len(iv)]) for iv in input_vectors]
        probs = self.softmax(logits)
        # Find index of the action
        try:
            action_index = actions_list.index(action)
        except ValueError:
            # Action not found; return a small probability
            return 1e-8
        return probs[action_index]
    
    def get_gradient(self, state, action):
        action_vector = self.get_action_vector(action)
        input_vector = np.concatenate([state, action_vector])
        return input_vector 

class GeneticPPOAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.01, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = [GeneticPPOAgent() for _ in range(population_size)]

    def evolve(self, generations=100, games_per_individual=10):
        for generation in range(generations):
            fitnesses, experiences = self.evaluate_population(games_per_individual)
            self.update_population(experiences)
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
        all_experiences = []
        for individual in self.population:
            wins, draws, losses = compete(individual, RandomPlayer(), games_per_individual)
            fitnesses.append(wins + 0.5 * draws)
            all_experiences.append(self.collect_experiences(individual, games_per_individual))
        return fitnesses, all_experiences

    def collect_experiences(self, agent, num_games):
        experiences = []
        for _ in range(num_games):
            game = Qwixx([agent, RandomPlayer()])
            game_experiences = []
            initial_score = game.scoreboards[0].score
            while not game.is_finished():
                old_score = game.scoreboards[0].score
                state = agent.get_state_vector(game.scoreboards[0], game.turn == 0)
                is_main = (game.turn == 0)
                player_index = game.turn

                # Get possible actions using the new method
                possible_actions = game.get_possible_actions(is_main, player_index)
                if not possible_actions:
                    possible_actions = [[]]  # No legal actions available
                
                actions_list = possible_actions.copy()
                # Agent selects an action
                chosen_action = agent.move(possible_actions, is_main, game.scoreboards[player_index], game.scoreboards[:player_index] + game.scoreboards[player_index+1:])

                # Validate chosen action
                if chosen_action not in possible_actions:
                    chosen_action = []

                # Apply the chosen action
                game.apply_action(player_index, chosen_action)

                new_score = game.scoreboards[0].score
                reward = new_score - old_score
                game_experiences.append((state, chosen_action, reward, actions_list))

            # Calculate final reward based on win/loss
            final_reward = game.scoreboards[0].score - initial_score
            if game.scoreboards[0].score > game.scoreboards[1].score:
                final_reward += 10  # Bonus for winning
            experiences.extend([(state, action, reward + final_reward, actions_list) for state, action, reward, actions_list in game_experiences])

        return experiences

    def update_population(self, all_experiences):
        for agent, experiences in zip(self.population, all_experiences):
            states, actions, rewards, actions_lists = zip(*experiences)
            advantages = self.compute_advantages(rewards)
            agent.update(states, actions, actions_lists, advantages)

    def compute_advantages(self, rewards, gamma=0.99):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        discounted_sum = 0
        for t in reversed(range(len(rewards))):
            discounted_sum = rewards[t] + gamma * discounted_sum
            advantages[t] = discounted_sum
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages


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

        child1 = GeneticPPOAgent()
        child2 = GeneticPPOAgent()
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

# Usage
if __name__ == '__main__':
    ga_ppo = GeneticPPOAlgorithm(population_size=50, mutation_rate=0.01, crossover_rate=0.7)
    best_agent = ga_ppo.evolve(generations=100, games_per_individual=10)

    # Compete against RandomPlayer
    wins, draws, losses = compete(best_agent, RandomPlayer(), n_games=1000)
    print(f"Best agent vs RandomPlayer: Wins: {wins}, Draws: {draws}, Losses: {losses}")

    # Compete against ImprovedPlayer
    from main import ImprovedPlayer
    wins, draws, losses = compete(best_agent, ImprovedPlayer(), n_games=1000)
    print(f"Best agent vs ImprovedPlayer: Wins: {wins}, Draws: {draws}, Losses: {losses}")