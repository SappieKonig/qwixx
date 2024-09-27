import random
import numpy as np
from copy import deepcopy
from qwixx.player import Player
from qwixx.game import Color, Action, Scoreboard, Qwixx, compete
import pickle

from init_player import RandomPlayer
import torch
import torch.nn as nn
import torch.optim as optim
from GA import GeneticAlgorithmAgent, GeneticAlgorithm, GA_Player

class GeneticPPOAgent(Player):
    def __init__(self, state_size, action_size, learning_rate=0.001, epsilon=0.2):
        super().__init__()  # Initialize the base Player class
        self.state_size = state_size
        self.action_size = action_size
        # Define the policy network with input size equal to state_size + action_size
        self.policy_net = nn.Sequential(
            nn.Linear(state_size + action_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.old_policy_net = deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # Define the value network with input size equal to state_size
        self.value_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.epsilon = epsilon

    def move(self, actions, is_main, scoreboard, other_scoreboards):
        # Since 'game_progress' is not passed, we remove it from the arguments
        state = self.get_state_vector(scoreboard, is_main)
        action_probabilities = self.get_action_probabilities(state, actions)
        chosen_action_index = np.random.choice(len(actions), p=action_probabilities)
        return actions[chosen_action_index]

    def get_state_vector(self, scoreboard, is_main):
        state = []
        for color in Color:
            track = scoreboard[color]
            # For RED and YELLOW, positions from 2 to 12
            # For GREEN and BLUE, positions from 12 down to 2
            if color in [Color.RED, Color.YELLOW]:
                state.extend([1 if i in track else 0 for i in range(2, 13)])
            else:
                state.extend([1 if i in track else 0 for i in range(12, 1, -1)])
        state.extend([scoreboard.crosses, scoreboard.score, int(is_main)])
        return np.array(state)

    def get_action_probabilities(self, state, actions):
        action_vectors = [self.get_action_vector(action) for action in actions]
        input_vectors = [np.concatenate([state, action_vector]) for action_vector in action_vectors]
        inputs = torch.tensor(input_vectors, dtype=torch.float32)
        logits = self.policy_net(inputs).squeeze(-1)
        probs = torch.softmax(logits, dim=0).detach().numpy()
        return probs

    def get_action_vector(self, action):
        vector = np.zeros(self.action_size)
        if not action:  # Empty action
            return vector
        # First action
        first_action = action[0]
        vector[0:4] = self.one_hot_encode_color(first_action.color)
        vector[4] = (first_action.n - 2) / 10  # Normalize number between 0 and 1
        if len(action) > 1:
            # Second action
            second_action = action[1]
            vector[5:9] = self.one_hot_encode_color(second_action.color)
            vector[9] = (second_action.n - 2) / 10  # Normalize
        return vector

    def one_hot_encode_color(self, color):
        color_index = list(Color).index(color)
        one_hot = np.zeros(4)
        one_hot[color_index] = 1
        return one_hot

    def update(self, states, actions, actions_lists, advantages, returns):
        for _ in range(5):  # Multiple optimization steps
            for state, action, actions_list, advantage, return_ in zip(states, actions, actions_lists, advantages, returns):
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action_vector = torch.tensor(self.get_action_vector(action), dtype=torch.float32)
                input_vector = torch.cat([state_tensor, action_vector])

                # Compute probabilities
                old_prob = self.get_action_probability(state, action, actions_list, use_old=True)
                new_prob = self.get_action_probability(state, action, actions_list)
                ratio = torch.tensor(new_prob / (old_prob + 1e-8))
                clipped_ratio = torch.clamp(ratio, torch.tensor(1 - self.epsilon),torch.tensor(1 + self.epsilon))
                policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

                # Compute value loss
                value = self.value_net(state_tensor)
                value_loss = 0.5 * (return_ - value).pow(2)

                # Total loss
                total_loss = policy_loss + value_loss

                # Backpropagation
                self.optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.value_optimizer.step()

        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

    def get_action_probability(self, state, action, actions_list, use_old=False):
        policy_net = self.old_policy_net if use_old else self.policy_net
        action_vectors = [self.get_action_vector(a) for a in actions_list]
        input_vectors = [np.concatenate([state, av]) for av in action_vectors]
        inputs = torch.tensor(input_vectors, dtype=torch.float32)
        logits = policy_net(inputs).squeeze(-1)
        probs = torch.softmax(logits, dim=0).detach().numpy()
        try:
            action_index = actions_list.index(action)
        except ValueError:
            return 1e-8
        return probs[action_index]


class GeneticPPOAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.state_size = 47  # Adjusted according to get_state_vector
        self.action_size = 10  # Adjusted according to get_action_vector
        self.population = [GeneticPPOAgent(self.state_size, self.action_size) for _ in range(population_size)]

    def evolve(self, generations=100, games_per_individual=10):
        for generation in range(generations):
            fitnesses, experiences = self.evaluate_population(games_per_individual)
            self.update_population(experiences)
            new_population = []

            # Elitism: keep the best individual
            elite_index = np.argmax(fitnesses)
            elite = self.population[elite_index]
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
            wins, draws, losses = compete(individual, GA_Player(pickle_file="antics/saved_agents/GA100_0.01_0.7_100_10.pkl"), games_per_individual)
            fitnesses.append(wins + 0.5 * draws)
            all_experiences.append(self.collect_experiences(individual, games_per_individual))
        return fitnesses, all_experiences
    
    def calculate_game_progress(self, game):
        total_numbers_per_color = 11  # Numbers from 2 to 12
        total_possible_numbers = total_numbers_per_color * len(Color)
        agent_scoreboard = game.scoreboards[0]
        numbers_crossed = sum(len(agent_scoreboard[color]) for color in Color)
        progress = numbers_crossed / total_possible_numbers
        return progress

    def feature_number_between(self, game):
        reward = 0
        if len(game.scoreboards[0].red) > 1:
            reward -= game.scoreboards[0].red[-1] - game.scoreboards[0].red[-2]
        if len(game.scoreboards[0].blue) > 1:
            reward -= game.scoreboards[0].blue[-1] - game.scoreboards[0].blue[-2]    
        if len(game.scoreboards[0].green) > 1:
            reward -= game.scoreboards[0].green[-1] - game.scoreboards[0].green[-2]
        if len(game.scoreboards[0].yellow) > 1:
            reward -= game.scoreboards[0].yellow[-1] - game.scoreboards[0].yellow[-2]

        return reward
    
    def feature_first_choices(self, game):
        reward = 0
        if len(game.scoreboards[0].red) < 2:
            reward -=np.sum(game.scoreboards[0].red)
        if len(game.scoreboards[0].yellow) <2:
            reward -=np.sum(game.scoreboards[0].yellow)

        if len(game.scoreboards[0].green) < 2:
            reward +=np.sum(game.scoreboards[0].green)
        if len(game.scoreboards[0].blue) <2:
            reward +=np.sum(game.scoreboards[0].blue)
        return reward
    def decrease_mutations(self, game):
        if self.mutation_rate < 0.01:
            self.mutation_rate = 0.01
        else:
            self.mutation_rate = self.mutation_rate* np.exp(-0.1*game.round)
        
    # def feature_kill_colour(self, game):
    #     reward = 0
    #     if game.scoreboards[0].score>game.scoreboards[1].score:

            

    def collect_experiences(self, agent, num_games):
        experiences = []
        for _ in range(num_games):
            game = Qwixx([agent, GA_Player(pickle_file="antics/saved_agents/GA100_0.01_0.7_100_10.pkl")])
            game_experiences = []
            initial_score = game.scoreboards[0].score
            while not game.is_finished():
                old_score = game.scoreboards[0].score
                state = agent.get_state_vector(game.scoreboards[0], game.turn == 0)
                is_main = (game.turn == 0)
                player_index = game.turn
                scoreboard = game.scoreboards[player_index]
                other_scoreboards = game.scoreboards[:player_index] + game.scoreboards[player_index+1:]

                possible_actions = game.get_possible_actions(is_main, player_index)
                if not possible_actions:
                    possible_actions = [[]]  # No legal actions available

                # Agent selects an action
                chosen_action = agent.move(possible_actions, is_main, scoreboard, other_scoreboards)

                # Validate chosen action
                if chosen_action not in possible_actions:
                    chosen_action = []
            
                # Apply the chosen action
                game.apply_action(player_index, chosen_action)

                # Compute reward
                progress = self.calculate_game_progress(game)
                
                
                colour_reward = self.feature_first_choices(game)
                feature_first = self.feature_first_choices(game)
                new_score = game.scoreboards[0].score
                reward = new_score - old_score+colour_reward+feature_first

                # Get the next state and done flag
                next_state = agent.get_state_vector(game.scoreboards[0], game.turn == 0)
                done = game.is_finished()
                self.decrease_mutations(game)
                # Store experience
                game_experiences.append((state, chosen_action, reward, next_state, done, possible_actions))

            experiences.extend(game_experiences)

        return experiences

    def update_population(self, all_experiences):
        for agent, experiences in zip(self.population, all_experiences):
            if len(experiences) == 0:
                continue  # Skip agents with no experiences
            states, actions, rewards, next_states, dones, actions_lists = zip(*experiences)
            states = np.array(states)
            next_states = np.array(next_states)
            rewards = np.array(rewards)
            dones = np.array(dones)

            # Compute values and next values
            with torch.no_grad():
                values = agent.value_net(torch.tensor(states, dtype=torch.float32)).squeeze(-1).numpy()
                next_values = agent.value_net(torch.tensor(next_states, dtype=torch.float32)).squeeze(-1).numpy()

            advantages = self.compute_advantages(rewards, values, next_values, dones)
            returns = advantages + values

            agent.update(states, actions, actions_lists, advantages, returns)

    def compute_advantages(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_non_terminal = 0.0
                delta = rewards[t] - values[t]
            else:
                next_non_terminal = 1.0
                delta = rewards[t] + gamma * next_values[t] - values[t]
            advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages

    def selection(self, fitnesses):
        total_fitness = sum(fitnesses)
        selection_probs = [f / total_fitness for f in fitnesses]
        selected_indices = np.random.choice(len(self.population), 2, p=selection_probs, replace=False)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]

    def crossover(self, parent1, parent2):
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        if random.random() < self.crossover_rate:
            for param1, param2 in zip(child1.policy_net.parameters(), child2.policy_net.parameters()):
                # Single-point crossover
                num_elements = param1.data.nelement()
                if num_elements == 0:
                    continue
                crossover_point = random.randint(0, num_elements - 1)
                flat_param1 = param1.data.view(-1)
                flat_param2 = param2.data.view(-1)
                temp = flat_param1[crossover_point:].clone()
                flat_param1[crossover_point:] = flat_param2[crossover_point:]
                flat_param2[crossover_point:] = temp

            # Similarly for value network
            for param1, param2 in zip(child1.value_net.parameters(), child2.value_net.parameters()):
                num_elements = param1.data.nelement()
                if num_elements == 0:
                    continue
                crossover_point = random.randint(0, num_elements - 1)
                flat_param1 = param1.data.view(-1)
                flat_param2 = param2.data.view(-1)
                temp = flat_param1[crossover_point:].clone()
                flat_param1[crossover_point:] = flat_param2[crossover_point:]
                flat_param2[crossover_point:] = temp

        return child1, child2

    def mutate(self, individual):
        for param in individual.policy_net.parameters():
            mutation_mask = torch.rand(param.data.size()) < self.mutation_rate
            param.data += mutation_mask.float() * torch.randn(param.data.size()) * 0.1

        for param in individual.value_net.parameters():
            mutation_mask = torch.rand(param.data.size()) < self.mutation_rate
            param.data += mutation_mask.float() * torch.randn(param.data.size()) * 0.1


if __name__ == '__main__':

    pop_size = 50
    mutation_rate = 0.1
    crossover_rate = 0.7
    generations = 30
    games_per_individual = 10

    algorithm = GeneticPPOAlgorithm(
        population_size=pop_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )

    best_agent = algorithm.evolve(generations=generations, games_per_individual=games_per_individual)

    # Compete against RandomPlayer
    wins, draws, losses = compete(best_agent, RandomPlayer(), n_games=1000)
    print(f"Best agent vs RandomPlayer: Wins: {wins}, Draws: {draws}, Losses: {losses}")

    # Save the best agent's policy and value networks
    file_name = f'saved_agents/GA_PPO_{pop_size}_{mutation_rate}_{crossover_rate}_{generations}_{games_per_individual}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump({
            'policy_net': best_agent.policy_net.state_dict(),
            'value_net': best_agent.value_net.state_dict()
        }, f)

    print(f"Best agent saved to {file_name}")
