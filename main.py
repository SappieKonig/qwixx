from copy import deepcopy
from random import choice
import numpy as np
import random
import pickle

from qwixx.game import Color, compete
from qwixx.player import Player
from GA import GeneticAlgorithmAgent, GeneticAlgorithm, GA_Player

from init_player import RandomPlayer



# Usage
if __name__ == '__main__':


    pop_size = 100
    mutation_rate = 0.01
    crossover_rate = 0.7
    generations = 100
    games_per_individual = 10

    ga = GeneticAlgorithm(population_size=pop_size, mutation_rate=mutation_rate, crossover_rate=crossover_rate)
    best_agent = ga.evolve(generations=generations, games_per_individual=games_per_individual)

    best_agent = GA_Player('saved_agents/GA50_0.01_0.7_100_10.pkl')
    # Compete against RandomPlayer
    wins, draws, losses = compete(best_agent, RandomPlayer(), n_games=1000)
    print(f"Best agent vs RandomPlayer: Wins: {wins}, Draws: {draws}, Losses: {losses}")


    file_name = f'saved_agents/GA{pop_size}_{mutation_rate}_{crossover_rate}_{generations}_{games_per_individual}.pkl'
    # Save the best agent's genome to a file
    with open(file_name, 'wb') as f:
        pickle.dump(best_agent.genome, f)

    print(f"Best agent saved!{file_name}")


    
