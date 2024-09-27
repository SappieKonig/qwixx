import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from qwixx import Player, Qwixx
from copy import deepcopy
import random


class PPONetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(11 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, 1)  # Output a single value for each action
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)


class PPOPlayer(Player):
    def __init__(self):
        self.net = PPONetwork()
        self.optimizer = optim.Adam(self.net.parameters(), lr=3e-4)

    def move(self, actions, is_main, scoreboard, scoreboards):
        states = [self.scoreboard_to_tensor(deepcopy(scoreboard).move(action)) for action in actions]
        states = torch.stack(states)

        with torch.no_grad():
            action_probs, _ = self.net(states)
            action_probs = F.softmax(action_probs, dim=0)
            dist = Categorical(action_probs)
            action = dist.sample()

        return actions[action.item()]

    @staticmethod
    def scoreboard_to_tensor(scoreboard):
        red = yellow = green = blue = torch.zeros(11)

        def set(t, li):
            for el in li:
                t[el - 2] = 1

        set(red, scoreboard.red)
        set(yellow, scoreboard.yellow)
        set(green, scoreboard.green)
        set(blue, scoreboard.blue)
        return torch.cat([red, yellow, green, blue])


def collect_trajectories(player, n_games=100):
    trajectories = []
    for _ in range(n_games):
        game = Qwixx([player, player])
        trajectory = []
        while not game.is_finished():
            state = player.scoreboard_to_tensor(game.scoreboards[game.current_player])
            actions = game.available_actions()
            action = player.move(actions, game.current_player == 0, game.scoreboards[game.current_player],
                                 game.scoreboards)
            action_idx = actions.index(action)
            game.move()
            reward = game.scoreboards[game.current_player].score - game.scoreboards[1 - game.current_player].score
            trajectory.append((state, action_idx, reward))
        trajectories.append(trajectory)
    return trajectories


def compute_advantages(trajectories, gamma=0.99, lambda_=0.95):
    advantages = []
    for trajectory in trajectories:
        values = []
        with torch.no_grad():
            for state, _, _ in trajectory:
                _, value = player.net(state.unsqueeze(0))
                values.append(value.item())

        advantages_traj = []
        gae = 0
        for t in reversed(range(len(trajectory))):
            if t == len(trajectory) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = trajectory[t][2] + gamma * next_value - values[t]
            gae = delta + gamma * lambda_ * gae
            advantages_traj.insert(0, gae)
        advantages.extend(advantages_traj)
    return torch.tensor(advantages)


def update_ppo(player, trajectories, epochs=4, epsilon=0.2):
    states = torch.cat([state for trajectory in trajectories for state, _, _ in trajectory])
    actions = torch.tensor([action for trajectory in trajectories for _, action, _ in trajectory])
    rewards = torch.tensor([reward for trajectory in trajectories for _, _, reward in trajectory])

    advantages = compute_advantages(trajectories)
    returns = advantages + rewards

    for _ in range(epochs):
        new_probs, state_values = player.net(states)
        new_probs = F.softmax(new_probs, dim=1)
        new_dist = Categorical(new_probs)

        old_probs, _ = player.net(states)
        old_probs = F.softmax(old_probs, dim=1).detach()
        old_dist = Categorical(old_probs)

        ratio = torch.exp(new_dist.log_prob(actions) - old_dist.log_prob(actions))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(state_values.squeeze(), returns)
        loss = actor_loss + 0.5 * critic_loss

        player.optimizer.zero_grad()
        loss.backward()
        player.optimizer.step()


if __name__ == '__main__':
    UPDATES = 100
    GAMES_PER_UPDATE = 100

    player = PPOPlayer()

    for update in range(UPDATES):
        trajectories = collect_trajectories(player, n_games=GAMES_PER_UPDATE)
        update_ppo(player, trajectories)

        # Evaluate performance
        if update % 10 == 0:
            wins = 0
            for _ in range(100):
                game = Qwixx([player, RandomPlayer()])
                while not game.is_finished():
                    game.move()
                if game.scoreboards[0].score > game.scoreboards[1].score:
                    wins += 1
            print(f"Update {update}, Win rate against RandomPlayer: {wins}%")