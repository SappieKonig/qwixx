from main import RandomPlayer
from qwixx import Player, compete, Qwixx
from qwixx.game import Color
import torch.nn as nn
import torch
import torch.nn.functional as F
from copy import deepcopy


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(10, 32)
        self.net = nn.Sequential(
            nn.Linear(32, 32),
            nn.Linear(32, 32),
        )
        self.pred_proj = nn.Linear(32, 1)

    def forward(self, x):
        x = self.down_proj(x)
        for l in self.net:
            x = F.relu(l(x) + x)
        return self.pred_proj(x)

    def update(self, noise_scale):
        new_network = deepcopy(self)

        # Add small random noise to all parameters\
        with torch.no_grad():
            for param in new_network.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)

        return new_network


class GeneticPlayer(Player):
    def __init__(self, net=None):
        if net is None:
            self.net = Network()
        else:
            self.net = net

    @torch.no_grad()
    def move(self, actions, is_main, scoreboard, scoreboards):
        inp = []
        saved = deepcopy(scoreboard)
        for action in actions:
            sb = deepcopy(saved)
            for act in action:
                sb.move(act)
            if len(action) == 0:
                sb.crosses += 1
            inp.append(self.scoreboard_to_tensor(sb))
        out = self.net(torch.stack(inp)).squeeze(dim=1)
        out = out.tolist()
        move_idx = max(range(len(out)), key=lambda x: out[x])
        return actions[move_idx]

    @staticmethod
    def scoreboard_to_tensor(scoreboard):
        inp = torch.zeros(10)
        sb = scoreboard
        inp[0] = sb.crosses / 4
        inp[1] = sb.score / 100
        inp[2] = len(sb.red) / 10
        inp[3] = max(sb.red, default=1) / 12
        inp[4] = len(sb.yellow) / 10
        inp[5] = max(sb.yellow, default=1) / 12
        inp[6] = len(sb.green) / 10
        inp[7] = max(sb.green, default=13) / 12
        inp[8] = len(sb.blue) / 10
        inp[9] = min(sb.blue, default=13) / 12
        return inp

    def evolve(self, noise_scale):
        player = GeneticPlayer(self.net.update(noise_scale))
        return player


if __name__ == '__main__':
    UPDATES = 10000
    N_GAMES = 100
    NOISE_POWER = 0.997
    GAMES_POWER = 1.01
    if False:
        dic = torch.load('genetic8.pt')
        player1 = GeneticPlayer(dic['model'])
        player1.temperature = 0.
        step = dic['step']
        noise_scale = 0.1 * NOISE_POWER ** step
        N_GAMES = int(N_GAMES * GAMES_POWER ** step)
    else:
        player1 = GeneticPlayer()
        noise_scale = 0.1
        step = 0

    for update in range(step, UPDATES):
        print(compete(player1, RandomPlayer(), n_games=min(N_GAMES, 1_000)))
        print(f'Noise scale: {noise_scale:.4f}')
        player2 = player1.evolve(noise_scale)
        res = compete(player1, player2, n_games=N_GAMES)
        if res[2] > res[0]:
            player1 = player2
            torch.save({'model': player1.net, 'step': update}, 'genetic8.pt')

        noise_scale *= NOISE_POWER
        N_GAMES = int(GAMES_POWER * N_GAMES)
