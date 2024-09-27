from qwixx import Player
import torch.nn as nn
import torch
from qwixx.game import Color


class MLPPlayer(Player):
    def __init__(self, saved_model=None, input_size=106, hidden_size=64, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        if saved_model:
            self.model.load_state_dict(torch.load(saved_model, map_location=torch.device('cuda'), weights_only=True))

    def move(self, actions, is_main, scoreboard, scoreboards):
        if not actions:
            return []

        input_tensors = torch.stack(
            [torch.tensor(self.extract_features(scoreboard, scoreboards, action, is_main), dtype=torch.float32) for
             action in actions])
        with torch.no_grad():
            values = self.model(input_tensors).squeeze()

        best_action_index = values.argmax().item()
        return actions[best_action_index]

    def extract_features(self, scoreboard, scoreboards, action, is_main):
        features = []

        # Player's scoreboard features
        for color in Color:
            features.extend(self.encode_color(scoreboard[color]))
            features.append(scoreboard.crossed[color])
            features.append(len(scoreboard[color]) / 13)  # Completion percentage
            features.append(self.calculate_color_potential(scoreboard[color], color))

        # Action encoding
        action_encoding = [0] * 4
        if action:
            action_encoding[list(Color).index(action[0].color)] = 1
            features.extend(action_encoding)
            features.append(action[0].n / 13)
            features.append(self.calculate_action_value(scoreboard, action[0]))
        else:
            features.extend(action_encoding)
            features.append(0)
            features.append(0)

        # Opponents' scoreboard features
        for other_scoreboard in scoreboards:
            for color in Color:
                features.append(len(other_scoreboard[color]) / 13)
                features.append(other_scoreboard.crossed[color])
                features.append(self.calculate_color_potential(other_scoreboard[color], color))

        # Game state features
        features.append(is_main)
        features.append(scoreboard.crosses / 4)
        features.append(sum(scoreboard.crossed.values()) / 4)  # Proportion of colors crossed
        features.append(self.calculate_score_difference(scoreboard, scoreboards))

        # Add some polynomial features
        features.extend(self.create_polynomial_features(features[:20], degree=2))

        return features

    @staticmethod
    def encode_color(color_scores):
        encoding = [0] * 13
        for score in color_scores:
            encoding[score - 2] = 1
        return encoding

    @staticmethod
    def calculate_color_potential(color_scores, color):
        if color in [Color.RED, Color.YELLOW]:
            return sum(1 for i in range(2, 13) if i not in color_scores and i > max(color_scores, default=1))
        else:
            return sum(1 for i in range(2, 13) if i not in color_scores and i < min(color_scores, default=13))

    @staticmethod
    def calculate_action_value(scoreboard, action):
        color_scores = scoreboard[action.color]
        if action.color in [Color.RED, Color.YELLOW]:
            return (action.n - max(color_scores, default=1)) / 13
        else:
            return (min(color_scores, default=13) - action.n) / 13

    @staticmethod
    def calculate_score_difference(scoreboard, scoreboards):
        player_score = scoreboard.score
        avg_opponent_score = sum(sb.score for sb in scoreboards) / len(scoreboards)
        return (player_score - avg_opponent_score) / 100  # Normalize the difference

    @staticmethod
    def create_polynomial_features(features, degree=2):
        poly_features = []
        for i in range(len(features)):
            for j in range(i, len(features)):
                poly_features.append(features[i] * features[j])
        return poly_features[:20]