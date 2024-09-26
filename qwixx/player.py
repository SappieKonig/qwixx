from abc import ABCMeta, abstractmethod


class Player(metaclass=ABCMeta):
    @abstractmethod
    def move(self, actions: list[list['Action']], is_main: bool, scoreboard: 'Scoreboard', scoreboards: list['Scoreboard']) -> list['Action']:
        pass