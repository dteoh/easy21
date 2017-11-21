from enum import Enum
from random import sample


class Action(Enum):
    STICK = 1
    HIT = 2

    @staticmethod
    def random():
        return sample(list(Action), 1)[0]


class State:
    def __init__(self):
        self.dealer_sum = self.generate_value()
        self.player_sum = self.generate_value()
        self.terminal = False

    def player_hits(self):
        self.player_sum += self.generate_card()
        if self.player_goes_bust():
            self.terminal = True
        return self

    def dealer_hits(self):
        self.dealer_sum += self.generate_card()
        if self.dealer_goes_bust():
            self.terminal = True
        return self

    def player_goes_bust(self):
        return self.goes_bust(self.player_sum)

    def dealer_goes_bust(self):
        return self.goes_bust(self.dealer_sum)

    def dealer_will_hit(self):
        return (not self.terminal) and self.dealer_sum < 17

    def reward(self):
        if self.dealer_goes_bust():
            return 1
        if self.player_sum > self.dealer_sum:
            return 1
        if self.player_sum < self.dealer_sum:
            return -1
        return 0

    def as_tuple(self):
        return self.player_sum, self.dealer_sum

    @staticmethod
    def goes_bust(sum):
        if sum > 21:
            return True
        elif sum < 1:
            return True
        else:
            return False

    @staticmethod
    def generate_value():
        return sample(range(1, 11), 1)[0]

    @staticmethod
    def generate_card():
        value = State.generate_value()
        if sample([1, 2, 2], 1)[0] == 1:
            return -value
        else:
            return value


def step(state: State, action: Action):
    if action is Action.HIT:
        state.player_hits()
    elif action is Action.STICK:
        while state.dealer_will_hit():
            state.dealer_hits()
        state.terminal = True
    if state.terminal:
        return state, state.reward()
    else:
        return state, 0


def generate_all_state_action_pairs():
    for player_sum in range(1, 22):
        for dealer_sum in range(1, 11):
            for action in list(Action):
                yield (player_sum, dealer_sum), action