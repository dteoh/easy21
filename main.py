import matplotlib.pyplot as plt
import numpy as np

from enum import Enum
from random import sample
from mpl_toolkits.mplot3d import Axes3D


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


def sample_action(action_probs):
    return np.random.choice(list(Action), 1, p=action_probs)[0]


def greedy_action(q_sa, s):
    v_stick = q_sa.get(s + (Action.STICK,), 0)
    v_hit = q_sa.get(s + (Action.HIT,), 0)
    if v_stick > v_hit:
        return Action.STICK
    elif v_hit > v_stick:
        return Action.HIT
    else:
        return Action.random()


def plot_q_sa(q_sa):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = list(range(1, 11))
    Y = list(range(1, 22))
    Z = []
    for dealer_card in range(1, 11):
        zs = []
        for player_sum in range(1, 22):
            s = (player_sum, dealer_card)
            a = greedy_action(q_sa, s)
            v = q_sa.get(s + (a,), 0)
            zs.append(v)
        Z.append(zs)

    X, Y = np.meshgrid(X, Y)
    Z = np.array(Z)

    ax.plot_surface(X, Y, Z.transpose(), color='b')
    ax.set_xlabel('Dealer Card')
    ax.set_ylabel('Player Sum')
    ax.set_zlabel('V*(s)')
    plt.show()


def mc_control(num_episodes=1000000):
    q_sa = {}
    p = {}
    n_s = {}
    n_sa = {}
    n0 = 100

    for _ in range(num_episodes):
        state = State()
        reward = 0
        episode_s = []
        episode_sa = []

        while not state.terminal:
            s = state.as_tuple()
            if s in p:
                a = sample_action(p[s])
            else:
                a = Action.random()

            episode_s.append(s)
            episode_sa.append(s + (a,))
            state, reward = step(state, a)

            ns = n_s.get(s, 0)
            n_s[s] = ns + 1

            sa = s + (a,)
            nsa = n_sa.get(sa, 0)
            n_sa[sa] = nsa + 1

        # GLIE MC Control
        for sa in set(episode_sa):
            nsa = n_sa[sa]
            qsa = q_sa.get(sa, 0)
            q_sa[sa] = qsa + ((reward - qsa) / nsa)

        # Improve policy
        for s in set(episode_s):
            a_best = greedy_action(q_sa, s)
            ns = n_s.get(s, 0)
            epsilon = n0 / (n0 + ns)

            selection_probs = []
            for a in list(Action):
                if a is a_best:
                    selection_probs.append(1 - epsilon + epsilon / len(Action))
                else:
                    selection_probs.append(epsilon / len(Action))
            p[s] = selection_probs

    plot_q_sa(q_sa)


def main():
    mc_control()


if __name__ == '__main__':
    main()
