from operator import itemgetter

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from environment import Action, generate_all_state_action_pairs


def sample_action(action_probs):
    return np.random.choice(list(Action), 1, p=action_probs)[0]


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


def greedy_action(q_sa, s):
    action_values = []
    for a in list(Action):
        val = q_sa.get(s + (a,), 0)
        action_values.append((a, val))
    action_values.sort(key=itemgetter(1), reverse=True)
    return action_values[0][0]


def mse(pred, base):
    sum = 0
    n = 0
    for (s, a) in generate_all_state_action_pairs():
        sa = s + (a,)
        error = pred.get(sa, 0) - base.get(sa, 0)
        sum += error**2
        n += 1
    return sum / n