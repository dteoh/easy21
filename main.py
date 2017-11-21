import matplotlib.pyplot as plt
import numpy as np

from enum import Enum
from random import sample
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter


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


def epsilon_greedy_action(q_sa, s, epsilon):
    a_best = greedy_action(q_sa, s)
    selection_probs = []
    for a in list(Action):
        if a is a_best:
            selection_probs.append(1 - epsilon + epsilon / len(Action))
        else:
            selection_probs.append(epsilon / len(Action))
    return sample_action(selection_probs)


def calculate_epsilon(n_s, s, n0=100):
    return n0 / (n0 + n_s.get(s, 0))


def mc_control(num_episodes=10000):
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
    return q_sa


def sarsa_lambda(num_episodes=1000, lamba=0, gamma=1, yield_progress=False):
    q_sa = {}
    n_s = {}
    n_sa = {}

    for n in range(num_episodes):
        e_sa = {}
        state = State()
        s = state.as_tuple()
        a = epsilon_greedy_action(q_sa, s, calculate_epsilon(n_s, s))
        while not state.terminal:
            state, reward = step(state, a)
            n_s[s] = n_s.get(s, 0) + 1

            s_next = state.as_tuple()
            a_next = epsilon_greedy_action(q_sa, s_next, calculate_epsilon(n_s, s_next))

            sa = s + (a,)
            sa_next = s_next + (a_next,)
            qsa = q_sa.get(sa, 0)
            qsa_next = q_sa.get(sa_next, 0)

            nsa = n_sa.get(sa, 0) + 1
            n_sa[sa] = nsa

            delta = reward + gamma * qsa_next - qsa
            e_sa[sa] = e_sa.get(sa, 0) + 1
            for (s, a) in generate_all_state_action_pairs():
                sa = s + (a,)
                q_sa[sa] = q_sa.get(sa, 0) + (delta * e_sa.get(sa, 0)) / nsa
                e_sa[sa] = gamma * lamba * e_sa.get(sa, 0)

            s = s_next
            a = a_next

        if yield_progress:
            yield n+1, q_sa

    if not yield_progress:
        yield num_episodes, q_sa


def create_cuboid_fn(d_interval, p_interval, action):
    d_lower, d_upper = d_interval
    p_lower, p_upper = p_interval

    def cuboid(p, d, a):
        if a is not action:
            return False
        if d < d_lower:
            return False
        if d > d_upper:
            return False
        if p < p_lower:
            return False
        if p > p_upper:
            return False
        return True
    return cuboid


def lfa_sarsa_lambda(num_episodes=1000, lamba=0, gamma=1, alpha=0.01, yield_progress=False):

    # Set up the coarse codes, initial weights.
    action_codes = {}
    for action in list(Action):
        action_fns = []
        for dealer_interval in [(1,4), (4,7), (7,10)]:
            for player_interval in [(1,6), (4,9), (7,12), (10,15), (13,18), (16,21)]:
                cuboid_fn = create_cuboid_fn(dealer_interval, player_interval, action)
                action_fns.append(cuboid_fn)
        action_codes[action] = action_fns

    def greedy(s, w):
        p, d = s
        action_values = []
        for a in list(Action):
            value = 0
            for cuboid_fn in action_codes[a]:
                if cuboid_fn(p, d, a):
                    value += w.get(cuboid_fn, 0)
            action_values.append((a, value))
        action_values.sort(key=itemgetter(1), reverse=True)
        return action_values[0][0]

    def e_greedy(s, w, epsilon=0.05):
        a_best = greedy(s, w)
        selection_probs = []
        default_p = epsilon / len(Action)
        for a in list(Action):
            if a is a_best:
                selection_probs.append(1 - epsilon + default_p)
            else:
                selection_probs.append(default_p)
        return sample_action(selection_probs)

    def f_sa(s, a):
        p, d = s
        for cuboid_fn in action_codes[a]:
            if cuboid_fn(p, d, a):
                yield cuboid_fn

    def compile_q_sa(w):
        q_sa = {}
        for (p, d), a in generate_all_state_action_pairs():
            sa = (p, d, a)
            val = 0
            for i in f_sa((p, d), a):
                val += w.get(i, 0)
            q_sa[sa] = val
        return q_sa

    w_f = {}
    for n in range(num_episodes):
        state = State()
        s = state.as_tuple()
        a = e_greedy(s, w_f)
        z_f = {}
        while not state.terminal:
            state, reward = step(state, a)
            delta = reward
            for i in f_sa(s, a):
                delta = delta - w_f.get(i, 0)
                z_f[i] = z_f.get(i, 0) + 1
            if state.terminal:
                for i, zi in z_f.items():
                    w_f[i] = w_f.get(i, 0) + alpha * delta * zi
                break
            s_next = state.as_tuple()
            a_next = e_greedy(s_next, w_f)
            for i in f_sa(s_next, a_next):
                delta = delta + gamma * w_f.get(i, 0)
            for i, zi in z_f.items():
                w_f[i] = w_f.get(i, 0) + alpha * delta * zi
                z_f[i] = gamma * lamba * zi
            s = s_next
            a = a_next
        if yield_progress:
            yield n+1, compile_q_sa(w_f)

    if not yield_progress:
        yield num_episodes, compile_q_sa(w_f)


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


def part_2():
    q_sa = mc_control(1000000)
    plot_q_sa(q_sa)


def part_3():
    def mse(pred, base):
        sum = 0
        n = 0
        for (s, a) in generate_all_state_action_pairs():
            sa = s + (a,)
            error = pred.get(sa, 0) - base.get(sa, 0)
            sum += error**2
            n += 1
        return sum / n

    mse_xs = []
    mse_ys = []
    q_star_sa = mc_control(1000000)
    for lamba in range(0, 11):
        for (_, q_sa) in sarsa_lambda(num_episodes=1000, lamba=lamba / 10):
            mse_xs.append(lamba / 10)
            mse_ys.append(mse(q_sa, q_star_sa))

    mse_0_xs = []
    mse_0_ys = []
    for (n, q_sa) in sarsa_lambda(num_episodes=1000, lamba=0, yield_progress=True):
        mse_0_xs.append(n)
        mse_0_ys.append(mse(q_sa, q_star_sa))

    mse_1_xs = []
    mse_1_ys = []
    for (n, q_sa) in sarsa_lambda(num_episodes=1000, lamba=1, yield_progress=True):
        mse_1_xs.append(n)
        mse_1_ys.append(mse(q_sa, q_star_sa))

    # fig = plt.figure()
    fig, axarr = plt.subplots(1, 3)
    ax0 = axarr[0]
    ax0.plot(mse_xs, mse_ys)
    ax0.axis([0, 1, 0, 1])
    ax0.set_xlabel("lambda")
    ax0.set_ylabel("MSE")

    ax1 = axarr[1]
    ax1.plot(mse_0_xs, mse_0_ys)
    ax1.axis([0, 1000, 0, 1])
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("MSE")
    ax1.set_title("lambda=0")

    ax2 = axarr[2]
    ax2.plot(mse_1_xs, mse_1_ys)
    ax2.axis([0, 1000, 0, 1])
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("MSE")
    ax2.set_title("lambda=1")

    plt.show()


def part_4():
    def mse(pred, base):
        sum = 0
        n = 0
        for (s, a) in generate_all_state_action_pairs():
            sa = s + (a,)
            error = pred.get(sa, 0) - base.get(sa, 0)
            sum += error**2
            n += 1
        return sum / n

    mse_xs = []
    mse_ys = []
    q_star_sa = mc_control(1000000)
    for lamba in range(0, 11):
        for (_, q_sa) in lfa_sarsa_lambda(num_episodes=1000, lamba=lamba / 10):
            mse_xs.append(lamba / 10)
            mse_ys.append(mse(q_sa, q_star_sa))

    mse_0_xs = []
    mse_0_ys = []
    for (n, q_sa) in lfa_sarsa_lambda(num_episodes=1000, lamba=0, yield_progress=True):
        mse_0_xs.append(n)
        mse_0_ys.append(mse(q_sa, q_star_sa))

    mse_1_xs = []
    mse_1_ys = []
    for (n, q_sa) in lfa_sarsa_lambda(num_episodes=1000, lamba=1, yield_progress=True):
        mse_1_xs.append(n)
        mse_1_ys.append(mse(q_sa, q_star_sa))

    # fig = plt.figure()
    fig, axarr = plt.subplots(1, 3)
    ax0 = axarr[0]
    ax0.plot(mse_xs, mse_ys)
    ax0.axis([0, 1, 0, 1])
    ax0.set_xlabel("lambda")
    ax0.set_ylabel("MSE")

    ax1 = axarr[1]
    ax1.plot(mse_0_xs, mse_0_ys)
    ax1.axis([0, 1000, 0, 1])
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("MSE")
    ax1.set_title("lambda=0")

    ax2 = axarr[2]
    ax2.plot(mse_1_xs, mse_1_ys)
    ax2.axis([0, 1000, 0, 1])
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("MSE")
    ax2.set_title("lambda=1")

    plt.show()

def main():
    # part_2()
    # part_3()
    # part_4()
    pass


if __name__ == '__main__':
    main()
