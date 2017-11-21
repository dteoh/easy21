from matplotlib import pyplot as plt

from environment import State, step, generate_all_state_action_pairs, Action
from part2 import mc_control
from utils import sample_action, greedy_action, mse


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


def calculate_epsilon(n_s, s, n0=100):
    return n0 / (n0 + n_s.get(s, 0))


def epsilon_greedy_action(q_sa, s, epsilon):
    a_best = greedy_action(q_sa, s)
    selection_probs = []
    for a in list(Action):
        if a is a_best:
            selection_probs.append(1 - epsilon + epsilon / len(Action))
        else:
            selection_probs.append(epsilon / len(Action))
    return sample_action(selection_probs)


def main():
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


if __name__ == '__main__':
    main()