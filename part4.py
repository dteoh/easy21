from operator import itemgetter

from matplotlib import pyplot as plt

from environment import Action, generate_all_state_action_pairs, State, step
from part2 import mc_control
from utils import sample_action, mse


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


def main():
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