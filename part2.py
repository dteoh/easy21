from environment import State, Action, step
from utils import sample_action, greedy_action, plot_q_sa


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


def main():
    q_sa = mc_control(1000000)
    plot_q_sa(q_sa)


if __name__ == '__main__':
    main()