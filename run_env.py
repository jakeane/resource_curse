import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from resource_curse import ResourceCurseEnv
from qagent import QAgent


def main():
    "play with environment"

    run_env({"discount": 0.6}, {})


def main_vis():
    "generate visualizations based on tweaking parameters"

    # Alpha
    q_diffs = list()
    for i in range(20):
        q_table = run_env({}, {"alpha": 1.2 + (i / 10)})
        q_diffs.append(q_table.T[0] - q_table.T[1])

    q_array = np.array(q_diffs)
    np.save("./qagent_alpha.npy", q_array)
    make_plot(q_array, "Alpha", 1.2, 3.2)

    # Discount
    q_diffs = list()
    for i in range(20):
        q_table = run_env({"discount": 0.45 + (i / 40)}, {})
        q_diffs.append(q_table.T[0] - q_table.T[1])

    q_array = np.array(q_diffs)
    np.save("./qagent_discount.npy", q_array)
    make_plot(q_array, "Discount", 0.45, 0.95)

    # Transition coefficient
    q_diffs = list()
    for i in range(20):
        q_table = run_env({}, {"tran_coef": 0.01 + (i / 100)})
        q_diffs.append(q_table.T[0] - q_table.T[1])

    q_array = np.array(q_diffs)
    np.save("./qagent_transition.npy", q_array)
    make_plot(q_array, "Transition", 0.01, 0.21)


def make_plot(a, ylabel, ylow, yhigh):
    "make color based visualization of `a`"

    plt.imshow(a, interpolation="nearest", origin="lower")
    cbar = plt.colorbar()
    cbar.set_label("Q(Diversify - Single Sector)", rotation=270, labelpad=20)

    # manual set x ticks and labels
    plt.xticks(
        np.arange(0, 20, step=4),
        map(lambda x: str(round(x, 1)), np.arange(0, 1, step=0.2)),
    )
    plt.xlabel("State")

    # manual set y ticks and labels
    ystep = (yhigh - ylow) / 5
    plt.yticks(
        np.arange(0, 20, step=4),
        map(lambda x: str(round(x, 2)), np.arange(ylow, yhigh, step=ystep)),
    )
    plt.ylabel(ylabel)

    plt.title(f"Q Agent Decision Making w.r.t. State and {ylabel}")
    plt.savefig(f"./qagent_{ylabel.lower()}.png")
    plt.close()


def run_env(agent_kwargs, env_kwargs):

    # init environment
    env = ResourceCurseEnv(**env_kwargs)
    country = QAgent(env, (-2, 7.5), 20, **agent_kwargs)
    world = QAgent(env, (-1, 5), 20)

    # train agents for x iterations
    for i in tqdm(range(2000)):

        # reset environment
        state = env.reset()
        done = False

        # run environment to completion
        while not done:

            # get agent actions based on environment
            country_action = country.predict(state, i)
            world_action = world.predict(state, i)
            action = country_action, world_action

            # get next state and rewards based on actions
            next_state, reward, done, _ = env.step(action)

            # update q-values
            if not done:
                country_reward, world_reward = reward
                country.update(state, next_state, country_action, country_reward)
                world.update(state, next_state, world_action, world_reward)

            state = next_state

    return country.q_table


if __name__ == "__main__":
    main()
