from resource_curse import ResourceCurseEnv
from qagent import QAgent


def main():
    env = ResourceCurseEnv(reward=5, social_reward=2, alpha=1.5, political_reward=1)

    country = QAgent(env, (-2, 7.5), 30)
    world = QAgent(env, (-1, 5), 30)

    for i in range(1000):
        state = env.reset()
        done = False
        while not done:
            country_action = country.predict(state, i)
            world_action = world.predict(state, i)
            action = country_action, world_action
            next_state, reward, done, _ = env.step(action)

            if not done:
                country_reward, world_reward = reward
                country.update(state, next_state, country_action, country_reward)
                world.update(state, next_state, world_action, world_reward)

            state = next_state

        if not i % 100:
            print("-" * 16)
            print("country q-table")
            print(country.q_table)
            print("world q-table")
            print(world.q_table)
            print("-" * 16)


if __name__ == "__main__":
    main()
