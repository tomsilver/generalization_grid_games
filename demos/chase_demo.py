from generalization_grid_games.envs import chase as c
from generalization_grid_games.envs.utils import run_random_agent_demo



def run_interactive_demos():
    c.ChaseGymEnv1(interactive=True)
    c.ChaseGymEnv2(interactive=True)
    c.ChaseGymEnv3(interactive=True)
    c.ChaseGymEnv4(interactive=True)
    c.ChaseGymEnv5(interactive=True)
    c.ChaseGymEnv6(interactive=True)
    c.ChaseGymEnv7(interactive=True)
    c.ChaseGymEnv8(interactive=True)


if __name__ == "__main__":
    run_interactive_demos()
    # run_random_agent_demo(c.ChaseGymEnv1)
