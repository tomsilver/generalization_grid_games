from generalization_grid_games.envs import reach_for_the_star as rfts
from generalization_grid_games.envs.utils import run_random_agent_demo


def run_interactive_demos():
    rfts.ReachForTheStarGymEnv1(interactive=True)
    rfts.ReachForTheStarGymEnv2(interactive=True)
    rfts.ReachForTheStarGymEnv3(interactive=True)
    rfts.ReachForTheStarGymEnv4(interactive=True)
    rfts.ReachForTheStarGymEnv5(interactive=True)
    rfts.ReachForTheStarGymEnv6(interactive=True)


if __name__ == "__main__":
    run_interactive_demos()
    # run_random_agent_demo(rfts.ReachForTheStarGymEnv1)
