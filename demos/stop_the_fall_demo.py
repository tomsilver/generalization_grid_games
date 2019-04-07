from generalization_grid_games.envs import stop_the_fall as stf
from generalization_grid_games.envs.utils import run_random_agent_demo


def run_interactive_demos():
    stf.StopTheFallGymEnv1(interactive=True)
    stf.StopTheFallGymEnv2(interactive=True)
    stf.StopTheFallGymEnv3(interactive=True)
    stf.StopTheFallGymEnv4(interactive=True)


if __name__ == "__main__":
    run_interactive_demos()
    # run_random_agent_demo(stf.StopTheFallGymEnv1)
