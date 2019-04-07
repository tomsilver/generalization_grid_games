from generalization_grid_games.envs import checkmate_tactic as c
from generalization_grid_games.envs.utils import run_random_agent_demo



def run_interactive_demos():
    c.CheckmateTacticGymEnv1(interactive=True)
    c.CheckmateTacticGymEnv2(interactive=True)
    c.CheckmateTacticGymEnv3(interactive=True)
    c.CheckmateTacticGymEnv4(interactive=True)
    c.CheckmateTacticGymEnv5(interactive=True)
    c.CheckmateTacticGymEnv6(interactive=True)
    c.CheckmateTacticGymEnv7(interactive=True)
    c.CheckmateTacticGymEnv8(interactive=True)


if __name__ == "__main__":
    run_interactive_demos()
    # run_random_agent_demo(c.CheckmateTacticGymEnv1)
