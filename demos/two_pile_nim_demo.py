from generalization_grid_games.envs import two_pile_nim as tpn
from generalization_grid_games.envs.utils import run_random_agent_demo


def run_interactive_demos():
    tpn.TwoPileNimGymEnv1(interactive=True)
    tpn.TwoPileNimGymEnv2(interactive=True)
    tpn.TwoPileNimGymEnv3(interactive=True)
    tpn.TwoPileNimGymEnv4(interactive=True)
    tpn.TwoPileNimGymEnv5(interactive=True)
    tpn.TwoPileNimGymEnv6(interactive=True)
    tpn.TwoPileNimGymEnv7(interactive=True)
    tpn.TwoPileNimGymEnv8(interactive=True)


if __name__ == "__main__":
    run_interactive_demos()
    # run_random_agent_demo(tpn.TwoPileNimGymEnv1)
