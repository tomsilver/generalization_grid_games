from gym.envs.registration import register


for i in range(20):
    register(
        id='TwoPileNim{}-v0'.format(i),
        entry_point='generalization_grid_games.envs:TwoPileNimGymEnv{}'.format(i),
    )

for i in range(20):
    register(
        id='CheckmateTactic{}-v0'.format(i),
        entry_point='generalization_grid_games.envs:CheckmateTacticGymEnv{}'.format(i),
    )

for i in range(20):
    register(
        id='StopTheFall{}-v0'.format(i),
        entry_point='generalization_grid_games.envs:StopTheFallGymEnv{}'.format(i),
    )

for i in range(20):
    register(
        id='Chase{}-v0'.format(i),
        entry_point='generalization_grid_games.envs:ChaseGymEnv{}'.format(i),
    )

for i in range(20):
    register(
        id='ReachForTheStar{}-v0'.format(i),
        entry_point='generalization_grid_games.envs:ReachForTheStarGymEnv{}'.format(i),
    )


