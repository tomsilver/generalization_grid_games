# Generalization Grid Games
Games on a 2D grid that require substantial generalization.

## System Requirements
We use Python 3.5.6 on macOS High Sierra. Other setups may work but have not been tested.

## Installation
```
pip install -e .
pip install matplotlib
pip install imageio
```

## Usage Examples

### OpenAI Gym Environment Compatibility

```
import gym
import generalization_grid_games

for base_class_name in ["TwoPileNim", "CheckmateTactic", "Chase", "StopTheFall", "ReachForTheStar"]:
    for task_instance in range(20):
        env_name = "{}{}-v0".format(base_class_name, task_instance)
        env = gym.make(env_name)
        obs = env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, done, debug_info = env.step(action)
            if done:
                break
```

### Interactive Demos
```python demos/two_pile_nim_demo.py```
```python demos/checkmate_tactic_demo.py```
```python demos/chase_demo.py```
```python demos/stop_the_fall_demo.py```
```python demos/reach_for_the_star_demo.py```
