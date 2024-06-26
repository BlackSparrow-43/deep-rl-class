# %%
!pip install "ray[rllib]" tensorflow torch
!pip install gym[atari] autorom[accept-rom-license]
!pip install pybullet
!pip install ale-py
!pip install -U ipywidgets
!pip install 'gymnasium[atari,accept-rom-license]'
!pip install gputil
!pip install gym[classic_control]
!sudo apt-get install xvfb
!pip install pyvirtualdisplay Pillow
!pip install huggingface_sb3
!pip install wandb
!pip install ray[rllib]

# %%
import GPUtil

gpus = GPUtil.getGPUs()
for gpu in gpus:
    print("GPU ID: {}, GPU Name: {}".format(gpu.id, gpu.name))
    print("GPU Load: {}, GPU Free Memory: {}".format(gpu.load, gpu.memoryFree))
    print("GPU Total Memory: {}, GPU Temperature: {}".format(gpu.memoryTotal, gpu.temperature))
    
# %%
try:
    import gymnasium as gym

    gymnasium = True
except Exception:
    import gym

    gymnasium = False

from ray.rllib.algorithms.ppo import PPOConfig
import ale_py

# Register the environments
#ale_py.gym.register()

env_name = "ALE/Pong-ram-v5"
env = gym.make(env_name)
print(env.action_space)
print(env.observation_space)

# %%
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=10)
    .resources(num_gpus=1)
    .environment(env="PongDeterministic-v4")
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
# %%
