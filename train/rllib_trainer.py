"""
Example of a custom gym environment. Run this example for a demo.

This example shows the usage of:
  - a custom environment
  - Ray Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import argparse
import numpy as np
import os
import sys

import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from env.singlecooked_gym import SinglecookedAI

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--map-name", type=str, default="cramped_room_tomato", help="cramped_room_tomato / forced_coordination_tomato / soup_coordination"
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=3, help="Number of iterations to train."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# 计算卷积层输出尺寸
def conv2d_size_out(size, kernel_size = 3, stride = 1):
    return (size - (kernel_size - 1) - 1) // stride + 1

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorNetwork, self).__init__()

        conv_height = conv2d_size_out(input_shape[0], 3)
        conv_width = conv2d_size_out(input_shape[1], 3)
        linear_input_size = conv_height * conv_width * 32

        self.base = nn.Sequential(
            layer_init(nn.Conv2d(input_shape[-1], 32, kernel_size=(3, 3), stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(linear_input_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU()
        )
        self.action_out = layer_init(nn.Linear(64, num_actions), std=0.01)

    def forward(self, x: torch.Tensor):
        x = x.to(dtype=torch.float)
        x = x.permute((0, 3, 1, 2))  # 调整x的形状以适应网络
        return self.action_out(self.base(x))

class CriticNetwork(nn.Module):
    def __init__(self, input_shape):
        super(CriticNetwork, self).__init__()
        
        conv_height = conv2d_size_out(input_shape[0], 3)
        conv_width = conv2d_size_out(input_shape[1], 3)
        linear_input_size = conv_height * conv_width * 32
        
        self.base = nn.Sequential(
            layer_init(nn.Conv2d(input_shape[-1], 32, kernel_size=(3, 3), stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(linear_input_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU()
        )

        self.critic_out = layer_init(nn.Linear(64, 1), std=1)

    def forward(self, x):
        x = x.to(dtype=torch.float)
        x = x.permute((0, 3, 1, 2))  # 调整x的形状以适应网络
        return self.critic_out(self.base(x))
    

# 创建自定义 RLlib 模型
class CustomActorCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.actor_network = ActorNetwork(obs_space.shape, action_space.n)
        self.critic_network = CriticNetwork(obs_space.shape)

        # 加载预训练的 Actor 网络权重
        actor_weights_path = f"/workspace/Competition_OvercookedAI-2/results0104/{args.map_name}/mp/1/oracle_8/models/actor.pth"
        key_map = {
            "base.cnn.cnn.0.weight": "base.0.weight",
            "base.cnn.cnn.0.bias": "base.0.bias",
            "base.cnn.cnn.3.weight": "base.3.weight",
            "base.cnn.cnn.3.bias": "base.3.bias",
            "base.cnn.cnn.5.weight": "base.5.weight",
            "base.cnn.cnn.5.bias": "base.5.bias",
            "act.action_out.linear.weight": "action_out.weight",
            "act.action_out.linear.bias": "action_out.bias"
        }
        policy_actor_state_dict = torch.load(actor_weights_path)
        new_state_dict = {key_map.get(k, k): v for k, v in policy_actor_state_dict.items()}
        self.actor_network.load_state_dict(new_state_dict)
        

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        logits = self.actor_network(obs)
        self._value_out = self.critic_network(obs).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out



if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)

    ModelCatalog.register_custom_model("custom_actor_critic_model", CustomActorCriticModel)
    
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        # or "corridor" if registered above
        .environment(SinglecookedAI, env_config={"map_name": args.map_name})
        .framework(args.framework)
        .rollouts(num_rollout_workers=32)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
        .training(
            model={
                "custom_model": "custom_actor_critic_model"
            }
        )
    )

    # manual training with train loop using PPO and fixed learning rate
    print("Running manual train loop without Ray Tune.")
    # use fixed learning rate instead of grid search (needs tune)
    config.lr = 1e-3
    algo = config.build()

    # run manual training loop and print results after each iteration
    for _ in range(args.stop_iters):
        result = algo.train()
        print(pretty_print(result))
    
     # 保存 Actor 网络的权重
    model = algo.get_policy().model
    actor_weights_path = f"/workspace/Competition_OvercookedAI-2/{args.map_name}.pth"
    torch.save(model.actor_network.state_dict(), actor_weights_path)
    print(f"Actor model saved at: {actor_weights_path}")
    algo.stop()
    

    ray.shutdown()