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

from env.singlecooked_gym import SinglecookedAI,TrainingCallbacks

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
    "--stop-iters", type=int, default=1000, help="Number of iterations to train."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)
parser.add_argument(
    "--clip_param", type=float, default=0.1543,
    help="PPO clipping factor"
)
parser.add_argument(
    "--gamma", type=float, default=0.9777,
    help="Discount factor"
)
parser.add_argument(
    "--grad_clip", type=float, default=0.2884,
    help="Gradient clipping value"
)
parser.add_argument(
    "--kl_coeff", type=float, default=0.2408,
    help="Initial coefficient for KL divergence"
)
parser.add_argument(
    "--lmbda", type=float, default=0.6,
    help="Lambda for GAE (Generalized Advantage Estimation)"
)
parser.add_argument(
    "--lr", type=float, default=2.69e-4,
    help="Learning rate"
)
parser.add_argument(
    "--num_training_iters", type=int, default=500,
    help="Number of training iterations"
)
parser.add_argument(
    "--reward_shaping_horizon", type=int, default=4500000,
    help="Reward shaping horizon"
)
parser.add_argument(
    "--use_phi", type=bool, default=False,
    help="Whether to use phi for dense reward"
)
parser.add_argument(
    "--vf_loss_coeff", type=float, default=0.0069,
    help="Value function loss coefficient"
)
parser.add_argument(
    "--seed", type=int, default=0,
    help="This will set the seed for the random number generator used by Ray, which can be important for reproducibility."
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
        .environment(SinglecookedAI, 
                     env_config={"map_name": args.map_name, 
                                 "use_phi":args.use_phi,
                                 "reward_shaping_horizon":args.reward_shaping_horizon}
        )
        .framework(args.framework)
        .rollouts(num_rollout_workers=30,rollout_fragment_length = 400)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
        .training(
            clip_param = args.clip_param,
            gamma = args.gamma,
            grad_clip = args.grad_clip, 
            kl_coeff = args.kl_coeff,
            lambda_ = args.lmbda,
            lr = args.lr,
            vf_loss_coeff = args.vf_loss_coeff,
            train_batch_size = 12000,
            sgd_minibatch_size = 2000,
            num_sgd_iter = 8,
            entropy_coeff_schedule = [
                (0, 0.2),
                (3e5, 0.1),
            ],
            model={
                "custom_model": "custom_actor_critic_model"
            }
        )
        .callbacks(TrainingCallbacks)
        .evaluation(evaluation_interval = 50)
    )

    # manual training with train loop using PPO and fixed learning rate
    print("Running manual train loop without Ray Tune.")
    config.seed=args.seed
    algo = config.build()

    # run manual training loop and print results after each iteration
    for _ in range(args.num_training_iters):
        result = algo.train()
        print(pretty_print(result))
    
     # 保存 Actor 网络的权重
    model = algo.get_policy().model
    actor_weights_path = f"/workspace/Competition_OvercookedAI-2/{args.map_name}_sd{args.seed}.pth"
    torch.save(model.actor_network.state_dict(), actor_weights_path)
    print(f"Actor model saved at: {actor_weights_path}")
    algo.stop()
    

    ray.shutdown()