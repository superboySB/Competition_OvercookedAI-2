<img src="imgs/Jidi%20logo.png" width='300px'> 

# Solution of RLChina Competition - Gui Mao Winter Season 癸卯年冬赛季

This repo provide the source code of a solution for the [RLChina Competition - Gui Mao Winter Season](http://www.jidiai.cn/compete_detail?compete=44), by BIT-LINC.


## Our Solution

### 安装教程
```sh
docker build -t zsc_image:1.0 .

docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --gpus all --network=host --name=zsc zsc_image:1.0 /bin/bash

conda create -n madrona python=3.10 && conda activate madrona && pip install torch numpy tensorboard 

cd /workspace && git clone https://github.com/superboySB/Competition_OvercookedAI-2 && cd Competition_OvercookedAI-2 && git submodule update --init --recursive

mkdir build && cd build && cmake -D CUDAToolkit_ROOT=/usr/local/cuda .. && make -j8

cd .. && pip install -e . && pip install -e overcooked_ai
```

### 目前计划 1225
1. 尝试根据比赛的obs，并且仔细看一下我现在solution训练的obs，去找到对应关系，使得我的模型可以apply，原理上可以参考这个博客，哈哈哈，我还是用类似GPU仿真这样的玩具，发现效果还不错
2. 尝试看能不能submission里面放三个pt（现在已经训出来了，每一个都是1千万步，你可以尝试在实验室机器配环境训练，差不多1块卡训1000万步只需要5分钟），看看官方是不是会有限制，但目前确实三个模型如果都是CNN的话，地图大小不一样，那模型的维度也都不一样。。。如果限制只能交一个pt，那可能会有distillation的需求
3. 看看目前训练脚本的超参数是不是还能对照overcooked比较合理的论文，进一步调一调。由于训练非常快（分钟级），所以直接log都打印在本地了，训完直接就测，所以没有接wandb和tensorboard，必要的时候也可以尝试接一下

还有不到1个月开始正赛，我觉得接下来这三件事情搞完应该能有一个还可以的成绩，也脱离了前沿顶会，追求了工程的极致。第一步应该是最难的地方，我昨天接通了这个训练后，也一直在看，这个要结合[作者的博客](https://bsarkar321.github.io/blog/overcooked_madrona/index.html)（这个作者是斯坦福做ZSC的那个组的学生，其实我们也算是紧跟前沿），以及实际的C++代码、Python代码，去和比赛定义的环境作对比

## 比赛官方介绍

### OvercookedAI-Integrated II
<img src='https://jidi-images.oss-cn-beijing.aliyuncs.com/jidi/env103.gif' width=400>

- The integrated game contains three official maps, they are： 
  1. Cramped Room Tomato; 
  2. Forced Coordination Tomato; 
  3. Soup Coordination
  
  Config details can be found in [layouts](.env/layouts)

- The game proceed by putting both agents sequentially in these three maps and ask them to prepare orders by cooperating with the other player. The ending state in a map is followed by an initial state in the next map and the agent observation will be marked *new_map=True*
- Each map will be run twice with agent index switched. For example in map one, player one controls agent one and player two controls agent two and they switch position and re-start the map when reaching an end. Thus, two players will play on three maps for six rounds in total.
- Each map last for 400 timesteps. The total episode length of the integrated game is 2400.
- Each agent observation global game state, the agent index and the new map indicator. The gloabl game state includes:
  - players: position and orientation of two characters and the held objects.
  - objects: the information and position of objects in the map.
  - bounus orders
  - all orders
  - timestep:  timestep in the current map
- Each agent has six action choices, they are:
  - Move Up
  - Move Down
  - Move Right
  - Move Left
  - Stay Still
  - Interact
- We use the default reward shaping, that is the value for each successful orders (x2 if bounus order). We will sum up all rewards in all maps as the score for one episode.


## Quick Start

You can use any tool to manage your python environment. Here, we use conda as an example.

```bash
conda create -n overcookai-venv python==3.7.5  #3.8, 3.9
conda activate overcookai-venv
```

Next, clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/jidiai/Competition_OvercookedAI-2.git
cd Competition_OvercookedAI-2
pip install -r requirements.txt
```

Finally, run the game by executing:
```bash
python run_log.py
```

## Visualization

We provide two visualization method:
- Console: set `self.render_mode="console"` in `env/overcookedai_integrated.py` and the game map will be displayed as string on the console.
- Window: set `self.render_mode="window"` and the game map will be displayed in a seperated pygame window.


## Navigation

```
|-- Competition_OvercookedAI-2               
	|-- agents                              // Agents that act in the environment
	|	|-- random                      // A random agent demo
	|	|	|-- submission.py       // A ready-to-submit random agent file
	|-- env		                        // scripts for the environment
	|	|-- config.py                   // environment configuration file
	|	|-- overcookedai_integrated.py  // The environment wrapper		      
	|-- utils               
	|-- run_log.py		                // run the game with provided agents (same way we evaluate your submission in the backend server)
```



## How to test submission

- You can train your own agents using any framework you like as long as using the provided environment wrapper. 

- For your ready-to-submit agent, make sure you check it using the ``run_log.py`` scrips, which is exactly how we 
evaluate your submission.

- ``run_log.py`` takes agents from path `agents/` and run a game. For example:

>python run_log.py --my_ai "random" --opponent "random"

set both agents as a random policy and run a game.

- You can put your agents in the `agent/` folder and create a `submission.py` with a `my_controller` function 
in it. Then run the `run_log.py` to test:

>python run_log.py --my_ai your_agent_name --opponent xxx

- If you pass the test, then you can submit it to the Jidi platform. You can make multiple submission and the previous submission will
be overwritten.


