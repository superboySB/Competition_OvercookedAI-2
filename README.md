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

### 基本使用方法
这是比赛submission的本地运行验证文件
```sh
python run_log.py
```
训练madrona并测试，可以对比一些其它文章对这些layout的训练结果，在公测上对比
```sh
python train_and_evaluate.py
```
关于比赛的候选方法，以及更多学术前沿的方法可以尝试跑这里的脚本，然后类似`agent/trial`的做法，打包成比赛模型提交
```sh
cd scripts

# 方法1
./train_sp.sh

# 方法2 (未加入，可模仿方法4加入)
./train_adap.sh
./cbr_adap.sh

# 方法3
./train_xp.sh
./cbr_xp.sh

# 方法4 (当前主要尝试方法, CoMeDi, Stanford 2023，依次运行两个脚本) 
./train_mp.sh
./cbr_mp.sh
```
目前train_mp用时大概xxx小时，cbr_mp用时大概xx小时

### 目前计划 0103
1. 已经从convention角度接入了翰澄推荐的[同组最新论文](http://iliad.stanford.edu/Diverse-Conventions/)，但还是要仔细去看魔改的[overcooked源码的MDP部分](https://github.com/Stanford-ILIAD/Diverse-Conventions/blob/master/src/overcooked2_env/sim.cpp)是否还是同一个环境，因为目前出现了本地测试越来越高，但在线匹配效果不佳的问题。
2. 目前考虑完整训练pipeline之后的用时应该会是一天左右了，应该仔细研究算法背后的逻辑原理，当然我觉得主要的问题应该还是在环境上，毕竟有队伍写的是决策树，但我们在没有基础的情况下，还是努力用zsc的AI模型解决问题，达到对前沿的有效尝试就好，just for fun

## 以下是比赛官方介绍

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


## Reference
主要基于斯坦福开源的新玩具，可以给RL训练做GPU交互加速的研究型游戏引擎（[作者的博客](https://bsarkar321.github.io/blog/overcooked_madrona/index.html)），因为不像isaac系列那样核心代码闭源，所以最近也在讨论这边能不能拿来支持启元这里的AI全栈国产化，


