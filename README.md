<img src="imgs/Jidi%20logo.png" width='300px'> 

# Solution of RLChina Competition - Gui Mao Winter Season 癸卯年冬赛季

This repo provide the source code of a solution for the [RLChina Competition - Gui Mao Winter Season](http://www.jidiai.cn/compete_detail?compete=44), by BIT-LINC.


## 我们方案的安装教程
```sh
docker build -t zsc_image:1.0 .

docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --gpus all --network=host --name=zsc zsc_image:1.0 /bin/bash

conda create -n madrona python=3.10 && conda activate madrona && pip install torch numpy tensorboard 

cd /workspace && git clone https://github.com/superboySB/Competition_OvercookedAI-2 && cd Competition_OvercookedAI-2 && git submodule update --init --recursive && mkdir build && cd build && cmake -D CUDAToolkit_ROOT=/usr/local/cuda .. && make -j8

cd .. && pip install -e . && pip install -e overcooked_ai
```

### Overcooked
Our experiments are conducted in three layouts from [On the Utility of Learning about Humans for Human-AI Coordination](https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019), named *Asymmetric Advantages*, *Coordination Ring*, and *Counter Circuit*,  and two designed layouts, named *Distant Tomato* and *Many Orders*. These layouts are named "unident_s", "random1", "random3", "distant_tomato" and "many_orders" respectively in the code.

Note:
- many_orders类似图1：狭窄房间（cramped_room_tomato）
- unident_s类似图2：强制协调（forced_coordination_tomato）
- unident_s类似图3：汤类合作（soup_coordination）
### Training

All training scripts are under directory `hsp/scripts`. All methods consist of two stages, in the first of which a pool of policies are trained and in the second of which an adaptive policy is trained against this policy pool. 

### Self-Play

To train self-play policies, change `layout` to one of "unident_s"(Asymmetric Advantages), "random1"(Coordination Ring), "random3"(Counter Circuit), "distant_tomato"(Distant_Tomato) and "many_orders"(Many Orders) and run `./train_overcooked_sp.sh`.

### FCP

In the first stage, run `./train_sp_all_S1.sh` to train 12 polcicies via self-play on each layout. After the first stage training is done, run `python extract_sp_S1_models.py` to extract init, middle and final checkpoints of the self-play policies into the policy pool. At this step, the policy pools of FCP on all layouts should be in the directory `hsp/policy_pool/LAYOUT/fcp/s1`. 

In the second stage, run `./train_fcp_all_S2.sh` to train an adaptive policy against the policy pool for each layout.

### MEP
We reimplemented [Maximum Entropy Population-Based Training for Zero-Shot Human-AI Coordination](https://github.com/ruizhaogit/maximum_entropy_population_based_training) and achieved significant higher episode reward when paired with human proxy models than reported in original paper. 

For the first stage, run `./train_mep_all_S1.sh`. After training is finished, run `python extract_mep_S1_models.py` to extract checkpoints of the MEP policies into the policy pool. 

For the second stage, run `./train_mep_all_S2.sh`.

### HSP
**Important:** Please make sure you finished the first stage training of MEP before the second stage of HSP.

For the first stage, run `./train_hsp_all_S1.sh`. After training is finished, run `python extract_hsp_S1_models.py` to collect HSP policies into the policy pool. 

Then run `./eval_events_all.sh` to do evaluation to obtain event features for each pair of biased policy and adaptive policy in HSP.  After evaluation is done, for each layout, run `python hsp/greedy_select.py --layout LAYOUT --k 18` to select HSP policies in a greedy manner and generate configuration of policy pool automatically.

For the second stage, run `./train_hsp_all_S2.sh`.

### Evaluation

Run `./eval_overcooked.sh` for evaluation. You can change the layout name, path to YAML file of population configuration and policies to evaluate in `eval_overcooked.sh`. To evaluate with script policies, change policy name to a string with `script:` as prefix, for example, `script:place_onion_and_deliver_soup`. For more script policies, check `script_agent.py` under the overcooked environment directories.

## Multi-Agent Game Evaluation Platform --- Jidi (及第)
Jidi supports online evaluation service for various games/simulators/environments/testbeds. Website: [www.jidiai.cn](www.jidiai.cn).

A tutorial on Jidi: [Tutorial](https://github.com/jidiai/ai_lib/blob/master/assets/Jidi%20tutorial.pdf)


## Environment
The competition uses an integrated version of [OvercookedAI games](https://github.com/HumanCompatibleAI/overcooked_ai)


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


