# -*- coding:utf-8  -*-
import os
import time
import json
import numpy as np
import argparse
import sys

sys.path.append("./olympics_engine")

from env.chooseenv import make
from utils.get_logger import get_logger
from env.obs_interfaces.observation import obs_type


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_players_and_action_space_list(g):
    if sum(g.agent_nums) != g.n_player:
        raise Exception("agent number = %d n_player = %d " % (sum(g.agent_nums), g.n_player))

    n_agent_num = list(g.agent_nums)
    for i in range(1, len(n_agent_num)):
        n_agent_num[i] += n_agent_num[i - 1]

    # 
    players_id = []
    actions_space = []
    for policy_i in range(len(g.obs_type)):
        if policy_i == 0:
            players_id_list = range(n_agent_num[policy_i])
        else:
            players_id_list = range(n_agent_num[policy_i - 1], n_agent_num[policy_i])
        players_id.append(players_id_list)

        action_space_list = [g.get_single_action_space(player_id) for player_id in players_id_list]
        actions_space.append(action_space_list)

    return players_id, actions_space


def get_joint_action_eval(game, multi_part_agent_ids, policy_list, actions_spaces, all_observes):
    if len(policy_list) != len(game.agent_nums):
        error = "%d" % (len(policy_list), len(game.agent_nums))
        raise Exception(error)

    # [[[0, 0, 0, 1]], [[0, 1, 0, 0]]]
    joint_action = []
    for policy_i in range(len(policy_list)):

        if game.obs_type[policy_i] not in obs_type:
            raise Exception("%s" % str(obs_type))

        agents_id_list = multi_part_agent_ids[policy_i]

        action_space_list = actions_spaces[policy_i]
        function_name = 'm%d' % policy_i
        for i in range(len(agents_id_list)):
            agent_id = agents_id_list[i]
            a_obs = all_observes[agent_id]
            each = eval(function_name)(a_obs, action_space_list[i], game.is_act_continuous)
            joint_action.append(each)
            #print(each)
    #print(joint_action)
    return joint_action


def set_seed(g, env_name):
    if env_name.split("-")[0] in ['magent']:
        g.reset()
        seed = g.create_seed()
        g.set_seed(seed)
def extract_env_actions(json_file_path):
    try:
        # 读取文件内容
        with open(json_file_path, 'r') as file:
            json_string = file.read()
            json_data = json.loads(json.loads(json_string))  # 两次解析

        # 提取 "env_actions"
        env_actions_list = []
        if isinstance(json_data, dict) and "steps" in json_data:
            for step in json_data["steps"]:
                if "info_before" in step and "env_actions" in step["info_before"]:
                    env_actions_list.append(step["info_before"]["env_actions"])
            return env_actions_list

    except json.JSONDecodeError:
        print("文件内容无法解析为 JSON。")
        return []
    except Exception as e:
        print(f"处理时发生错误：{e}")
        return []

def replace_env_actions(env_actions_list):
    replacement_rules = {
        (0, -1): [[1, 0, 0, 0, 0, 0]], #上
        (0, 1): [[0, 1, 0, 0, 0, 0]], #下
        (1, 0): [[0, 0, 0, 1, 0, 0]], #右
        (-1, 0): [[0, 0, 1, 0, 0, 0]], #左
        (0, 0): [[0, 0, 0, 0, 1, 0]],
        'interact': [[0, 0, 0, 0, 0, 1]]
    }
    for i, action_pair in enumerate(env_actions_list):
        for j, action in enumerate(action_pair):
            action_tuple = tuple(action) if isinstance(action, list) else action
            if action_tuple in replacement_rules:
                env_actions_list[i][j] = replacement_rules[action_tuple]
    return env_actions_list

def run_game(g, env_name, multi_part_agent_ids, actions_spaces, policy_list, render_mode):
    """
    This function is used to generate log for Vue rendering. Saves .json file
    """
    i = 0
    log_path = os.getcwd() + '/logs/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = get_logger(log_path, g.game_name, json_file=render_mode)
    set_seed(g, env_name)

    for i in range(len(policy_list)):
        if policy_list[i] not in get_valid_agents():
            raise Exception("agent {} not valid!".format(policy_list[i]))

        file_path = os.path.dirname(os.path.abspath(__file__)) + "/agents/" + policy_list[i] + "/submission.py"
        if not os.path.exists(file_path):
            raise Exception("file {} not exist!".format(file_path))

        import_path = '.'.join(file_path.split('/')[-3:])[:-3]
        function_name = 'm%d' % i
        import_name = "my_controller"
        import_s = "from %s import %s as %s" % (import_path, import_name, function_name)
        print(import_s)
        exec(import_s, globals())

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info = {"game_name": env_name,
                 "n_player": g.n_player,
                 "board_height": g.board_height if hasattr(g, "board_height") else None,
                 "board_width": g.board_width if hasattr(g, "board_width") else None,
                 "init_info": g.init_info,
                 "start_time": st,
                 "mode": "terminal",
                 "seed": g.seed if hasattr(g, "seed") else None,
                 "map_size": g.map_size if hasattr(g, "map_size") else None}

    steps = []
    all_observes = g.all_observes

    # 指定您的 JSON 文件路径
    json_file_path = 'log_1705048238243.json'  # 替换为您的文件路径

    # 提取 env_actions
    extracted_env_actions = extract_env_actions(json_file_path)

    # 应用替换规则
    modified_env_actions = replace_env_actions(extracted_env_actions)

    # 打印提取结果的前几个元素（如果有的话）
    #print(modified_env_actions[:5])



    while not g.is_terminal():
        step = "step%d" % g.step_cnt
        time.sleep(0.01)
        if g.step_cnt % 10 == 0:
            #print(step)
            pass

        if render_mode and hasattr(g, "env_core"):
            if hasattr(g.env_core, "render"):
                g.env_core.render()
        elif render_mode and hasattr(g, 'render'):
            g.render()

        info_dict = {"time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}
        joint_act = get_joint_action_eval(g, multi_part_agent_ids, policy_list, actions_spaces, all_observes)

        joint_act = modified_env_actions[i]
        #joint_act = [modified_env_actions[i][1],modified_env_actions[i][0]]

        i +=1


        all_observes, reward, done, info_before, info_after = g.step(joint_act)
        print(step, joint_act,reward)

        if env_name.split("-")[0] in ["magent"]:
            info_dict["joint_action"] = g.decode(joint_act)
        if info_before:
            info_dict["info_before"] = info_before
        info_dict["reward"] = reward
        if info_after:
            info_dict["info_after"] = info_after
        steps.append(info_dict)

    game_info["steps"] = steps
    game_info["winner"] = g.check_win()
    game_info["winner_information"] = g.won
    game_info["n_return"] = g.n_return
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed
    logs = json.dumps(game_info, ensure_ascii=False, cls=NpEncoder)
    logger.info(logs)


def get_valid_agents():
    dir_path = os.path.join(os.path.dirname(__file__), 'agents')
    return [f for f in os.listdir(dir_path) if f != "__pycache__"]


if __name__ == "__main__":

    env_type = "overcookedai-integrated"
    game = make(env_type, seed=None)

    render_mode = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="random", help="random")
    parser.add_argument("--opponent", default="finetune_0111", help="random")
    args = parser.parse_args()
    
    print("policy")

    # policy_list = ["random"] * len(game.agent_nums)
    policy_list = [args.opponent, args.my_ai] #["random"] * len(game.agent_nums), here we control agent 2 (green agent)

    multi_part_agent_ids, actions_space = get_players_and_action_space_list(game)

    run_game(game, env_type, multi_part_agent_ids, actions_space, policy_list, render_mode)