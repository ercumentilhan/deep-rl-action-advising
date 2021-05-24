from time import localtime, strftime
import random
from constants.general import *


# ======================================================================================================================

def generate_run_id(env_name, dqn_type, config_set):
    run_id = str(ENV_INFO[env_name][0]) + '_' \
             + ('000' if config_set is None else str(config_set).zfill(3)) + '_' \
             + str(random.randint(0, 999)).zfill(3) + '_' \
             + strftime("%Y%m%d-%H%M%S", localtime())
    return run_id


# ======================================================================================================================

def generate_envs(env_key, env_training_seed, env_evaluation_seed):
    env, eval_env = None, None

    env_info = ENV_INFO[env_key]
    env_type = env_info[1]
    env_name = env_info[4]
    time_limit = env_info[8]

    if env_type == ALE:
        import gym
        import ale_preprocessing
        import frame_stack

        # Frame stacking is enabled by default
        env = gym.make(env_name)
        env = ale_preprocessing.AtariPreprocessing(env)
        env = frame_stack.FrameStack(env, num_stack=4)
        env.seed(env_training_seed)

        eval_env = gym.make(env_name)
        eval_env = ale_preprocessing.AtariPreprocessing(eval_env)
        eval_env = frame_stack.FrameStack(eval_env, num_stack=4)
        eval_env.seed(env_evaluation_seed)

    elif env_type == BOX2D:
        import gym

        env = gym.make(env_name)
        env.seed(env_training_seed)

        eval_env = gym.make(env_name)
        eval_env.seed(env_evaluation_seed)

    elif env_type == MINATAR:
        # import minatar_original as minatar
        import minatar_extended as minatar

        env = minatar.Environment(env_name=env_name,
                                  sticky_action_prob=0.1,
                                  difficulty_ramping=True,
                                  random_seed=env_training_seed,
                                  time_limit=time_limit + 1000)

        eval_env = minatar.Environment(env_name=env_name,
                                       sticky_action_prob=0.0,
                                       difficulty_ramping=True,
                                       random_seed=env_evaluation_seed,
                                       time_limit=time_limit + 1000)

    elif env_type == MAPE:
        from mape.environment import MultiAgentEnv
        import mape.scenarios as scenarios

        scenario_h = scenarios.load(env_name + '_h' + '.py').Scenario()
        # scenario_h = scenarios.load(env_name + '.py').Scenario()
        world = scenario_h.make_world()
        env = MultiAgentEnv(world, scenario_h.reset_world, scenario_h.reward, scenario_h.observation,
                            scenario_h.benchmark_data)
        env.discrete_action_input = True
        env.render_resolution = (200, 200)

        scenario = scenarios.load(env_name + ".py").Scenario()

        eval_world = scenario.make_world()
        eval_env = MultiAgentEnv(eval_world, scenario.reset_world, scenario.reward, scenario.observation,
                                 scenario.benchmark_data)
        eval_env.discrete_action_input = True
        eval_env.render_resolution = (200, 200)

    return env, eval_env


# ======================================================================================================================

def config_to_command(config):
    command = 'python main.py'
    for key in config[0]:
        command += ' --' + key + ' ' + str(config[0][key])
    for key in config[1]:
        if config[1][key]:
            command += ' --' + key
    return command


# ======================================================================================================================

def config_to_executor_config(config):
    exec_config = {}
    for key in config[0]:
        exec_config[key.replace('-', '_')] = config[0][key]
    for key in config[1]:
        exec_config[key.replace('-', '_')] = config[1][key]
    return exec_config


# ======================================================================================================================

def print_config(config):
    for key in config:
        print('{}: {}'.format(key, config[key]))
