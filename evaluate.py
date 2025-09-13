import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from copy import deepcopy
import numpy as np

class StoreAndTerminateWrapper(gym.Wrapper):
    '''
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    '''
    def __init__(self, env):
        super(StoreAndTerminateWrapper,self).__init__(env)
        self.max_steps = 200
        self.current_step = 0
        self.env=env
        self.mem = []
        self.TotalReward = 0.0
        self.first_state = 0
        self.first_obs = 0
        self.prev_obs = 0
        self.states_list = []
        self.info = {}

    def reset(self, *args, **kwargs):
        self.current_step = 0
        obs, info = self.env.reset(*args, **kwargs)
        self.TotalReward = 0.0
        self.first_obs = obs
        return obs,info

    def step(self, action):
        if self.current_step == 0:
            self.prev_obs = self.first_obs
            self.first_state = deepcopy(self.env)
            self.states_list.append(self.first_state)
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.TotalReward += reward
        self.mem.append(tuple((self.prev_obs,action)))
        self.prev_obs = obs
        if self.current_step >= self.max_steps:
            truncated = True
        if obs[0] <= -1.2:
            truncated = True
            reward = -201 - self.TotalReward
            self.TotalReward = -200
        if terminated or truncated:
            self.mem.append(tuple(('done',self.TotalReward)))
        self.info['mem'] = self.mem
        self.info['state'] = self.states_list
        return obs, reward, terminated, truncated, info

    def set_state(self, state):
        self.env = deepcopy(state)
        obs = np.array(list(self.env.unwrapped.state))
        self.current_step = 0
        self.TotalReward = 0.0
        self.first_obs = obs
        return obs

mtc = gym.make('MountainCar-v0')
env = StoreAndTerminateWrapper(mtc)
dqn_model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    train_freq=16,
    gradient_steps=8,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.07,
    target_update_interval=600,
    learning_starts=1000,
    buffer_size=10000,
    batch_size=128,
    learning_rate=4e-3,
    policy_kwargs=dict(net_arch=[256, 256]),
    seed=2,
)

def evaluate_failure_rate(model, eval_env, num_eval_episodes=100) :
        num_failures = 0
        for _ in range(num_eval_episodes):
            obs, info = eval_env.reset()
            terminated = truncated = False
            state = False
            while not (terminated or truncated):
                action, state = model.predict(obs, state=state, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
            num_failures += int(terminated == False)
        failure_rate = num_failures / num_eval_episodes
        return failure_rate

model=DQN.load('D:\\code\\RLtest\\3.zip')
print(evaluate_policy(model, env, n_eval_episodes=2000))
print(evaluate_failure_rate(model, env, num_eval_episodes=2000))