
import numpy as np, gym, pickle
from tqdm import trange
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n
low, high = env.observation_space.low, env.observation_space.high

def scale_state_varibles(s, eta, low=env.observation_space.low, high=env.observation_space.high):
    """ Rescaling of s to the box [0,1]^2 
        and features transformation 
    """
    x = (s - low) / (high - low)
    return np.cos(np.pi * np.dot(eta, x))


def Qvalues(s, w):
    """ Q Value computation """
    return np.dot(w, s)


N_EPISODES = 50
CONFIDENCE_PASS = -135
p = 3
try:
    f = open('weights.pkl', 'rb')
    data = pickle.load(f)
    if 'W' not in data or 'N' not in data:
        print('Matrix W or N are missing in the dictionary.')
        exit(-1)
    w = data['W']
    eta = data['N']
    if w.shape[1] != eta.shape[0]:
        print('m is not the same for the matrices W and N')
        exit(-1)
    m = w.shape[1]
    if w.shape[0] != k:
        print('The first dimension of W is not {}'.format(k))
        exit(-1)
    if eta.shape[1] != 2:
        print('The first dimension of W is not {}'.format(2))
        exit(-1)
except:
    print('File weights.pkl not found!')
    exit(-1)

episode_reward_list = []
print('Checking solution...')
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES:
    EPISODES.set_description('Episode {}'.format(i))
    done = False
    state = scale_state_varibles(env.reset(), eta, low, high)
    total_episode_reward = 0.0
    qvalues = Qvalues(state, w)
    action = np.argmax(qvalues)
    while not done:
        next_state, reward, done, _ = env.step(action)
        next_state = scale_state_varibles(next_state, eta, low, high)
        qvalues_next = Qvalues(next_state, w)
        next_action = np.argmax(qvalues_next)
        total_episode_reward += reward
        state = next_state
        qvalues = qvalues_next
        action = next_action

    episode_reward_list.append(total_episode_reward)
    env.close()

avg_reward = np.mean(episode_reward_list)
confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)
print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(avg_reward, confidence))
if avg_reward > CONFIDENCE_PASS:
    print('Your policy passed the test!')
else:
    print('Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence'.format(CONFIDENCE_PASS))
