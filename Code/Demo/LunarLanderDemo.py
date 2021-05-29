import gym
env = gym.make('LunarLander-v2')
for i_episode in range(5):
    print('|--------------------------------------------------|')
    observation = env.reset()
    for t in range(200):
        env.render()
        print(observation)
        action = env.action_space.sample()
        if (t % 2 == 0):
            observation, reward, done, info = env.step(2)
        else:
            observation, reward, done, info = env.step(0)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()