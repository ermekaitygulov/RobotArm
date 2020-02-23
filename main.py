from environments.env import RozumEnv

if __name__ == '__main__':
    env = RozumEnv()
    a = env.sample_action()
    env.step(a)
    a = env.reset()
    print(a.shape)
    env.close()