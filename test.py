import gfootball.env as football_env

env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)

state = env.reset()

while True:
  observation, reward, done, info = env.step(env.action_space.sample())
  if done:
    env.reset()
