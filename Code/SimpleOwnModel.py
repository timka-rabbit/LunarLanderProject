import gym
from EnvModel import EnvironmentModel
from SolverClass import Solver
from ControlClass import ControlModule
import pylab

class SimpleOwnModel():
    def __init__(self):
        # Инициализируем среду
        self.env = gym.make('LunarLander-v2')
        # Объект модуля среды
        self.env_model = EnvironmentModel()
        # Объект класса поиска управления
        self.solver = Solver(self.env_model)
        # Объект модуля управления
        self.control_module = ControlModule(self.solver)

    ''' Функция запуска обучения '''
    def Run(self, count_of_episodes):
        pylab.figure(figsize=(18, 9))
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        reward_list = []
        average_list = []
        episode = 0
        while True:
            observation = self.env.reset()
            reward_for_episode = 0
            while True:
                self.env.render()
                action = self.control_module.GetControl(observation[0:6])
                
                observation, reward, done, info = self.env.step(action)
                
                reward_for_episode += reward
                if done:
                    episode += 1
                    self.control_module.AddTrajectory(reward_for_episode)
                    reward_list.append(reward_for_episode)
                    average_list.append(sum(reward_list[-50:]) / len(reward_list[-50:])) 
                    print("Episode: {}/{}, Score: {:.2f}, Average: {:.2f}".format(episode, count_of_episodes, reward_for_episode, average_list[-1]))
                    break
                
            if episode % 10 == 0:
                pylab.xlabel('Steps', fontsize = 18)
                pylab.ylabel('Score', fontsize = 18)
                pylab.title('LunarLander-v2 Own Model training cycle', fontsize = 18)
                pylab.plot([(i + 1) for i in range(episode)], reward_list, color = '#00fa9a')
                pylab.plot([(i + 1) for i in range(episode)], average_list, color = '#026340')
                try:
                    pylab.grid(True)
                    pylab.savefig('../Data/Graphics/own_model.png')
                except OSError:
                    pass
                
            if episode >= count_of_episodes:
                break
                    
        self.env.close()