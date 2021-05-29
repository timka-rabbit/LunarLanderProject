import pylab
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import gym

class QLearningAgent():
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.forest = self.__learn_forest()
        
    def __learn_forest(self):
        df = pd.read_csv('../Data/lander_data_episodes.csv').head(5000)
        X = df[['X', 'Y', 'VelX', 'VelY', 'Angle', 'AngularVel']].to_numpy()
        y = np.array([i for i in range(len(X))])

        # 90% выборки идёт на обучение 
        train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                            test_size = 0.1, 
                                                            random_state = 3)
        
        # Создание RDF
        model = RandomForestClassifier(n_estimators = 30, criterion = 'gini')
        # Обучение модели на тренировочных данных 
        model.fit(train_X, train_y)     
        return model

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[self.forest.predict([state, state])[0]][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[self.forest.predict([state, state])[0]][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0
        q_vals = [self.get_qvalue(state, a) for a in possible_actions]
        return max(q_vals)

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        q_sa_hat = reward + gamma * self.get_value(next_state)
        new_q_sa = learning_rate * q_sa_hat + (1 - learning_rate)*self.get_qvalue(state, action)

        self.set_qvalue(state, action, new_q_sa)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values). 
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        q_vals = np.array([self.get_qvalue(state, a) for a in possible_actions])
        return possible_actions[np.argmax(q_vals)]

    def get_action(self, state):
        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon
        if np.random.rand() < epsilon:
            chosen_action = np.random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action
    
    

''' Запуск модели обучения Q-Learning '''
class QLearningModel():
    def __init__(self):
        
        self.env = gym.make("LunarLander-v2")
        
        self.n_actions = self.env.action_space.n
        
        self.agent = QLearningAgent(
                     alpha = 0.25, epsilon = 0.25, discount = 0.95,
                     get_legal_actions=lambda s: range(self.n_actions))

    ''' Функция запуска обучения '''
    def Run(self, count_of_episodes):
        pylab.figure(figsize=(18, 9))
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        reward_list = []
        average_list = []
        episode = 0
        while True:
            total_reward = 0.0
            s = self.env.reset()
    
            while True:
                self.env.render()
                # get agent to pick action given state s.
                a = self.agent.get_action(s[0:6])
        
                next_s, r, done, _ = self.env.step(a)
        
                # train (update) agent for state s
                self.agent.update(s[0:6], a, r, next_s[0:6])
        
                s = next_s
                total_reward += r
                if done:
                    episode += 1
                    reward_list.append(total_reward)
                    average_list.append(sum(reward_list[-50:]) / len(reward_list[-50:]))
                    print("Episode: {}/{}, Score: {:.2f}, Average: {:.2f}".format(episode, count_of_episodes, total_reward, average_list[-1]))
                    break
            
            self.agent.epsilon *= 0.99
        
            if episode % 10 == 0:
                pylab.xlabel('Steps', fontsize = 18)
                pylab.ylabel('Score', fontsize = 18)
                pylab.title('LunarLander-v2 Q-Learning training cycle', fontsize = 18)
                pylab.plot([(i + 1) for i in range(episode)], reward_list, color = '#f5e23d')
                pylab.plot([(i + 1) for i in range(episode)], average_list, color = '#b0980e')
                try:
                    pylab.grid(True)
                    pylab.savefig('../Data/Graphics/q-learning_model.png')
                except OSError:
                    pass
                
            if episode >= count_of_episodes:
                break
                        
        self.env.close()
        