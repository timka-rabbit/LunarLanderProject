import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Класс предсказания состояний на основе RDF
class EnvironmentModel():
    def __init__(self):
        self.df    = pd.read_csv('../Data/lander_data_episodes.csv').head(5000)
        self.seed  = 3
        self.__learn_model()
        self.labels = ['X', 'Y', 'VelX', 'VelY', 'Angle', 'AngularVel']
        self.next_labels = ['ResX', 'ResY', 'ResVelX', 'ResVelY', 'ResAngle', 'ResAngularVel']
        
    # Обучение модели (RDF)
    def __learn_model(self):   
        # Выбираем найзвания столбцов из датасета
        X = self.df[['X', 'Y', 'VelX', 'VelY', 'Angle', 'AngularVel', 'ActionID']].to_numpy()
        y = np.array(list(map(int, abs(self.df['ResX'].values) * 100
                                  + abs(self.df['Angle'].values) * 100
                                  + abs(self.df['ResVelY'].values) * 100)))

        # 90% выборки идёт на обучение 
        train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                            test_size = 0.1, 
                                                            random_state = self.seed)
        
        # Создание RDF
        self.RDFmodel = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
        # Обучение модели на тренировочных данных 
        self.RDFmodel.fit(train_X, train_y)
    
    
    # Обучение модели на обновлённом наборе траекторий
    # @ trajectories - новые траектории
    def Retrain(self, trajectories):
        for tr in trajectories:
            id_tr = self.df.iloc[-1]['ID_episode'] + 1
            for st in tr:
                id_st = self.df.iloc[-1]['ID'] + 1
                self.df.loc[len(self.df)] = [id_st, id_tr] + st
        
        count = len(trajectories) // 2
        for i in range(count):
            self.df = self.df[self.df['TotalReward'] != min(self.df['TotalReward'])]
        self.__learn_model()
        print('##### Retrain complete #####')
        
    # Метод расчёта Евклидова расстояния между двумя траекториями
    # @ state_1 - 1-ое состояние
    # @ state_2 - 2-ое состояние
    # @ dist    - расстояние между ними
    def __euqlidDist(self, state_1, state_2):
        dist = 0
        for i in range(len(state_1)):
            dist += (state_1[i] - state_2[i])**2
        dist = math.sqrt(dist)
        return dist
        
    
    # Метод предсказания reward на основе RDF по текущему состоянию
    # @ current_states -  состояние
    # @ prediction     -  предсказанные rewards по состояниям
    def predictByRDF(self, current_state):
        df = pd.DataFrame(index = [0, 1, 2, 3], 
        columns = ['X', 'Y', 'VelX', 'VelY', 'Angle', 'AngularVel', 'ActionID'])

        for i in range(4):
            for j in range(7):
                df.loc[i][j] = current_state[j] if j != 6 else i

        df = df.to_numpy()
        prediction = self.RDFmodel.predict(df)
        return prediction
        
    
    # Метод предсказания действия табличным способом на текущем состоянии
    # @ current_states -  состояние
    # @ prediction     -  предсказанные rewards по состояниям
    def predictByTable(self, current_state):
        delta = 1.0
        count_states = 10
        u_rew = []
        for i in range(len(self.df)):
            if(self.__euqlidDist(current_state, np.array(self.df.iloc[i][self.labels])) <= delta):
                u_rew.append(np.array((self.df.iloc[i]['ActionID'], self.df.iloc[i]['TotalReward'])))
               
            if(len(u_rew) >= count_states):
                break
        
        prediction = 0
        max_r = -10000
        for i in range(len(u_rew)):
            if(max_r < u_rew[i][1]):
                max_r = u_rew[i][1]
                prediction = u_rew[i][0]
            
        return int(prediction)
