import numpy as np
from EnvModel import EnvironmentModel


''' Класс подбора оптимального управления '''
class Solver():
    def __init__(self, envModel : EnvironmentModel):
        self.envModel = envModel
        
    # Подбор оптимального действия по текущему состоянию
    # @ state  - состояния
    # @ action - действия
    def Solve(self, state):
        # RDF предсказания
        prediction = self.envModel.predictByRDF(state)
        action = np.argmin(prediction)
        
        # Табличные предсказания
        #action = self.envModel.predictByTable(state)
        return action
    
    # Отправляем траектории на переобучение
    # @ trajectories  - набор траекторий
    def UpdateEnv(self, trajectories):
        self.envModel.Retrain(trajectories)