from SolverClass import Solver

''' Класс модуля управления '''
class ControlModule():
    def __init__(self, solver : Solver):
        self.solver = solver
        self.trajectories = []
        self.cur_trajectory = []
        self.cur_reward_reach = -100
        self.prev_state = None
        
        
    # Метод получения управления от состояния
    # @ state  - сотояние
    # @ action - действие, полученное алгоритмом
    def GetControl(self, state):
        if self.prev_state is not None:
            self.cur_trajectory[len(self.cur_trajectory) - 1] += list(map(lambda x: round(x, 2),state))
        self.cur_trajectory.append(list(map(lambda x: round(x, 2),state)))
        
        action = self.solver.Solve(state)
        
        self.cur_trajectory[len(self.cur_trajectory) - 1].append(action)
        self.prev_state = state
        return action
    
    
    # Метод добавления траектории
    # @ reward - итоговая "цена" траектории
    def AddTrajectory(self, reward):
        if reward > self.cur_reward_reach:
            print("#### Added trajectory with reward: {:.2f}".format(reward))
            del  self.cur_trajectory[len(self.cur_trajectory) - 1]
            for st in self.cur_trajectory:
                st.append(self.cur_reward_reach)
            self.trajectories.append(self.cur_trajectory)
        self.cur_trajectory = []
        self.prev_state = None
        
        if len(self.trajectories) == 10:
            print("##### Sent 10 trajectories to RDF #####")
            self.solver.UpdateEnv(self.trajectories)
            self.trajectories = []
            self.cur_reward_reach += 5
            