''' Клиентский код '''

import os
from SimpleOwnModel import SimpleOwnModel
from QLearningModel import QLearningModel
from PPONetModel import NetModel

def RunModel(model, episodes):
    model.Run(episodes)

def main():
    os.system("cls")
    model_num = '0'
    while True:
        print('Выберите модель обучения (введите цифру):'
              + '\n1) Собственная модель'
              + '\n2) Q-Learning модель'
              + '\n3) Нейросетевая модель')
        
        nums = ['1', '2', '3']
        model_num = input()
        
        if model_num in nums:
            break
        print('Укажите один из пунктов выше!')
    
    while True:
        try:
            episodes = int(input('Введите желаемое число циклов обучения: '))
            break
        except:
            print('Введите число!')
    
    if model_num == '1':
        RunModel(SimpleOwnModel(), episodes)
        
    elif model_num == '2':
        RunModel(QLearningModel(), episodes)
        
    else:
        RunModel(NetModel(), episodes)


main()