''' Код приложения '''

import os
import gym
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import warnings
import pylab
import random
warnings.filterwarnings('ignore')
tf.compat.v1.disable_eager_execution()
tf.autograph.experimental.do_not_convert()

env = gym.make('LunarLander-v2')

class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = tf.keras.layers.Input(input_shape)
        self.action_space = action_space

        X = tf.keras.layers.Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = tf.keras.layers.Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = tf.keras.layers.Dense(self.action_space, activation="softmax")(X)

        self.Actor = tf.keras.models.Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr), experimental_run_tf_function=False)

    def ppo_loss(self, y_true, y_pred):
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = tf.keras.backend.clip(prob, 1e-10, 1.0)
        old_prob = tf.keras.backend.clip(old_prob, 1e-10, 1.0)

        ratio = tf.keras.backend.exp(tf.keras.backend.log(prob) - tf.keras.backend.log(old_prob))
        
        p1 = ratio * advantages
        p2 = tf.keras.backend.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -tf.keras.backend.mean(tf.keras.backend.minimum(p1, p2))

        entropy = -(y_pred * tf.keras.backend.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * tf.keras.backend.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state):
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(env.action_space.n, p = prediction)
        return action


if __name__ == "__main__":  
    actor = Actor_Model(env.observation_space.shape, env.action_space.n, 0.00025, tf.keras.optimizers.Adam)
    actor.Actor.load_weights("LunarLander-v2_PPO_Actor.h5")
    
    os.system("cls")
    while True:
        try:
            episodes = int(input('Введите число эпизодов: '))
            break
        except:
            print('Введите число!')
    
    while True:
        agree = input('Сохранить диаграммы? [y/n] ')
        
        if agree == 'y':
            chart_flag = True
            break
        elif agree == 'n':
            chart_flag = False
            break
        else:
            print('Выберите [y/n]')
    
    reward_list = []
    
    for episode in range(episodes):
        total_reward = 0.0
        observation = env.reset()
        while True:
            env.render()
            action = actor.predict(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                reward_list.append(total_reward)
                print("Эпизод: {0}/{1}, Счёт: {2:.2f}".format(episode + 1, episodes, total_reward))
                break
    env.close()
    
    # Построение диаграммы разброса и гистограммы 
    if chart_flag == True:
        reward_list.sort()
        
        min_limit = int((reward_list[0] // 100) * 100)
        max_limit = int((reward_list[-1] // 100 + 1) * 100)
        group_list = []
        label_list = []
        for i in range(min_limit, max_limit, 100):
            group_list.append(list(filter(lambda x: x > i and x <= i + 100, reward_list)))
            label_list.append(f'{i}:{i + 100}')
        
        
        pylab.figure(figsize = (22, 10))
        pylab.subplots_adjust(left = 0.05, right = 0.98, top = 0.96, bottom = 0.08)
        
        # Диаграмма разброса
        pylab.subplot (1, 2, 1)
        for i in range(len(group_list)):
            shuffle_list_1 = random.sample(group_list[i], len(group_list[i]))
            shuffle_list_2 = random.sample(group_list[i], len(group_list[i]))
            pylab.scatter(shuffle_list_1, shuffle_list_2, s = 20,
                          data = group_list[i],
                          label = label_list[i])
        
        
        pylab.gca().set(xlim = (min_limit, max_limit),
                        ylim = (min_limit, max_limit),
                        xlabel = 'Score', ylabel = 'Score')
        pylab.title("Разброс результатов обучившейся модели", fontsize = 20)
        pylab.legend(fontsize = 12)
        
        # Гистограмма
        pylab.subplot (1, 2, 2)
        all_colors = list(pylab.cm.colors.cnames.keys())
        random.seed(100)
        col = random.choices(all_colors, k = len(group_list))
        pylab.bar(label_list, list(map(lambda x: len(x), group_list)),  color = col, width = .5)
        for i in range(len(group_list)):
            pylab.text(i, len(group_list[i]), len(group_list[i]), horizontalalignment = 'center',
                       verticalalignment = 'bottom', fontdict = {'fontweight':500, 'size':12})
        
        pylab.gca().set_xticklabels(label_list, rotation = 60, horizontalalignment = 'right')
        pylab.gca().set(ylabel = 'Count')
        pylab.title("Гистограмма результатов обучившейся модели", fontsize = 20)
        
        # Сохранение картинки
        pylab.savefig(f'scatter_chart_{len(reward_list)}.png')
    
    input('Нажмите любую кнопку для завершения...')