import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import copy
import gym
import pylab
import tensorboardX
import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.disable_eager_execution()


''' Модель «Актер» выполняет задачу изучения того, какие действия
следует предпринять в конкретном наблюдаемом состоянии окружающей среды.
В случае LunarLander-v2 он принимает список из 8 значений игры в качестве
входных данных, который представляет текущее состояние нашей ракеты и дает
конкретное действие, какой двигатель запускать в качестве выходных данных. '''
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
        return self.Actor.predict(state)
    
    
''' Мы отправляем действие, предсказанное "Актером", в нашу среду и
наблюдаем, что происходит в игре. Если в результате наших действий
происходит что-то положительное, например, посадка космического корабля,
окружающая среда отправляет положительный ответ в виде награды. Но если
наш космический корабль упадет, мы получим отрицательную награду. Эти
награды получаются путем обучения нашей модели "Критиков" '''
class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = tf.keras.layers.Input(input_shape)
        old_values = tf.keras.layers.Input(shape=(1,))

        V = tf.keras.layers.Dense(512, activation="relu", kernel_initializer='he_uniform')(X_input)
        V = tf.keras.layers.Dense(256, activation="relu", kernel_initializer='he_uniform')(V)
        V = tf.keras.layers.Dense(64, activation="relu", kernel_initializer='he_uniform')(V)
        value = tf.keras.layers.Dense(1, activation=None)(V)

        self.Critic = tf.keras.models.Model(inputs=[X_input, old_values], outputs = value)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr), experimental_run_tf_function=False)

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + tf.keras.backend.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * tf.keras.backend.mean(tf.keras.backend.maximum(v_loss1, v_loss2))
            return value_loss
        return loss

    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])


''' Обёртка агента обучения '''
class NetModel:
    # PPO Main Optimization Algorithm
    def __init__(self):
        # Initialization
        # Environment and PPO parameters
        self.env_name = 'LunarLander-v2'       
        self.env = gym.make(self.env_name)
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 0 # total episodes to train
        self.episode = 0 # used to track the episodes total count of episodes
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle=False
        self.Training_batch = 1000
        self.optimizer = tf.keras.optimizers.Adam

        self.replay_count = 0
        self.writer = tensorboardX.SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        
        self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.h5"

    ''' Выбор дейсвия по предсказанным Актёром вероятностям '''
    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction

    ''' Перерасчёт ошибок '''
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    ''' Переобучение Актёра и Критика '''
    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1
 
    ''' Загрузка весов '''
    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    ''' Сохранение весов '''
    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)
    
    ''' Построение графика '''
    def PlotModel(self, score, episode):
        pylab.figure(figsize=(18, 9))
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        # Сохранение графика каждые 10 эпизодов
        if episode % 10 == 0:
            pylab.plot(self.episodes_, self.scores_, color = '#ffc2cc')
            pylab.plot(self.episodes_, self.average_, color = '#800080')
            pylab.title(self.env_name+" PPO training cycle", fontsize=18)
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig('../Data/Graphics/net_model.png')
            except OSError:
                pass
        # Сохранение весов сетей каждые 50 эпизодов
        if episode % 50 == 0:
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
            # decreaate learning rate every saved model
            self.lr *= 0.95
            tf.keras.backend.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            tf.keras.backend.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING
    
    ''' Функция запуска обучения '''
    def Run(self, count_of_episodes, loading_weights = False):
        self.episode = 0    
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING = False, 0, ''
        self.EPISODES = count_of_episodes
        if loading_weights:
            self.load()
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            while True:
                self.env.render()
                # Actor picks an action
                action, action_onehot, prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)
                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("Episode: {}/{}, Score: {:.2f}, Average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)
                    
                    self.replay(states, actions, rewards, predictions, dones, next_states)

                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = np.reshape(state, [1, self.state_size[0]])
                    break
                    
            if self.episode >= self.EPISODES:
                break
        self.env.close()
