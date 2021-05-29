import itertools
import numpy as np
cimport numpy as np
from itertools import *

cdef class RegressionTreeCython:
    
    cdef public int max_depth
    cdef public int feature_idx
    cdef public int min_size
    cdef public int averages 
    
    cdef public np.float64_t feature_threshold
    cdef public np.float64_t value

    cdef RegressionTreeCython left
    cdef RegressionTreeCython right
    
    def __init__(self, max_depth=3, min_size=4, averages=1):
        
        self.max_depth = max_depth
        self.min_size = min_size
        self.value = 0
        self.averages = averages
        self.feature_idx = -1
        self.feature_threshold = 0
        self.left = None
        self.right = None
            

    def data_transform(self, np.ndarray[np.float64_t, ndim=2] X, list index_tuples):
        
        # преобразование данных - дополнение новыми признаками в виде суммы
        for i in index_tuples:
            # добавляем суммы, индексы которых переданы в качестве аргумента
            X = np.hstack((X, X[:, i[0]:(i[1]+1)].sum(axis=1).reshape(X.shape[0],1)))
        return X
    
    def fit(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y):

        cdef np.float64_t mean1 = 0.0
        cdef np.float64_t mean2 = 0.0
        cdef long N = X.shape[0]
        cdef long N1 = X.shape[0]
        cdef long N2 = 0
        cdef np.float64_t delta1 = 0.0
        cdef np.float64_t delta2 = 0.0
        cdef np.float64_t sm1 = 0.0
        cdef np.float64_t sm2 = 0.0
        cdef list index_tuples
        cdef list stuff
        cdef long idx = 0
        
        cdef np.float64_t prev_error1 = 0.0
        cdef np.float64_t prev_error2 = 0.0
        cdef long thres = 0
        cdef np.float64_t error = 0.0
        
        cdef np.ndarray[long, ndim=1] idxs
        
        cdef np.float64_t x = 0.0
        
        # такую процедуру необходимо сделать только один раз
        # генерируем индексы, по которым суммируем признаки
        if self.averages:
            stuff = list(range(0,X.shape[1],1))
            index_tuples = list(combinations(stuff,2))
            # выполняем преобразование данных
            X = self.data_transform(X, index_tuples)
            
        # начальное значение - среднее значение y
        self.value = y.mean()
        # начальная ошибка - mse между значением в листе (пока нет разбиения, 
        # это среднее по всем объектам) и объектами
        base_error = ((y - self.value) ** 2).sum()
        error = base_error
        flag = 0
        
        # пришли на максимальную глубину
        if self.max_depth <= 1:
            return
    
        dim_shape = X.shape[1]
        
        left_value, right_value = 0, 0
        
        for feat in range(dim_shape):
            
            prev_error1, prev_error2 = base_error, 0 
            idxs = np.argsort(X[:, feat])
            
            # переменные для быстрого переброса суммы
            mean1, mean2 = y.mean(), 0
            sm1, sm2 = y.sum(), 0
            
            N = X.shape[0]
            N1, N2 = N, 0
            thres = 1
            
            while thres < N - 1:
                N1 -= 1
                N2 += 1

                idx = idxs[thres]
                x = X[idx, feat]
                
                # вычисляем дельты - по ним в основном будет делаться переброс
                delta1 = (sm1 - y[idx]) * 1.0 / N1 - mean1
                delta2 = (sm2 + y[idx]) * 1.0 / N2 - mean2
                
                # увеличиваем суммы
                sm1 -= y[idx]
                sm2 += y[idx]
                
                # пересчитываем ошибки за O(1)
                prev_error1 += (delta1**2) * N1 
                prev_error1 -= (y[idx] - mean1)**2 
                prev_error1 -= 2 * delta1 * (sm1 - mean1 * N1)
                mean1 = sm1/N1
                
                prev_error2 += (delta2**2) * N2 
                prev_error2 += (y[idx] - mean2)**2 
                prev_error2 -= 2 * delta2 * (sm2 - mean2 * N2)
                mean2 = sm2/N2
                
                # пропускаем близкие друг к другу значения
                if thres < N - 1 and np.abs(x - X[idxs[thres + 1], feat]) < 1e-5:
                    thres += 1
                    continue
                
                # 2 условия осуществления сплита - уменьшение ошибки
                # и минимальное кол-во элементов в в каждом листе
                if (prev_error1 + prev_error2 < error):
                    if (min(N1,N2) > self.min_size):
                    
                        # переопределяем самый лучший признак и границу по нему
                        self.feature_idx, self.feature_threshold = feat, x
                        # переопределяем значения в листах
                        left_value, right_value = mean1, mean2

                        # флаг - значит сделали хороший сплит
                        flag = 1
                        error = prev_error1 + prev_error2
                                     
                thres += 1
        
        # self.feature_idx - индекс самой крутой разделяющей фичи. 
        # Если это какая-то из сумм, и если есть какое-то экспертное знание 
        # о данных, то интересно посмотреть, что значит эта сумма 
        
        # ничего не разделили, выходим
        if self.feature_idx == -1:
            return
        
        # вызываем потомков дерева
        self.left = RegressionTreeCython(self.max_depth - 1, averages=0)
        self.left.value = left_value
        self.right = RegressionTreeCython(self.max_depth - 1, averages=0)
        self.right.value = right_value
        
        # новые индексы для обучения потомков
        idxs_l = (X[:, self.feature_idx] > self.feature_threshold)
        idxs_r = (X[:, self.feature_idx] <= self.feature_threshold)
        
        # обучение потомков
        self.left.fit(X[idxs_l, :], y[idxs_l])
        self.right.fit(X[idxs_r, :], y[idxs_r])
        
    def __predict(self, np.ndarray[np.float64_t, ndim=1] x):
        
        if self.feature_idx == -1:
            return self.value
        
        if x[self.feature_idx] > self.feature_threshold:
            return self.left.__predict(x)
        else:
            return self.right.__predict(x)
        
    def predict(self, np.ndarray[np.float64_t, ndim=2] X):

        # чтобы делать предикты, нужно также добавить суммы в тестовую выборку
        if self.averages:
            stuff = list(range(0,X.shape[1],1))
            index_tuples = list(itertools.combinations(stuff,2))
            X = self.data_transform(X, index_tuples)
            
        y = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            y[i] = self.__predict(X[i])
            
        return y
