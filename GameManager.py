# -*- coding: utf-8 -*-
# Данный файл содержит envinroment как в гуме
# Готовит список файлов с полными именами для загрузки
import numpy as np
import pandas as pd
from Config import Config



    
# Реализация класса environment наподобие такогого из openai.gym
class environment:
    def __init__(self, game_name):#, leng):
        # self.big_data = data_frame.values
        # self.len = leng
        # self.start_len = self.big_data.shape[0]-leng
        #print(self.start_len, leng)
        # start_point = int(np.random.rand()*self.start_len)
        # self.data = self.big_data[start_point:start_point+leng,:]
        try:
            R = pd.read_pickle(game_name)
        except:
            print("ERROR loading pickle: Если нет файла, то его нужно создать с помощью load_data.ipynb")
        #Нормируем данные на нормальное распределение:
        R = (R - R.mean()) / (R.max() - R.min())
        vals = R.values
        dep = Config.DATA_DEPTH
        D = np.hstack([vals[i:-dep+i-1, :] for i in range(dep,0,-1)])
        data_frame = pd.DataFrame(D, R[dep:-1].index)
        self.data = data_frame.values
        #close_prices
        self.prices = self.data[:,3]
        self.iter = 0
        self.n_shares = 0
        self.cash = 0
        self.max_shares = 1
        self.max_iter = self.prices.shape[0]
        self.done = False
        self.prev_equity = 0
        self.equity = 0
        self.comission = 1e-3
        # Штраф за повторы
        self.same_steps = 0
        self.prev_act = 0
        self.action_space_n = 3
        self.mean = 0
        self.dev = 0
        self.prev_sharpe = 0

    def calc_reward(self, act):
        # Действие act is -1(sell) 0 (nothing) and +1(buy)
        #if(act != self.n_shares
        #if abs(self.n_shares + act) <= self.max_shares:
        if(self.n_shares != act):
            self.cash = self.cash - self.prices[self.iter-1]*(act - self.n_shares) - self.comission*(1 + 1*(self.same_steps < 3))
        self.n_shares = act
        # Эквити - суммарный объем денег, если сейчас все продать
        self.equity = self.cash + self.prices[self.iter]*self.n_shares
        reward = self.equity - self.prev_equity
        self.prev_equity = self.equity
        #Магические константы - штраф равен 0.01% за ход на 10 одинаковых действий
        if Config.SHARPE:
            n = self.iter + 1 
            if n = 1:
                self.mean = 0
                self.dev = 0
                self.prev_sharpe = 0
            A = float(n)/(n+1)
            new_mean = self.mean*A + self.equity/(n+1)
            delta = (new_mean-self.mean)**2
            new_dev = A*(self.dev + delta) + ((self.equity-new_mean)**2)/(n+1)
            self.shape = new_mean/np.sqrt(new_dev)
            if n<5:
                #Для СТАБИЛЬНОСТИ сначала копим данные
                self.sharpe = 0
            reward = self.sharpe - self.prev_sharpe 
            self.prev_sharpe = self.sharpe
        return reward - self.comission*(self.same_steps/1000)
    
    def step(self, act):
        # Один шаг системы - получить на вход act = [-1,0,1][a]
        # Если не конец игры:
        if self.done == False:
            self.iter += 1
        # Извлечь следующие наблюдения
        # Состояние системы - одно число self.iter
        observation = self.data[self.iter]
        reward = self.calc_reward(act)
        #Считаем число одинаковых действий подряд
        self.same_steps += 1
        if act != self.prev_act:
            self.same_steps = 0
            
        if self.iter >= self.max_iter-1:
            self.done = True
        else:
            self.done = False
        info = 'lol'
        self.prev_act = act
        return observation, reward, self.done, self.n_shares
                
    def reset(self):
        self.iter = 0
        self.done = False
        #start_point = int(np.random.rand()*self.start_len)
        #self.data = self.big_data[start_point:start_point+self.len, :]
        observation = self.data[self.iter]
        self.prices = self.data[:,3]
        self.n_shares = 0
        self.cash = 0
        self.prev_equity = 0
        self.equity = 0
        return observation
    
    # Генерирует shifted_act
    def sample(self):
        return np.random.randint(0,3)   

    def render(self):
        return



class GameManager:
    def __init__(self, game_name, display):
        self.game_name = game_name
        self.display = display

        self.env = environment(game_name)
        self.reset()

    def reset(self):
        observation = self.env.reset()
        return observation

    def step(self, action):
        self._update_display()
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def _update_display(self):
        if self.display:
            self.env.render()

