import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
from plotly.graph_objs import Line
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss
import plotly.graph_objects as go
import pandas as pd
from functools import partial

class ModelLSTM():
    n = 1000
    def __init__(self, path, target):
        self.path=path
        self.target=target
    def df_download(self):
        """
        Загрузить датафрейм из файла и обработать пропуски
        """
        df = pd.read_csv(f'{self.path}')
        df['date'] = df['date'].apply(pd.to_datetime)
        df.set_index('date', inplace=True)
        for column in ['open', 'high', 'low']:
            df[column] = df[column].where(df[column] != 0, df[column].shift())
        na_counts = df.isna().sum()
        print(na_counts)
        return df

    def show_Table(self, df):
        """
        Показать таблицу с загруженными значениями
        :param df: Датафрейм с данными
        """

        column_names = list(df.columns)
        column_names.insert(0, 'date')
        column_data = [df.index.tolist()] + [df[col].tolist() for col in df.columns]
        fig = go.Figure(data=[go.Table(
            header=dict(values=column_names,
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=column_data,
                       fill_color='lavender',
                       align='left'))
        ])

        fig.show()

    def show_subplots(self, df):
        """
        Вывести график OHLС
        :param df: Датафрейм с данными
        """

        columns = list(df.columns)
        n = len(columns)
        fig = make_subplots(rows=n, cols=1, subplot_titles=df.columns)
        for i in range(n):
            column = columns[i]
            fig.add_trace(
                Line(x=df.index, y=df[column]),
                row=i + 1, col=1
            )

        fig.update_layout(height=1400, width=1000, title_text="OHLC Line Plots")

        fig.show()

    def decompose(self, df, sel_period:int):
        """
        Разложение на тренд, цикличность, шум
        :param df: Датафрейм с данными
        :param sel_period: Период разложения ряда
        """

        result = seasonal_decompose(df[self.target], model='additive', period=sel_period)
        fig = go.Figure()
        fig = result.plot()
        fig.set_size_inches(20, 19)

    def show_tests(self, df):
        """
        Построить графики ACF, PACF. Провести тесты на гетероскедастичность, на стационарность
        :param df: Датафрейм с данными
        """

        plot_acf(df[self.target], lags=50)
        plt.show()

        plot_pacf(df[self.target], lags=50)
        plt.show()

        arch_test = het_arch(df[self.target])
        print('\nARCH Test Statistic:', arch_test[0])
        print('p-value:', arch_test[1], '\n')

        result = adfuller(df[self.target])
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])

        for key, value in result[4].items():
            print('Critial Values:')
            print(f'   {key}, {value}')

        result = kpss(df[self.target], regression='c')
        print('\nKPSS Statistic:', result[0])
        print('p-value:', result[1])

        for key, value in result[3].items():
            print('Critical Values:')
            print(f'   {key}, {value}')

    def show_candlestick(self, df):
        """
        Построить свечной график

        :param df:Датафрейм с данными
        """
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['open'],
                                             high=df['high'],
                                             low=df['low'],
                                             close=df['close'])])

        fig.show()

    def data_preparation(self, df, n: int):
        """
        Подготовка тренировочной, валидационной и тестовой выборок

        :param df: Датафрейм с данными
        :param n: Количество данных для обучения
        :return:
        train_df: Датафрейм тренировочных данных
        scaler:Это объект, который используется для нормализации данных.
        train_data: Это массив данных, который используется для обучения модели.
        valid_data: Это массив данных, который используется для валидации модели.
        valid_df: Это датафрейм, содержащий данные для валидации.
        scaled_data:  Это нормализованный массив данных final_dataset.
        new_df: Это датафрейм, который содержит только столбец целевой переменной (target) из исходного датафрейма df
        """

        new_df = pd.DataFrame()
        new_df = df[self.target]
        new_df.index = df.index

        scaler = MinMaxScaler(feature_range=(0, 1))

        final_dataset = new_df.values
        train_data = final_dataset[0:n, ]
        valid_data = final_dataset[n:, ]

        train_df = pd.DataFrame()
        valid_df = pd.DataFrame()

        train_df['Close'] = train_data
        train_df.index = new_df[0:n].index
        valid_df['Close'] = valid_data
        valid_df.index = new_df[n:].index

        scaled_data = scaler.fit_transform(final_dataset.reshape(-1, 1))

        return train_df, scaler, train_data, valid_data, valid_df, scaled_data, new_df

    def create_dataset(self, data, time_step=1):

        """
        Создание датасета с шагом

        :param data: Входные данные
        :param time_step: Шаг для датасета
        """

        X, Y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    def objective(self,trial,train_df, scaler, train_data, valid_data, valid_df, scaled_data, new_df):
        """
        Построить модель LSTM для дальнейшего процесса оптимизации гиперпараметров
        :param trial: Объект, предоставляемый библиотекой Optuna для выполнения одного эксперимента в процессе оптимизации гиперпараметров.
        :param train_df: Датафрейм тренировочных данных
        :param scaler: Это объект, который используется для нормализации данных.
        :param train_data: Это массив данных, который используется для обучения модели.
        :param valid_data: Это массив данных, который используется для валидации модели.
        :param valid_df: Это датафрейм, содержащий данные для валидации.
        :param scaled_data: Это нормализованный массив данных final_dataset.
        :param new_df: Это датафрейм, который содержит только столбец целевой переменной (target) из исходного датафрейма df
        :return: r2: Значение коэффициента детерминации для модели
        """
        time_step = trial.suggest_int('time_step', 30, 60)
        units = trial.suggest_int('units', 50, 100)
        batch_size = trial.suggest_int('batch_size', 7, 14)
        epochs = trial.suggest_int('epochs', 5, 20)

        x_train_data, y_train_data = self.create_dataset(scaled_data[:1000], time_step)
        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

        # Define and compile the LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        lstm_model.add(LSTM(units=units, return_sequences=False))
        lstm_model.add(Dense(25))
        lstm_model.add(Dense(1))

        lstm_model.compile(loss='mean_squared_error', optimizer='adam')

        # Fit the model
        lstm_model.fit(x_train_data, y_train_data, epochs=epochs, batch_size=batch_size, verbose=0)

        # Prepare test data
        inputs_data = new_df[len(new_df) - len(valid_data) - time_step:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)

        X_test, _ = self.create_dataset(inputs_data, time_step)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Predict and evaluate
        predicted_closing_price = lstm_model.predict(X_test)
        predicted_closing_price = scaler.inverse_transform(predicted_closing_price)
        # Ensure length of predictions matches the length of valid_df
        valid_df['Predictions'] = predicted_closing_price
        r2 = r2_score(valid_df['Close'], valid_df['Predictions'])

        return r2

    def evolution_models(self, scaled_data, new_df, valid_data, scaler,train_df, train_data, valid_df):
        """
        Оптимизация моделей с разными гиперпараметрами
        :param scaled_data: Это нормализованный массив данных final_dataset.
        :param new_df: Это датафрейм, который содержит только столбец целевой переменной (target) из исходного датафрейма df
        :param valid_data: Это массив данных, который используется для валидации модели.
        :param scaler: Это объект, который используется для нормализации данных.
        :param train_df: Датафрейм тренировочных данных
        :param train_data: Это массив данных, который используется для обучения модели.
        :param valid_df: Это датафрейм, содержащий данные для валидации.
        """
        self.study = optuna.create_study(direction='maximize')
        objective_with_args = partial(self.objective, scaled_data=scaled_data, new_df=new_df, valid_data=valid_data,
                                    scaler=scaler, train_df=train_df,train_data=train_data, valid_df=valid_df)
        self.study.optimize(objective_with_args, n_trials=50)

        print(f"Best parameters: {self.study.best_params}")
        print(f"Best R^2 score: {self.study.best_value}")

    def best_model(self, scaled_data,new_df, scaler, valid_data, valid_df):
        """
        Создание модели с лучшими значениями гиперпараметров
        :param scaled_data: Это нормализованный массив данных final_dataset.
        :param new_df: Это датафрейм, который содержит только столбец целевой переменной (target) из исходного датафрейма df
        :param scaler: Это объект, который используется для нормализации данных.
        :param valid_data: Это массив данных, который используется для валидации модели.
        :param valid_df: Это датафрейм, содержащий данные для валидации.
        :return: valid_df:Это датафрейм, содержащий данные для валидации и прогнозные значения
        """
        best_params = self.study.best_params
        time_step = best_params['time_step']
        units = best_params['units']
        batch_size = best_params['batch_size']
        epochs = best_params['epochs']

        x_train_data, y_train_data = self.create_dataset(scaled_data[:1000], time_step)
        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

        lstm_model = Sequential()
        lstm_model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        lstm_model.add(LSTM(units=units, return_sequences=False))
        lstm_model.add(Dense(25))
        lstm_model.add(Dense(1))

        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(x_train_data, y_train_data, epochs=epochs, batch_size=batch_size, verbose=2)

        inputs_data = new_df[len(new_df) - len(valid_data) - time_step:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)

        X_test, _ = self.create_dataset(inputs_data, time_step)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_closing_price = lstm_model.predict(X_test)
        predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

        valid_df['Predictions'] = predicted_closing_price
        print('Final R^2 score is', r2_score(valid_df['Close'], valid_df['Predictions']))
        print('Final Mean squared error score is', mean_squared_error(valid_df['Close'], valid_df['Predictions']))
        return valid_df

    def show_predictions_with_initial_data(self, train_df, valid_df):
        """
        График сравнения предсказаний модели и  реальных данных
        :param train_df: Датафрейм тренировочных данных
        :param valid_df: Это датафрейм, содержащий данные для валидации и прогнозные значения
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'],
                                 mode='lines',
                                 name='Train Data'))
        fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df['Close'],
                                 mode='lines',
                                 name='Valid Data'))
        fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df['Predictions'],
                                 mode='lines',
                                 name='Prediction'))
        fig.show()

    def show_predictions(self, valid_df):
        """
        График сравнения предсказаний модели и  реальных данных
        :param valid_df: Это датафрейм, содержащий данные для валидации и прогнозные значения
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df['Close'],
                                 mode='lines',
                                 name='Test'))
        fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df['Predictions'],
                                 mode='lines',
                                 name='Predicted'))
        fig.show()


if __name__ == "__main__":

    model = ModelLSTM('Time series.csv', 'close')
    df = model.df_download()
    model.show_Table(df)
    model.show_subplots(df)
    model.decompose(df, 30)
    model.show_tests(df)
    model.show_candlestick(df)
    train_df, scaler, train_data, valid_data, valid_df, scaled_data, new_df = model.data_preparation(df,1000)
    model.evolution_models(scaled_data, new_df, valid_data, scaler,train_df, train_data, valid_df)
    valid_df = model.best_model(scaled_data, new_df, scaler, valid_data, valid_df)
    model.show_predictions_with_initial_data(train_df, valid_df)
    model.show_predictions(valid_df)



