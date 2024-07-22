# Проект ModelLSTM

## Описание
`ModelLSTM` - это класс для анализа временных рядов с использованием модели LSTM (долгая краткосрочная память). Проект включает функции для загрузки и предобработки данных, визуализации, разложения временных рядов, тестирования на стационарность, создания и оптимизации модели LSTM, а также визуализации прогнозов модели.

## Установка и зависимости
Для работы с проектом необходимо установить следующие библиотеки:

```bash
pip install -r requirements.txt
```

## Использование

### Инициализация
```python
model = ModelLSTM('Time series.csv', 'close')
```

### Загрузка и предобработка данных
```python
df = model.df_download()
```

### Визуализация данных
```python
model.show_Table(df)
model.show_subplots(df)
model.show_candlestick(df)
```

### Разложение временного ряда
```python
model.decompose(df, 30)
```

### Тестирование на стационарность и гетероскедастичность
```python
model.show_tests(df)
```

### Подготовка данных
```python
train_df, scaler, train_data, valid_data, valid_df, scaled_data, new_df = model.data_preparation(df, 1000)
```

### Оптимизация гиперпараметров модели LSTM
```python
model.evolution_models(scaled_data, new_df, valid_data, scaler, train_df, train_data, valid_df)
```

### Создание модели с лучшими гиперпараметрами
```python
valid_df = model.best_model(scaled_data, new_df, scaler, valid_data, valid_df)
```

### Визуализация прогнозов
```python
model.show_predictions_with_initial_data(train_df, valid_df)
model.show_predictions(valid_df)
```

## Методы

### `df_download()`
Загружает датафрейм из файла и обрабатывает пропуски.

### `show_Table(df)`
Показывает таблицу с загруженными значениями.

### `show_subplots(df)`
Выводит графики OHLC.

### `decompose(df, sel_period)`
Разлагает временной ряд на тренд, цикличность и шум.

### `show_tests(df)`
Проводит тесты на гетероскедастичность и стационарность, строит графики ACF и PACF.

### `show_candlestick(df)`
Показывает свечной график.

### `data_preparation(df, n)`
Подготавливает тренировочную, валидационную и тестовую выборки.

### `create_dataset(data, time_step)`
Создает датасет с шагом.

### `objective(trial, train_df, scaler, train_data, valid_data, valid_df, scaled_data, new_df)`
Создает и обучает модель LSTM для оптимизации гиперпараметров.

### `evolution_models(scaled_data, new_df, valid_data, scaler, train_df, train_data, valid_df)`
Оптимизирует модель с различными гиперпараметрами.

### `best_model(scaled_data, new_df, scaler, valid_data, valid_df)`
Создает модель с лучшими значениями гиперпараметров.

### `show_predictions_with_initial_data(train_df, valid_df)`
Показывает график сравнения предсказаний модели и реальных данных.

### `show_predictions(valid_df)`
Показывает график сравнения предсказаний модели и реальных данных.

## Пример использования

```python
if __name__ == "__main__":
    model = ModelLSTM('Time series.csv', 'close')
    df = model.df_download()
    model.show_Table(df)
    model.show_subplots(df)
    model.decompose(df, 30)
    model.show_tests(df)
    model.show_candlestick(df)
    train_df, scaler, train_data, valid_data, valid_df, scaled_data, new_df = model.data_preparation(df, 1000)
    model.evolution_models(scaled_data, new_df, valid_data, scaler, train_df, train_data, valid_df)
    valid_df = model.best_model(scaled_data, new_df, scaler, valid_data, valid_df)
    model.show_predictions_with_initial_data(train_df, valid_df)
    model.show_predictions(valid_df)
```

