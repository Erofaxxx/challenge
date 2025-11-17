# Промпт для создания Mamba-2 решения

Используйте этот промпт в новом чате для создания решения с нуля.

---

Мне нужно создать решение на архитектуре Mamba-2 для соревнования по предсказанию временных рядов. Важно точно следовать формату и правильно понимать данные.

## 1. КРИТИЧЕСКИ ВАЖНО: Структура данных

Данные организованы как **независимые последовательности**, НЕ как сплошной поток!

**Формат данных (train.parquet):**
- Таблица с колонками: `seq_ix`, `step_in_seq`, `need_prediction`, `f_0`, `f_1`, ..., `f_N`
- `seq_ix`: ID последовательности (например, 500 уникальных значений = 500 последовательностей)
- `step_in_seq`: шаг внутри последовательности (от 0 до 999, всегда 1000 шагов)
- `need_prediction`: True если нужно делать предсказание для следующего шага
- `f_0` до `f_N`: анонимизированные признаки (примерно 20-30 фичей)

**Ключевые свойства:**
1. Каждая последовательность **полностью независима** от других
2. Каждая последовательность содержит **ровно 1000 шагов** (0-999)
3. Последовательности **перемешаны** в датасете (seq_ix не идут подряд)
4. Внутри последовательности строки **упорядочены** по step_in_seq

**Warm-up и scoring:**
- Шаги 0-99 (100 шагов): **warm-up период** - используем для накопления состояния, НЕ делаем предсказания
- Шаги 100-999 (900 шагов): **scored predictions** - делаем предсказания, они оцениваются
- `need_prediction == True` только для шагов, где нужно предсказать следующий шаг

**Что это означает для модели:**
- При смене `seq_ix` нужно **полностью сбрасывать** состояние модели
- Нормализация должна быть **per-sequence** (каждая последовательность нормализуется независимо)
- Feature engineering тоже должен сбрасываться при смене последовательности

## 2. Формат решения (ОБЯЗАТЕЛЬНЫЙ)

Решение должно быть в файле `solution.py` с классом `PredictionModel`:

```python
from utils import DataPoint
import numpy as np
from typing import Optional

class PredictionModel:
    def __init__(self):
        # Инициализация модели
        pass

    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        """
        Делает предсказание для следующего шага

        Args:
            data_point: объект с полями:
                - state: np.ndarray - текущие признаки (shape: [N,])
                - seq_ix: int - ID последовательности
                - step_in_seq: int - шаг в последовательности (0-999)
                - need_prediction: bool - нужно ли предсказание

        Returns:
            np.ndarray - предсказание следующего состояния (shape: [N,])
                        или None если need_prediction == False
        """
        # КРИТИЧНО: проверить смену последовательности!
        if self.current_seq_ix != data_point.seq_ix:
            self._reset_sequence()  # Сбросить ВСЁ состояние

        # Обработка данных...

        if not data_point.need_prediction:
            return None  # Во время warm-up

        # Делаем предсказание
        return prediction  # np.ndarray[N]
```

## 3. Требования к архитектуре Mamba-2

**Архитектура:**
- Selective State Space Model (SSM) - ключевая особенность Mamba
- Несколько слоёв Mamba-2 блоков (4-6 слоёв)
- Residual connections
- Layer Normalization
- Gating mechanisms

**Mamba-2 Block должен включать:**
- Convolutional layer для локального контекста
- SSM с learnable parameters (A, B, C, D, delta)
- Gating для контроля информационного потока
- Output projection

**Оптимизация под ресурсы (16GB RAM, 8 CPU):**
- d_model: 128-192 (НЕ больше!)
- n_layers: 4-6
- batch_size: 32-64
- Использовать float32, НЕ float64
- Memory cleanup после каждой эпохи

## 4. Feature Engineering

**Обязательные фичи:**
1. Raw features (исходные признаки)
2. Delta features (первая разность: state[t] - state[t-1])
3. Acceleration (вторая разность)
4. Multi-window moving averages (окна: 3, 5, 10, 20, 50)
5. Standard deviations для каждого окна
6. Min/Max за каждое окно
7. Exponential Moving Averages (разные decay rates)

**ВАЖНО:**
- Feature engineer должен **сбрасываться** при смене последовательности
- Использовать `collections.deque` с `maxlen` для эффективного хранения истории

## 5. Нормализация

**Per-Sequence Normalization (КРИТИЧНО!):**
```python
class SequenceNormalizer:
    def reset(self):
        """Сбрасываем статистики для новой последовательности"""
        self.mean = None
        self.std = None
        self.count = 0

    def update_and_normalize(self, x):
        """Обновляем статистики и нормализуем за один проход"""
        # Running mean и std для ЭТОЙ последовательности
        ...
```

НЕ использовать глобальную нормализацию!

## 6. Обучение модели

**Подготовка данных:**
```python
def prepare_data(data_path):
    # Для КАЖДОЙ последовательности отдельно:
    for seq_id in unique_sequences:
        seq_data = df[df['seq_ix'] == seq_id].sort_values('step_in_seq')
        states = seq_data.iloc[:, 3:].values  # Все признаки

        # Создать НОВЫЙ feature engineer и normalizer для ЭТОЙ последовательности
        feat_eng = FeatureEngineer()
        normalizer = SequenceNormalizer()

        # Обработать все 1000 шагов
        for step in range(1000):
            eng_features = feat_eng.engineer_features(states[step])
            norm_features = normalizer.update_and_normalize(eng_features)

        # Создать обучающие примеры
        # ВАЖНО: начинаем с i=99 (предсказываем шаг 100)
        for i in range(99, 999):  # i от 99 до 998
            X = normalized_features[i-context_len:i+1]  # Контекст до шага i
            y = states[i+1]  # Целевой шаг (raw, не engineered!)
            # При i=99: контекст до шага 99, предсказываем шаг 100 (первый после warm-up)
            # При i=998: контекст до шага 998, предсказываем шаг 999 (последний)
```

**Параметры обучения:**
- Batch size: 32-64
- Learning rate: 0.0003-0.0005
- Optimizer: AdamW с weight_decay=1e-4
- Scheduler: CosineAnnealingWarmRestarts
- Epochs: 20-30 (с early stopping)
- Gradient clipping: 1.0

**Ансамбль:**
- Обучить 3-5 моделей с разными random seeds
- Усреднять предсказания

## 7. Метрика

R² score (coefficient of determination):
```
R²_i = 1 - SS_res_i / SS_tot_i
Final Score = mean(R²_i for all features)
```

Чем ближе к 1, тем лучше. Целевой R² ≥ 0.39

## 8. Структура файлов

```
advanced_solution/
├── solution.py          # PredictionModel + Mamba2Model
├── train.py            # Скрипт обучения
├── requirements.txt    # Зависимости
├── README.md          # Инструкции
└── mamba_model_*.pt   # Обученные веса (после обучения)
```

## 9. Инструкции для запуска

Создай детальный README с:

**Установка:**
```bash
cd advanced_solution
pip install -r requirements.txt
```

**Обучение:**
```bash
# Базовое обучение
python train.py --data ../datasets/train.parquet

# С параметрами
python train.py \
    --num-models 3 \
    --d-model 192 \
    --n-layers 4 \
    --batch-size 64 \
    --epochs 25 \
    --lr 0.0005
```

**Тестирование:**
```bash
# Тест без обученных моделей (baseline)
python solution.py

# С обученными моделями
python test_solution.py
```

**Создание submission:**
```bash
bash prepare_submission.sh
# Создаст submission.zip с solution.py и mamba_model_*.pt
```

## 10. Чек-лист для проверки

Убедись что решение:
- ✅ Сбрасывает состояние при смене seq_ix
- ✅ Использует per-sequence нормализацию
- ✅ Обрабатывает все 1000 шагов каждой последовательности
- ✅ Создаёт 900 обучающих примеров на последовательность (шаги 100-999)
- ✅ Возвращает None во время warm-up (need_prediction == False)
- ✅ Возвращает np.ndarray[base_dim] когда need_prediction == True
- ✅ Загружает обученные веса из mamba_model_*.pt
- ✅ Работает в CPU-only режиме
- ✅ Умещается в 16GB RAM

## 11. Важные детали реализации

**SSM scan operation:**
```python
def _ssm_scan(self, x, delta, A, B, C):
    """
    Simplified SSM scan - ядро Mamba-2
    x: входная последовательность
    delta: адаптивный time step (learnable)
    A, B, C: параметры state space model
    """
    # Рекуррентное обновление скрытого состояния
    # h_t = A * h_{t-1} + B * x_t
    # y_t = C * h_t
```

**Memory efficiency:**
```python
import gc

# После обработки каждой последовательности
del engineered_states
gc.collect()

# После каждой эпохи
if epoch % 5 == 0:
    gc.collect()
```

## Задание

Создай полное решение с:
1. `solution.py` - Mamba-2 модель с правильной обработкой последовательностей
2. `train.py` - обучающий скрипт с per-sequence обработкой
3. `requirements.txt` - только необходимые зависимости
4. `README.md` - детальные инструкции по установке, обучению, тестированию
5. `test_solution.py` - для локального тестирования

Используй PyTorch для реализации. Код должен быть хорошо документирован с комментариями.

**Особое внимание удели:**
- Правильной обработке независимых последовательностей (сброс состояния!)
- Per-sequence нормализации (НЕ глобальной!)
- Эффективному использованию памяти (16GB RAM, 8 CPU)
- Правильному warm-up периоду (0-99: накопление, 100-999: предсказания)

**Проверь что:**
- При i=99: предсказываем шаг 100 (первый scored prediction)
- При i=998: предсказываем шаг 999 (последний scored prediction)
- Всего 900 предсказаний на последовательность

---

## Дополнительные подсказки

Если возникнут проблемы:

**Проблема: Слишком много памяти**
- Уменьши d_model до 128
- Уменьши batch_size до 32
- Уменьши context_length до 60

**Проблема: Низкий R² score**
- Увеличь количество моделей в ансамбле (5-7)
- Добавь больше feature engineering
- Увеличь n_layers до 6
- Попробуй разные lookback windows

**Проблема: Долгое обучение**
- Используй меньше последовательностей для валидации (10%)
- Уменьши epochs до 20
- Используй более агрессивный early stopping (patience=5)
