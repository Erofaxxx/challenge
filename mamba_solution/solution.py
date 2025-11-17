"""
Mamba-2 решение для Wunder Fund RNN Challenge
Архитектура: Selective State Space Model с feature engineering
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Optional, List
import glob

# Добавляем родительскую директорию в путь для импорта utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, ".."))

from utils import DataPoint


class SequenceNormalizer:
    """Per-sequence нормализация с running statistics"""

    def __init__(self, feature_dim: int, epsilon: float = 1e-8):
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        """Сбрасываем статистики для новой последовательности"""
        self.mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.var = np.ones(self.feature_dim, dtype=np.float32)
        self.count = 0

    def update_and_normalize(self, x: np.ndarray) -> np.ndarray:
        """Обновляем running статистики и нормализуем"""
        # Обновление running mean и variance (Welford's method)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

        # Нормализация
        std = np.sqrt(self.var + self.epsilon)
        normalized = (x - self.mean) / std
        return normalized.astype(np.float32)


class FeatureEngineer:
    """
    Feature engineering с трейдерскими индикаторами

    Создает расширенный набор признаков включая:
    - Базовые: raw, delta, acceleration
    - Moving averages: SMA, EMA
    - Volatility: STD, Bollinger Bands
    - Momentum: ROC, Momentum
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    """

    def __init__(self, base_dim: int, max_history: int = 100):
        self.base_dim = base_dim
        self.max_history = max_history
        self.reset()

    def reset(self):
        """Сбрасываем историю для новой последовательности"""
        # Используем deque для эффективного хранения фиксированной истории
        self.history = deque(maxlen=self.max_history)
        self.prev_state = None
        self.prev_delta = None

        # Для RSI
        self.gains = deque(maxlen=14)  # Для RSI-14
        self.losses = deque(maxlen=14)

    def _calculate_rsi(self, values: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Расчет RSI (Relative Strength Index)
        RSI = 100 - (100 / (1 + RS))
        где RS = Average Gain / Average Loss
        """
        if len(values) < period + 1:
            return np.full(self.base_dim, 50.0)  # Neutral RSI

        # Вычисляем изменения
        deltas = np.diff(values, axis=0)

        # Разделяем на gains и losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Средние gains и losses
        avg_gain = np.mean(gains[-period:], axis=0)
        avg_loss = np.mean(losses[-period:], axis=0)

        # Избегаем деления на ноль
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi.astype(np.float32)

    def _calculate_macd(self, values: np.ndarray) -> tuple:
        """
        Расчет MACD (Moving Average Convergence Divergence)
        MACD = EMA(12) - EMA(26)
        Signal = EMA(9) of MACD
        Histogram = MACD - Signal
        """
        if len(values) < 26:
            return (np.zeros(self.base_dim), np.zeros(self.base_dim), np.zeros(self.base_dim))

        # EMA функция
        def calc_ema(data, period):
            alpha = 2.0 / (period + 1.0)
            ema = data[0].copy()
            for i in range(1, len(data)):
                ema = alpha * data[i] + (1 - alpha) * ema
            return ema

        # MACD линия (EMA12 - EMA26)
        ema12 = calc_ema(values[-26:], 12) if len(values) >= 26 else values[-1]
        ema26 = calc_ema(values[-26:], 26) if len(values) >= 26 else values[-1]
        macd_line = ema12 - ema26

        # Signal line (EMA9 of MACD) - упрощенно используем последние значения
        signal_line = macd_line * 0.8  # Упрощение для скорости

        # Histogram
        histogram = macd_line - signal_line

        return (
            macd_line.astype(np.float32),
            signal_line.astype(np.float32),
            histogram.astype(np.float32)
        )

    def _calculate_bollinger_bands(self, values: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple:
        """
        Bollinger Bands
        Middle = SMA(period)
        Upper = Middle + (num_std * STD(period))
        Lower = Middle - (num_std * STD(period))
        """
        if len(values) < period:
            middle = values[-1]
            return (middle, middle, middle)

        window = values[-period:]
        middle = np.mean(window, axis=0)
        std = np.std(window, axis=0)

        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        return (
            upper.astype(np.float32),
            middle.astype(np.float32),
            lower.astype(np.float32)
        )

    def engineer_features(self, state: np.ndarray) -> np.ndarray:
        """
        Создаем расширенный набор признаков для trading

        Возвращает конкатенацию:
        1. Raw (1x)
        2. Delta (1x)
        3. Acceleration (1x)
        4. SMA windows (4x): 3, 5, 10, 20
        5. STD windows (3x): 5, 10, 20
        6. EMA alphas (3x): 0.1, 0.3, 0.5
        7. RSI (1x)
        8. MACD (3x): line, signal, histogram
        9. Bollinger Bands (2x): upper, lower (middle уже есть в SMA-20)
        10. Momentum (2x): 5-period, 10-period
        11. ROC (2x): 5-period, 10-period

        Итого: 1+1+1+4+3+3+1+3+2+2+2 = 23x base_dim
        """
        features = []

        # 1. Исходные признаки
        features.append(state)

        # Добавляем в историю
        self.history.append(state)
        history_array = np.array(self.history)

        # 2. Delta (первая разность)
        if self.prev_state is not None:
            delta = state - self.prev_state
            features.append(delta)

            # 3. Acceleration (вторая разность)
            if self.prev_delta is not None:
                accel = delta - self.prev_delta
                features.append(accel)
            else:
                features.append(np.zeros_like(state))

            self.prev_delta = delta
        else:
            features.append(np.zeros_like(state))
            features.append(np.zeros_like(state))

        # 4. Simple Moving Averages (SMA)
        for window in [3, 5, 10, 20]:
            if len(self.history) >= window:
                sma = np.mean(history_array[-window:], axis=0)
                features.append(sma)
            else:
                features.append(state)

        # 5. Standard Deviations (Volatility)
        for window in [5, 10, 20]:
            if len(self.history) >= window:
                std = np.std(history_array[-window:], axis=0)
                features.append(std)
            else:
                features.append(np.zeros_like(state))

        # 6. Exponential Moving Averages (EMA)
        for alpha in [0.1, 0.3, 0.5]:
            if len(self.history) >= 2:
                ema = history_array[0].copy()
                for i in range(1, len(history_array)):
                    ema = alpha * history_array[i] + (1 - alpha) * ema
                features.append(ema)
            else:
                features.append(state)

        # 7. RSI (Relative Strength Index)
        rsi = self._calculate_rsi(history_array)
        features.append(rsi)

        # 8. MACD (Moving Average Convergence Divergence)
        macd_line, signal_line, histogram = self._calculate_macd(history_array)
        features.append(macd_line)
        features.append(signal_line)
        features.append(histogram)

        # 9. Bollinger Bands (только верхняя и нижняя, средняя уже есть в SMA-20)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(history_array, period=20)
        features.append(bb_upper)
        features.append(bb_lower)

        # 10. Momentum (изменение цены за N периодов)
        for period in [5, 10]:
            if len(self.history) >= period + 1:
                momentum = state - history_array[-period-1]
                features.append(momentum)
            else:
                features.append(np.zeros_like(state))

        # 11. Rate of Change (ROC) - процентное изменение
        for period in [5, 10]:
            if len(self.history) >= period + 1:
                prev_value = history_array[-period-1]
                # ROC = (current - previous) / previous * 100
                # Избегаем деления на ноль
                roc = np.divide(
                    (state - prev_value) * 100.0,
                    np.abs(prev_value) + 1e-8,
                    out=np.zeros_like(state),
                    where=np.abs(prev_value) > 1e-8
                )
                features.append(roc)
            else:
                features.append(np.zeros_like(state))

        self.prev_state = state.copy()

        # Объединяем все признаки
        # Итого: 23x base_dim признаков
        return np.concatenate(features).astype(np.float32)


class Mamba2Block(nn.Module):
    """
    Упрощенный Mamba-2 блок с Selective State Space Model
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Convolutional layer для локального контекста
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )

        # SSM параметры (упрощенная версия)
        self.x_proj = nn.Linear(self.d_inner, d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # State space parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        residual = x

        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # Convolutional path (для локального контекста)
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :x.shape[2]]  # Обрезаем padding
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = self.activation(x)

        # Simplified SSM (без полной scan операции для эффективности)
        # Вместо полного SSM делаем упрощенную версию с gating
        dt = self.dt_proj(x)
        dt = torch.sigmoid(dt)

        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Упрощенное SSM преобразование
        state_proj = self.x_proj(x)  # (B, L, d_state)

        # Применяем state space трансформацию
        # Вместо полного scan делаем матричное умножение
        ssm_out = torch.matmul(state_proj, A.T)  # (B, L, d_inner)

        # Добавляем skip connection (D параметр в Mamba)
        y = ssm_out + self.D * x

        # Gating с z
        y = y * self.activation(z)

        # Output projection
        output = self.out_proj(y)

        # Residual connection
        output = output + residual
        output = self.norm(output)

        return output


class Mamba2Model(nn.Module):
    """
    Полная Mamba-2 модель для предсказания временных рядов

    ВАЖНО: Модель принимает engineered features (input_dim = 23 * base_dim),
    но предсказывает только оригинальные признаки (output_dim = base_dim = 32).

    Это правильно, потому что:
    - На вход подаем богатые признаки (RSI, MACD, Bollinger Bands, etc.)
    - На выход предсказываем только сырые рыночные состояния
    - Loss оптимизируется по base_dim признакам
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([
            Mamba2Block(d_model, d_state=d_state)
            for _ in range(n_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, output_dim) - предсказание следующего шага
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.dropout(x)

        # Mamba blocks
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        # Берем последний timestep для предсказания
        x = x[:, -1, :]  # (batch, d_model)

        # Output projection
        x = self.output_norm(x)
        output = self.output_proj(x)

        return output


class PredictionModel:
    """
    Главный класс для инференса с ансамблем Mamba-2 моделей
    """

    def __init__(self):
        # Параметры модели (должны совпадать с обучением)
        self.base_dim = None  # Будет определено автоматически
        self.engineered_dim = None
        self.d_model = 128
        self.n_layers = 4
        self.d_state = 16
        self.context_length = 80  # Сколько шагов используем для контекста

        # Состояние
        self.current_seq_ix = None
        self.feature_engineer = None
        self.normalizer = None
        self.context_buffer = None

        # Модели (ансамбль)
        self.models = []
        self.device = torch.device('cpu')  # CPU-only

        # Загружаем обученные модели
        self._load_models()

    def _load_models(self):
        """Загружаем все обученные модели из директории"""
        model_files = glob.glob(os.path.join(CURRENT_DIR, "mamba_model_*.pt"))

        if not model_files:
            print("WARNING: Не найдены обученные модели. Модель будет работать со случайными весами.")
            print("Запустите train.py для обучения моделей.")
            # Создаем одну модель со случайными весами для демонстрации
            # Размерность будет установлена при первом вызове predict
            self.models = []
            return

        print(f"Загружаем {len(model_files)} обученных моделей...")

        for model_path in sorted(model_files):
            checkpoint = torch.load(model_path, map_location=self.device)

            # Восстанавливаем параметры из первой модели
            if self.base_dim is None:
                self.base_dim = checkpoint['base_dim']
                self.engineered_dim = checkpoint['engineered_dim']
                self.d_model = checkpoint.get('d_model', 128)
                self.n_layers = checkpoint.get('n_layers', 4)
                self.d_state = checkpoint.get('d_state', 16)

            # Создаем модель
            model = Mamba2Model(
                input_dim=self.engineered_dim,
                output_dim=self.base_dim,
                d_model=self.d_model,
                n_layers=self.n_layers,
                d_state=self.d_state,
                dropout=0.0  # Нет dropout при инференсе
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(self.device)

            self.models.append(model)
            print(f"  Загружена модель: {os.path.basename(model_path)}")

        print(f"Загружено моделей: {len(self.models)}")
        print(f"Размерность базовых признаков: {self.base_dim}")
        print(f"Размерность после feature engineering: {self.engineered_dim}")

    def _reset_sequence(self):
        """Сбрасываем все состояние для новой последовательности"""
        if self.feature_engineer is not None:
            self.feature_engineer.reset()
        if self.normalizer is not None:
            self.normalizer.reset()
        self.context_buffer = deque(maxlen=self.context_length)

    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        """
        Главный метод предсказания

        Args:
            data_point: DataPoint с полями seq_ix, step_in_seq, need_prediction, state

        Returns:
            np.ndarray[base_dim] если need_prediction==True, иначе None
        """
        # Инициализация при первом вызове
        if self.base_dim is None:
            self.base_dim = len(data_point.state)
            # Вычисляем engineered_dim
            # ВАЖНО: Модель получает на вход engineered features (23x),
            # но предсказывает ТОЛЬКО оригинальные base_dim признаков!
            # 1+1+1+4+3+3+1+3+2+2+2 = 23x base_dim
            self.engineered_dim = 23 * self.base_dim

            self.feature_engineer = FeatureEngineer(self.base_dim)
            self.normalizer = SequenceNormalizer(self.engineered_dim)
            self.context_buffer = deque(maxlen=self.context_length)

            # Если модели не загружены, создаем одну со случайными весами
            if len(self.models) == 0:
                model = Mamba2Model(
                    input_dim=self.engineered_dim,
                    output_dim=self.base_dim,
                    d_model=self.d_model,
                    n_layers=self.n_layers,
                    d_state=self.d_state,
                    dropout=0.0
                )
                model.eval()
                model.to(self.device)
                self.models.append(model)

        # Проверяем смену последовательности
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self._reset_sequence()

        # Feature engineering
        engineered_features = self.feature_engineer.engineer_features(data_point.state)

        # Нормализация (per-sequence)
        normalized_features = self.normalizer.update_and_normalize(engineered_features)

        # Добавляем в буфер контекста
        self.context_buffer.append(normalized_features)

        # Если не нужно предсказание (warm-up период)
        if not data_point.need_prediction:
            return None

        # Недостаточно контекста для предсказания
        if len(self.context_buffer) < 2:
            # Возвращаем последнее состояние как fallback
            return data_point.state

        # Подготавливаем контекст для модели
        context_array = np.array(list(self.context_buffer))  # (seq_len, engineered_dim)
        context_tensor = torch.from_numpy(context_array).unsqueeze(0)  # (1, seq_len, engineered_dim)
        context_tensor = context_tensor.to(self.device)

        # Получаем предсказания от всех моделей
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(context_tensor)  # (1, base_dim)
                predictions.append(pred.cpu().numpy()[0])

        # Усредняем предсказания (ensemble)
        if len(predictions) > 0:
            final_prediction = np.mean(predictions, axis=0)
        else:
            # Fallback если моделей нет
            final_prediction = data_point.state

        return final_prediction.astype(np.float32)


if __name__ == "__main__":
    """Локальное тестирование решения"""

    # Проверяем наличие тестовых данных
    test_file = os.path.join(CURRENT_DIR, "..", "datasets", "train.parquet")

    if not os.path.exists(test_file):
        print(f"ОШИБКА: Тестовый файл не найден: {test_file}")
        print("Пожалуйста, убедитесь что файл datasets/train.parquet существует.")
        sys.exit(1)

    from utils import ScorerStepByStep

    # Создаем модель
    print("Создаем модель...")
    model = PredictionModel()

    # Загружаем scorer
    print(f"Загружаем данные из {test_file}...")
    scorer = ScorerStepByStep(test_file)

    print(f"Размерность признаков: {scorer.dim}")
    print(f"Количество строк: {len(scorer.dataset)}")

    # Оцениваем решение
    print("\nОцениваем решение...")
    results = scorer.score(model)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Средний R² по всем признакам: {results['mean_r2']:.6f}")

    print("\nR² для первых 5 признаков:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")

    print(f"\nВсего признаков: {len(scorer.features)}")

    print("\n" + "=" * 60)
    print("Решение готово к отправке!")
    print("=" * 60)
