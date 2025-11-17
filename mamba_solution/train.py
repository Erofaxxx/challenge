"""
Скрипт обучения Mamba-2 моделей
Обучает ансамбль из нескольких моделей с разными random seeds
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm
import gc
from collections import deque

# Добавляем текущую директорию для импорта solution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, ".."))

from solution import Mamba2Model, FeatureEngineer, SequenceNormalizer


class TimeSeriesDataset(Dataset):
    """
    Dataset для временных рядов с правильной обработкой последовательностей
    """

    def __init__(
        self,
        sequences_data: list,
        context_length: int = 40,
        prediction_start: int = 99,
        stride: int = 2
    ):
        """
        Args:
            sequences_data: список dict с ключами 'engineered_features', 'targets'
            context_length: длина контекста для модели
            prediction_start: с какого шага начинаем делать предсказания (99 = предсказываем шаг 100)
            stride: шаг сэмплирования (1=все шаги, 2=каждый 2-й, 3=каждый 3-й)
        """
        self.context_length = context_length
        self.prediction_start = prediction_start
        self.stride = stride
        self.samples = []

        # Создаем обучающие примеры из всех последовательностей
        for seq_data in sequences_data:
            engineered = seq_data['engineered_features']  # (1000, engineered_dim)
            targets = seq_data['targets']  # (1000, base_dim)

            # Создаем примеры: предсказываем шаги 100-999 с заданным stride
            # При i=99: контекст до шага 99, предсказываем шаг 100
            for i in range(prediction_start, len(engineered) - 1, stride):
                # Контекст: берем последние context_length шагов до i включительно
                start_idx = max(0, i - context_length + 1)
                context = engineered[start_idx:i + 1]  # (<=context_length, engineered_dim)

                # Если контекст короче чем нужно, паддим нулями в начале
                if len(context) < context_length:
                    padding = np.zeros((context_length - len(context), context.shape[1]), dtype=np.float32)
                    context = np.vstack([padding, context])

                target = targets[i + 1]  # Следующий шаг (raw features)

                self.samples.append({
                    'context': context.astype(np.float32),
                    'target': target.astype(np.float32)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.from_numpy(sample['context']),
            torch.from_numpy(sample['target'])
        )


def prepare_sequences(data_path: str, train_ratio: float = 0.8, max_sequences: int = None):
    """
    Загружаем и обрабатываем все последовательности

    Args:
        data_path: путь к файлу parquet
        train_ratio: доля последовательностей для обучения
        max_sequences: максимальное количество последовательностей (для экономии памяти)

    Returns:
        train_sequences, val_sequences: списки dict с обработанными последовательностями
    """
    print(f"Загружаем данные из {data_path}...")
    df = pd.read_parquet(data_path)

    print(f"Загружено строк: {len(df)}")
    print(f"Колонки: {list(df.columns)}")

    # Определяем размерности
    base_dim = df.shape[1] - 3  # Минус seq_ix, step_in_seq, need_prediction
    print(f"Размерность базовых признаков: {base_dim}")

    # Получаем уникальные последовательности
    unique_sequences = df['seq_ix'].unique()
    print(f"Количество последовательностей: {len(unique_sequences)}")

    # Ограничиваем количество если задано
    if max_sequences is not None and max_sequences < len(unique_sequences):
        print(f"⚠️  ОГРАНИЧЕНИЕ: Используем только {max_sequences} последовательностей (для экономии памяти)")
        unique_sequences = unique_sequences[:max_sequences]

    # Разделяем на train/val
    np.random.seed(42)
    np.random.shuffle(unique_sequences)
    split_idx = int(len(unique_sequences) * train_ratio)
    train_seq_ids = unique_sequences[:split_idx]
    val_seq_ids = unique_sequences[split_idx:]

    print(f"Train последовательностей: {len(train_seq_ids)}")
    print(f"Val последовательностей: {len(val_seq_ids)}")

    # Обрабатываем последовательности
    train_sequences = process_sequences(df, train_seq_ids, base_dim)
    val_sequences = process_sequences(df, val_seq_ids, base_dim)

    return train_sequences, val_sequences, base_dim


def process_sequences(df: pd.DataFrame, seq_ids: list, base_dim: int):
    """Обрабатываем список последовательностей"""
    sequences = []

    for seq_id in tqdm(seq_ids, desc="Обработка последовательностей"):
        # Получаем данные последовательности
        seq_data = df[df['seq_ix'] == seq_id].sort_values('step_in_seq')

        if len(seq_data) != 1000:
            print(f"WARNING: Последовательность {seq_id} имеет {len(seq_data)} шагов вместо 1000")
            continue

        # Извлекаем raw features (все колонки кроме первых 3)
        states = seq_data.iloc[:, 3:].values.astype(np.float32)  # (1000, base_dim)

        # Создаем feature engineer и normalizer для ЭТОЙ последовательности
        feat_eng = FeatureEngineer(base_dim)
        normalizer = SequenceNormalizer(12 * base_dim)  # engineered_dim (12x оптимизированный набор)

        # Обрабатываем все шаги
        engineered_features = []
        for step in range(1000):
            # Feature engineering
            eng_feat = feat_eng.engineer_features(states[step])

            # Per-sequence нормализация
            norm_feat = normalizer.update_and_normalize(eng_feat)

            engineered_features.append(norm_feat)

        engineered_features = np.array(engineered_features)  # (1000, engineered_dim)

        sequences.append({
            'engineered_features': engineered_features,
            'targets': states  # Целевые значения - raw features
        })

        # Очистка памяти
        del seq_data, states, feat_eng, normalizer

    gc.collect()
    return sequences


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 0.0005,
    save_path: str = None
):
    """Обучение одной модели"""

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0

        pbar = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{epochs} [Train]")
        for context, target in pbar:
            context = context.to(device)  # (batch, seq_len, engineered_dim)
            target = target.to(device)    # (batch, base_dim)

            optimizer.zero_grad()

            # Forward pass
            pred = model(context)  # (batch, base_dim)

            # Loss
            loss = criterion(pred, target)

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * context.size(0)
            train_samples += context.size(0)

            pbar.set_postfix({'loss': train_loss / train_samples})

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for context, target in tqdm(val_loader, desc=f"Эпоха {epoch + 1}/{epochs} [Val]"):
                context = context.to(device)
                target = target.to(device)

                pred = model(context)
                loss = criterion(pred, target)

                val_loss += loss.item() * context.size(0)
                val_samples += context.size(0)

        train_loss /= train_samples
        val_loss /= val_samples

        print(f"Эпоха {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Сохраняем лучшую модель
            if save_path:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'base_dim': model.output_dim,
                    'engineered_dim': model.input_dim,
                    'd_model': model.d_model,
                    'n_layers': len(model.layers),
                    'd_state': 16
                }, save_path)
                print(f"  Модель сохранена: {save_path}")
        else:
            patience_counter += 1
            print(f"  Early stopping patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

        # Очистка памяти
        if epoch % 5 == 0:
            gc.collect()

    print(f"Лучший val loss: {best_val_loss:.6f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Обучение Mamba-2 моделей')
    parser.add_argument('--data', type=str, default='../datasets/train.parquet',
                        help='Путь к файлу с данными')
    parser.add_argument('--num-models', type=int, default=3,
                        help='Количество моделей в ансамбле')
    parser.add_argument('--d-model', type=int, default=128,
                        help='Размерность модели')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Количество Mamba слоев')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Размер батча')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Количество эпох')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--context-length', type=int, default=40,
                        help='Длина контекста (уменьшено для экономии памяти)')
    parser.add_argument('--stride', type=int, default=2,
                        help='Шаг сэмплирования (1=все шаги, 2=каждый 2-й) - для экономии памяти')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Максимальное количество последовательностей для обучения (опционально)')

    args = parser.parse_args()

    print("=" * 60)
    print("ОБУЧЕНИЕ MAMBA-2 МОДЕЛЕЙ")
    print("=" * 60)
    print(f"Параметры:")
    print(f"  Данные: {args.data}")
    print(f"  Количество моделей: {args.num_models}")
    print(f"  d_model: {args.d_model}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  lr: {args.lr}")
    print(f"  context_length: {args.context_length}")
    print(f"  stride: {args.stride} (каждый {args.stride}-й шаг)")
    if args.max_sequences:
        print(f"  max_sequences: {args.max_sequences} (ограничение)")
    print("=" * 60)

    # Device
    device = torch.device('cpu')
    print(f"Используем устройство: {device}")

    # Загружаем и обрабатываем данные
    train_sequences, val_sequences, base_dim = prepare_sequences(args.data, max_sequences=args.max_sequences)

    # Вычисляем engineered_dim (оптимизированный набор)
    # 1+1+2+1+1+1+1+2+1+1 = 12x base_dim (СОКРАЩЕНО для оптимизации!)
    engineered_dim = 12 * base_dim
    print(f"\nРазмерность после feature engineering: {engineered_dim}")
    print(f"Оптимизированный набор (12x): raw, delta, SMA(2), STD, EMA, RSI, MACD, Bollinger(2), Momentum, ROC")
    print(f"Сокращено с 23x для баланса качества и скорости!")

    # Создаем datasets
    print("\nСоздаем datasets...")
    print(f"Используем stride={args.stride} для экономии памяти")
    train_dataset = TimeSeriesDataset(train_sequences, context_length=args.context_length, stride=args.stride)
    val_dataset = TimeSeriesDataset(val_sequences, context_length=args.context_length, stride=args.stride)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Оценка использования памяти
    memory_per_sample = args.context_length * engineered_dim * 4 / (1024**2)  # MB
    total_memory = memory_per_sample * len(train_dataset) / 1024  # GB
    print(f"Примерное использование памяти: {total_memory:.1f} GB")

    # Создаем dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # CPU-only
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Обучаем несколько моделей с разными random seeds
    for model_idx in range(args.num_models):
        print("\n" + "=" * 60)
        print(f"ОБУЧЕНИЕ МОДЕЛИ {model_idx + 1}/{args.num_models}")
        print("=" * 60)

        # Устанавливаем random seed
        seed = 42 + model_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Создаем модель
        model = Mamba2Model(
            input_dim=engineered_dim,
            output_dim=base_dim,
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_state=16,
            dropout=0.1
        )
        model.to(device)

        # Подсчет параметров
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Параметров в модели: {total_params:,}")

        # Путь для сохранения
        save_path = os.path.join(CURRENT_DIR, f"mamba_model_{model_idx}.pt")

        # Обучение
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_path=save_path
        )

        # Очистка памяти
        del model
        gc.collect()

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"Обученные модели сохранены в {CURRENT_DIR}")
    print(f"Файлы: mamba_model_0.pt, mamba_model_1.pt, ...")


if __name__ == "__main__":
    main()
