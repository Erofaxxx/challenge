#!/bin/bash

# Скрипт для запуска обучения в tmux с кастомными параметрами

SESSION_NAME="mamba_training"

# Проверяем установлен ли tmux
if ! command -v tmux &> /dev/null; then
    echo "❌ ОШИБКА: tmux не установлен!"
    echo "Установите: sudo apt-get install tmux"
    exit 1
fi

echo "======================================"
echo "Настройка обучения в tmux"
echo "======================================"
echo ""

# Интерактивный выбор параметров
echo "⚙️  Выберите конфигурацию:"
echo ""
echo "1) Быстрое тестирование (1 модель, 5 эпох)"
echo "2) Базовое обучение (3 модели, 15 эпох) - рекомендуется"
echo "3) Улучшенное обучение (5 моделей, 20 эпох)"
echo "4) Кастомные параметры"
echo ""
read -p "Ваш выбор (1-4): " config_choice

case $config_choice in
    1)
        NUM_MODELS=1
        EPOCHS=5
        BATCH_SIZE=32
        D_MODEL=128
        N_LAYERS=4
        STRIDE=2
        CONTEXT=40
        echo "✅ Выбрано: Быстрое тестирование"
        ;;
    2)
        NUM_MODELS=3
        EPOCHS=15
        BATCH_SIZE=64
        D_MODEL=128
        N_LAYERS=4
        STRIDE=2
        CONTEXT=40
        echo "✅ Выбрано: Базовое обучение"
        ;;
    3)
        NUM_MODELS=5
        EPOCHS=20
        BATCH_SIZE=64
        D_MODEL=192
        N_LAYERS=6
        STRIDE=2
        CONTEXT=50
        echo "✅ Выбрано: Улучшенное обучение"
        ;;
    4)
        echo ""
        read -p "Количество моделей (1-7): " NUM_MODELS
        read -p "Количество эпох (5-30): " EPOCHS
        read -p "Batch size (16/32/64/128): " BATCH_SIZE
        read -p "d_model (64/96/128/192): " D_MODEL
        read -p "n_layers (3/4/5/6): " N_LAYERS
        read -p "Context length (30-80): " CONTEXT
        read -p "Stride (1-3, 2=рекомендуется): " STRIDE
        echo "✅ Выбрано: Кастомная конфигурация"
        ;;
    *)
        echo "❌ Неверный выбор"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Параметры обучения:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  • Количество моделей: $NUM_MODELS"
echo "  • Эпохи: $EPOCHS"
echo "  • Batch size: $BATCH_SIZE"
echo "  • d_model: $D_MODEL"
echo "  • n_layers: $N_LAYERS"
echo "  • context_length: $CONTEXT"
echo "  • stride: $STRIDE (каждый $STRIDE-й шаг)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Оценка времени
total_time=$((NUM_MODELS * EPOCHS * 8))  # Примерно 8 минут на эпоху
hours=$((total_time / 60))
minutes=$((total_time % 60))
echo "⏱️  Примерное время обучения: ${hours}ч ${minutes}мин"
echo ""

read -p "Продолжить? (y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Отмена."
    exit 0
fi

# Проверяем существует ли сессия
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo ""
    echo "⚠️  Сессия '$SESSION_NAME' уже существует!"
    read -p "Удалить и создать новую? (y/n): " kill_confirm
    if [[ $kill_confirm == "y" || $kill_confirm == "Y" ]]; then
        tmux kill-session -t $SESSION_NAME
    else
        echo "Отмена."
        exit 0
    fi
fi

echo ""
echo "Запускаем обучение..."

# Создаем команду запуска
TRAIN_CMD="python3 train.py --data ../datasets/train.parquet --num-models $NUM_MODELS --epochs $EPOCHS --batch-size $BATCH_SIZE --d-model $D_MODEL --n-layers $N_LAYERS --context-length $CONTEXT --stride $STRIDE 2>&1 | tee training_${NUM_MODELS}models_${EPOCHS}ep.log"

# Создаем tmux сессию
tmux new-session -d -s $SESSION_NAME -n "training"

# Настраиваем и запускаем
tmux send-keys -t $SESSION_NAME "cd $(pwd)" C-m
tmux send-keys -t $SESSION_NAME "clear" C-m
tmux send-keys -t $SESSION_NAME "echo '===================================='" C-m
tmux send-keys -t $SESSION_NAME "echo 'ОБУЧЕНИЕ MAMBA-2 МОДЕЛЕЙ'" C-m
tmux send-keys -t $SESSION_NAME "echo '===================================='" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Параметры:'" C-m
tmux send-keys -t $SESSION_NAME "echo '  Модели: $NUM_MODELS'" C-m
tmux send-keys -t $SESSION_NAME "echo '  Эпохи: $EPOCHS'" C-m
tmux send-keys -t $SESSION_NAME "echo '  Batch: $BATCH_SIZE'" C-m
tmux send-keys -t $SESSION_NAME "echo '  d_model: $D_MODEL'" C-m
tmux send-keys -t $SESSION_NAME "echo '  n_layers: $N_LAYERS'" C-m
tmux send-keys -t $SESSION_NAME "echo '  context: $CONTEXT, stride: $STRIDE'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Старт: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "$TRAIN_CMD" C-m

echo ""
echo "✅ Обучение запущено в tmux сессии '$SESSION_NAME'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Как работать с tmux:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. 👀 Подключиться к сессии:"
echo "   tmux attach-session -t $SESSION_NAME"
echo ""
echo "2. 🚪 Выйти из сессии (Ctrl+B, затем D)"
echo ""
echo "3. 📊 Посмотреть логи:"
echo "   tail -f training_${NUM_MODELS}models_${EPOCHS}ep.log"
echo ""
echo "4. ⏹️  Остановить:"
echo "   tmux kill-session -t $SESSION_NAME"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

sleep 2
tmux attach-session -t $SESSION_NAME
