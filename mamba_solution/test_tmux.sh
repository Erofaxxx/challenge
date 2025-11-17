#!/bin/bash

# Скрипт для запуска тестирования solution.py в tmux

SESSION_NAME="mamba_testing"

# Проверяем установлен ли tmux
if ! command -v tmux &> /dev/null; then
    echo "❌ ОШИБКА: tmux не установлен!"
    echo "Установите: sudo apt-get install tmux"
    exit 1
fi

echo "======================================"
echo "Запуск тестирования в tmux"
echo "======================================"
echo ""

# Проверяем существует ли сессия
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Сессия '$SESSION_NAME' уже существует!"
    read -p "Подключиться к ней? (y/n): " choice
    if [[ $choice == "y" || $choice == "Y" ]]; then
        tmux attach-session -t $SESSION_NAME
        exit 0
    else
        echo "Удаляем старую сессию..."
        tmux kill-session -t $SESSION_NAME
    fi
fi

echo "Запускаем тестирование..."
echo ""

# Создаем tmux сессию
tmux new-session -d -s $SESSION_NAME -n "testing"

# Переходим в директорию и запускаем
tmux send-keys -t $SESSION_NAME "cd $(pwd)" C-m
tmux send-keys -t $SESSION_NAME "clear" C-m
tmux send-keys -t $SESSION_NAME "echo '===================================='" C-m
tmux send-keys -t $SESSION_NAME "echo 'ТЕСТИРОВАНИЕ РЕШЕНИЯ'" C-m
tmux send-keys -t $SESSION_NAME "echo '===================================='" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Время старта: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "python3 solution.py 2>&1 | tee testing.log" C-m

echo "✅ Тестирование запущено в tmux сессии '$SESSION_NAME'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Как работать с tmux:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. 👀 Подключиться к сессии:"
echo "   tmux attach-session -t $SESSION_NAME"
echo ""
echo "2. 🚪 Выйти из сессии (тест продолжится):"
echo "   Нажмите: Ctrl+B, затем D"
echo ""
echo "3. 📊 Посмотреть логи:"
echo "   tail -f testing.log"
echo ""
echo "4. ⏹️  Остановить:"
echo "   tmux kill-session -t $SESSION_NAME"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

sleep 2
tmux attach-session -t $SESSION_NAME
