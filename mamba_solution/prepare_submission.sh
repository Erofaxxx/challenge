#!/bin/bash

echo "======================================"
echo "Подготовка submission файла"
echo "======================================"

# Создаем временную директорию
mkdir -p submission_temp

# Копируем необходимые файлы
echo "Копируем файлы..."
cp solution.py submission_temp/

# Копируем обученные модели (если есть)
if ls mamba_model_*.pt 1> /dev/null 2>&1; then
    cp mamba_model_*.pt submission_temp/
    echo "✓ Скопированы обученные модели"
else
    echo "⚠ ВНИМАНИЕ: Обученные модели не найдены!"
    echo "  Запустите: python train.py --data ../datasets/train.parquet"
fi

# Создаем архив
echo "Создаем архив..."
cd submission_temp
zip -r ../submission.zip . > /dev/null
cd ..

# Удаляем временную директорию
rm -rf submission_temp

echo ""
echo "======================================"
echo "✓ Создан файл submission.zip"
echo "======================================"
echo "Размер:"
ls -lh submission.zip

echo ""
echo "Содержимое архива:"
unzip -l submission.zip

echo ""
echo "======================================"
echo "Готово к отправке!"
echo "======================================"
