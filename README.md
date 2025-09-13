# Chord Classifier

Обучение и оценка классификатора аккордов. Репозиторий содержит исходный `ipynb`, экспортированный `.py`, инструкции по установке и запуску.

## Возможности
- Загрузка аудио и извлечение признаков (например, chroma/MFCC).
- Обучение модели классификации аккордов.
- Оценка качества и инференс на пользовательских аудиофайлах.

## Установка
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск notebook
```bash
jupyter notebook
# Откройте notebooks/Chord_classifier.ipynb
```

## Экспортированный код
Сгенерирован файл `src/notebook_export.py` с объединённым кодом из ячеек.  
Он не заменяет notebook, но удобен для быстрого просмотра и статического анализа.

## Структура
```
.
├─ notebooks/
│  └─ Chord_classifier.ipynb    # оригинальный notebook
├─ src/
│  └─ notebook_export.py        # экспорт из .ipynb
├─ requirements.txt
└─ README.md
```

## Данные
Датасетом может послужить набор аудифайлов, содержанием которых является явный проигрыш какого-либо аккорда.
Прикладываю такой набор: Audio_files

