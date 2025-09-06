## Первый запуск

1) Установка зависимостей:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

2) Генерация датасета:

```bash
python generate_dataset.py
```

3) Обучение:

```bash
python train_model.py
```

4) Пример инференса:

```bash
python predict.py
```

5) Оценка качества (BLEU + валидность JSON):

```bash
python evaluate.py
```


