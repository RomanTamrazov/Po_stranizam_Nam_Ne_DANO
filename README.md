![Project Logo](Screenshot 2026-03-17 at 21.34.41.png)
# Решение задачи хакатона академии ии "По страницам"
Private lb score: 0.834 (топ 3 среди не участников НТО ИИ)

## Что делает проект

Проект формирует файл `submission_Werserk_I_Love_U.csv` для задачи рекомендательной витрины.

По условию (см. `task.md`): для каждого `user_id` из `submit/targets.csv` нужно выбрать и ранжировать **20 изданий** (`edition_id`) из **200 кандидатов** из `submit/candidates.csv` (на каждого пользователя), где `rank=1` — верх витрины.

## Что делает код

Единая точка входа — скрипт `po_stranizam_Nam_Ne_DANO.py`.

На высоком уровне он:

- Загружает данные из `data/` и `submit/`.
- Строит признаки для кандидатов пользователя, совмещая коллаборативные и контентные сигналы:
  - коллаборативные эмбеддинги на основе матрицы взаимодействий (SVD по user×item),
  - текстовые признаки из `title`/`description` (TF‑IDF + TruncatedSVD),
  - статистики по пользователям/изданиям и дополнительные агрегаты.
- Обучает ранжирующую модель `CatBoostRanker` (loss `YetiRank`) для сортировки 200 кандидатов на пользователя.
- Считает предсказанный скор для каждой пары `(user_id, edition_id)`, выбирает top‑20 и сохраняет результат в `submission_Werserk_I_Love_U.csv` формата:

```
user_id,edition_id,rank
```

## Входные данные

- `data/`:
  - `interactions.csv`, `editions.csv`, `book_genres.csv`, `users.csv` (и др. справочники)
- `submit/`:
  - `targets.csv` — список пользователей
  - `candidates.csv` — 200 кандидатов на каждого пользователя

## Как запустить (получить submission_Werserk_I_Love_U.csv)


```bash

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# (опционально) код был запущен и работает с помощью GPU ( При смене на CPU может повлиять на метрику)


python po_stranizam_Nam_Ne_DANO.py
```

После выполнения в текущей директории появится `submission_Werserk_I_Love_U.csv`.
