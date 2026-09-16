# Золотой набор данных для оценки качества Wildfire Nexus Core

## Статус данных

**Все данные в этой папке являются СИНТЕТИЧЕСКИМИ (demo_fixture).**

Они предназначены исключительно для:
- Тестирования модулей оценки качества
- Демонстрации работы метрик
- Отладки пайплайна оценки
- Хакатонной демонстрации

**Не используйте эти данные для научных выводов или реального мониторинга.**

---

## Файлы

### `golden_set.json`

Золотой набор для оценки детекции очагов горения.

**Структура:**
```json
{
  "metadata": {...},
  "detection_samples": [
    {
      "id": "sample_001",
      "sensor": "VIIRS",
      "latitude": 56.0184,
      "longitude": 92.8672,
      "brightness_temp_k": 345.2,
      "frp_mw": 45.3,
      "confidence_score": 0.92,
      "is_fire": 1,
      "reason": "..."
    },
    ...
  ]
}
```

**Характеристики:**
- 50 образцов total
- 30 реальных пожаров (is_fire = 1)
- 20 ложных срабатываний (is_fire = 0)

**Типы ложных срабатываний:**
1. Вода/озёра
2. Облачные края
3. Блики солнца на воде
4. Пустыня/открытый грунт
5. Индустриальные зоны (металлургия)
6. Одиночные шумовые пиксели
7. Снежные поля
8. Газовые факелы
9. Артефакты геолокации
10. Низкое качество снимка

---

### `burned_labels.geojson`

Эталонные полигоны гари для оценки картирования.

**Структура:**
```geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "event_id": "evt_demo_001",
        "area_ha": 185.4,
        "severity_class": "moderate_severity",
        "confidence": 0.87,
        "pre_scene_date": "2026-08-15",
        "post_scene_date": "2026-09-10",
        "cloud_cover_percent": 12
      },
      "geometry": {...}
    }
  ]
}
```

**Характеристики:**
- 3 эталонных события
- Диапазон площадей: 98–313 га
- Различные степени поражения (low/moderate/high)

---

## Использование

### Через EvaluationService

```python
from pathlib import Path
from app.services.evaluation import EvaluationService

eval_service = EvaluationService(Path("data/eval"))

# Проверка готовности
status = eval_service.get_evaluation_status()
print(status)

# Оценка детекции
metrics = eval_service.evaluate_detection_from_predictions(predictions)
print(f"F1: {metrics['f1']:.3f}")

# Полная оценка
report = eval_service.run_full_evaluation(
    fire_predictions=predictions,
    burned_predictions=burned_predictions
)
print(f"Overall score: {report['overall_score']:.3f}")
```

### Через API

```bash
curl http://localhost:8000/evaluate
```

Ответ:
```json
{
  "detection": {
    "precision": 0.87,
    "recall": 0.83,
    "f1": 0.85,
    "false_positive_rate": 0.12
  },
  "burned_area": {
    "area_mae_ha": 23.4,
    "area_mape_percent": 14.2
  },
  "overall_score": 0.78
}
```

---

## Метрики

### Детекция очагов
- **Precision**: Доля правильных положительных предсказаний
- **Recall**: Доля найденных реальных пожаров
- **F1-score**: Гармоническое среднее precision и recall
- **False Positive Rate**: Доля ложных срабатываний среди всех негативных случаев
- **Brier Score**: Качество вероятностных предсказаний (меньше = лучше)
- **ECE (Expected Calibration Error)**: Калибровка вероятностей

### Картирование гарей
- **MAE (га)**: Средняя абсолютная ошибка площади
- **MAPE (%)**: Средняя относительная ошибка площади
- **IoU**: Intersection over Union для масок
- **Dice coefficient**: Мера сходства масок

### Общая оценка
**Overall Score** рассчитывается как взвешенное среднее:
- Detection F1: 35%
- Burned Area IoU: 30%
- Calibration ECE: 20%
- Area MAPE (обратный): 15%

---

## Расширение набора

Для добавления новых образцов:

1. Добавьте запись в `detection_samples` с уникальным `id`
2. Укажите все обязательные поля:
   - sensor (MODIS | VIIRS | LANDSAT)
   - latitude, longitude
   - brightness_temp_k
   - frp_mw
   - confidence_score
   - is_fire (0 или 1)
   - reason (человекочитаемое объяснение)

3. Для полигонов добавьте Feature в `burned_labels.geojson`

---

## Ограничения

- Данные синтетические, не используйте для калибровки реальных моделей
- Пространственное покрытие ограничено (~Сибирь)
- Не представляет все типы ложных срабатываний
- Требуется расширение для production-использования

---

## Планы расширения

В будущих версиях планируется:
- Реальные размеченные данные FIRMS
- Больше типов ложных срабатываний
- Глобальное покрытие
- Временные серии для проверки persistence
- Мультиспектральные данные для контекста
