# Архитектура Wildfire Nexus Core

## Диаграмма потоков данных

```
[MODIS/VIIRS] → [FireDetectionService] → [FalsePositiveFilter] → [FireClusteringService]
                                                                         ↓
[Sentinel-2] → [BurnedAreaMapper] → [EventReport] ← [FireEvent]
                                                                         ↓
                                                                [TelegramAlert]
                                                                [LeafletMap]
```

## Слои

- **adapters/** — загрузка данных из внешних API
- **services/** — бизнес-логика (детекция, фильтрация, кластеризация)
- **api/** — REST endpoints
- **core/** — схемы данных и конфигурация

## Ключевые алгоритмы

- **Кластеризация:** DBSCAN-подобный с пространственно-временным окном (3 км, 24 ч)
- **NBR/dNBR:** (B08-B12)/(B08+B12) для оценки степени поражения
- **Фильтрация:** whitelist промзон + проверка confidence + термические пороги
