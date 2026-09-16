#!/usr/bin/env python3
"""
Скрипт загрузки/генерации демо-данных для Wildfire Nexus Core.

Поддерживает два режима:
1. Онлайн (по умолчанию): попытка скачать реальные данные (требует API ключи)
2. Офлайн (--offline): генерация синтетических фикстур для демо

Все синтетические данные явно помечаются как demo_fixture.
"""

import argparse
import json
import csv
import os
from pathlib import Path
from datetime import datetime, timedelta
import random
import math

# Константы для синтетических данных
FIXTURE_REGION = {
    "name": "Siberia Demo Region",
    "center_lat": 56.0,
    "center_lon": 92.0,
    "description": "Синтетический регион для демонстрации работы сервиса"
}

def generate_modis_fixture(output_path: Path):
    """Генерация синтетических данных MODIS"""
    print(f"Генерация MODIS фикстур в {output_path}")
    
    # Создаем точки пожара (кластер из ~15 точек)
    fire_points = []
    base_time = datetime(2026, 7, 15, 10, 30)
    
    # Реальные точки пожара (высокий confidence, высокая температура)
    for i in range(15):
        lat = FIXTURE_REGION["center_lat"] + random.uniform(-0.05, 0.05)
        lon = FIXTURE_REGION["center_lon"] + random.uniform(-0.05, 0.05)
        brightness = random.uniform(340, 380)  # Высокая температура
        confidence = random.uniform(0.7, 1.0)  # Высокая уверенность
        frp = random.uniform(20, 80)  # Мощность пожара
        
        fire_points.append({
            "id": f"MODIS_{i:04d}",
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "acq_date": (base_time + timedelta(minutes=i*5)).strftime("%Y-%m-%d"),
            "acq_time": (base_time + timedelta(minutes=i*5)).strftime("%H%M"),
            "satellite": "Terra",
            "instrument": "MODIS",
            "version": "6.1",
            "bright": round(brightness, 2),
            "frp": round(frp, 2),
            "confidence": round(confidence, 2),
            "daynight": "D"
        })
    
    # Ложные срабатывания (низкий confidence или аномалии)
    false_points = [
        # Точка с низким confidence
        {
            "id": "MODIS_FP_001",
            "latitude": FIXTURE_REGION["center_lat"] + 0.2,
            "longitude": FIXTURE_REGION["center_lon"] - 0.2,
            "acq_date": base_time.strftime("%Y-%m-%d"),
            "acq_time": "1030",
            "satellite": "Aqua",
            "instrument": "MODIS",
            "version": "6.1",
            "bright": 315,
            "frp": 3.5,
            "confidence": 0.25,  # Низкая уверенность
            "daynight": "D"
        },
        # Промышленная точка (стабильная, вне леса)
        {
            "id": "MODIS_FP_002",
            "latitude": FIXTURE_REGION["center_lat"] - 0.15,
            "longitude": FIXTURE_REGION["center_lon"] + 0.1,
            "acq_date": base_time.strftime("%Y-%m-%d"),
            "acq_time": "1035",
            "satellite": "Terra",
            "instrument": "MODIS",
            "version": "6.1",
            "bright": 350,
            "frp": 45,
            "confidence": 0.85,
            "daynight": "D"
        }
    ]
    
    all_points = fire_points + false_points
    
    # Запись в CSV
    with open(output_path, 'w', newline='') as f:
        fieldnames = ["id", "latitude", "longitude", "acq_date", "acq_time", 
                     "satellite", "instrument", "version", "bright", "frp", 
                     "confidence", "daynight"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_points)
    
    print(f"  Создано {len(fire_points)} точек пожара + {len(false_points)} ложных")
    return len(fire_points), len(false_points)


def generate_viirs_fixture(output_path: Path):
    """Генерация синтетических данных VIIRS (375м разрешение)"""
    print(f"Генерация VIIRS фикстур в {output_path}")
    
    viirs_points = []
    base_time = datetime(2026, 7, 15, 11, 0)
    
    # VIIRS точки (более высокое разрешение, больше точек)
    for i in range(25):
        lat = FIXTURE_REGION["center_lat"] + random.uniform(-0.03, 0.03)
        lon = FIXTURE_REGION["center_lon"] + random.uniform(-0.03, 0.03)
        brightness = random.uniform(330, 370)
        confidence = random.uniform(0.6, 0.95)
        frp = random.uniform(15, 60)
        
        viirs_points.append({
            "id": f"VIIRS_{i:04d}",
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "acq_date": (base_time + timedelta(minutes=i*3)).strftime("%Y-%m-%d"),
            "acq_time": (base_time + timedelta(minutes=i*3)).strftime("%H%M"),
            "satellite": "Suomi-NPP",
            "instrument": "VIIRS",
            "version": "1.0",
            "bright": round(brightness, 2),
            "frp": round(frp, 2),
            "confidence": round(confidence, 2),
            "daynight": "D",
            "type": "V"
        })
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ["id", "latitude", "longitude", "acq_date", "acq_time",
                     "satellite", "instrument", "version", "bright", "frp",
                     "confidence", "daynight", "type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(viirs_points)
    
    print(f"  Создано {len(viirs_points)} точек VIIRS")
    return len(viirs_points)


def generate_sentinel2_fixture(output_dir: Path):
    """Генерация синтетических данных Sentinel-2 (упрощенные растровые фикстуры)"""
    print(f"Генерация Sentinel-2 фикстур в {output_dir}")
    
    # Метаданные события
    event_metadata = {
        "event_id": "demo_fire_001",
        "type": "Feature",
        "properties": {
            "source": "demo_fixture",
            "description": "Синтетическое пожарное событие для демонстрации",
            "region": FIXTURE_REGION["name"],
            "center": [FIXTURE_REGION["center_lon"], FIXTURE_REGION["center_lat"]],
            "pre_scene": {
                "scene_id": "S2A_MSIL2A_20260601_T000000_N0000_R000_T00XXX",
                "datetime": "2026-06-01T00:00:00Z",
                "cloud_cover": 5.2,
                "days_before_fire": 45
            },
            "post_scene": {
                "scene_id": "S2A_MSIL2A_20260720_T000000_N0000_R000_T00XXX",
                "datetime": "2026-07-20T00:00:00Z",
                "cloud_cover": 8.7,
                "days_after_fire": 5
            },
            "expected_burned_area_ha": {
                "min": 50,
                "max": 500,
                "estimate": 180
            },
            "severity_distribution": {
                "high": 0.25,
                "moderate": 0.45,
                "low": 0.20,
                "unburned": 0.10
            }
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [FIXTURE_REGION["center_lon"] - 0.1, FIXTURE_REGION["center_lat"] - 0.1],
                [FIXTURE_REGION["center_lon"] + 0.1, FIXTURE_REGION["center_lat"] - 0.1],
                [FIXTURE_REGION["center_lon"] + 0.1, FIXTURE_REGION["center_lat"] + 0.1],
                [FIXTURE_REGION["center_lon"] - 0.1, FIXTURE_REGION["center_lat"] + 0.1],
                [FIXTURE_REGION["center_lon"] - 0.1, FIXTURE_REGION["center_lat"] - 0.1]
            ]]
        }
    }
    
    # Сохраняем метаданные
    metadata_path = output_dir / "demo_event.json"
    with open(metadata_path, 'w') as f:
        json.dump(event_metadata, f, indent=2)
    
    # Генерируем упрощенные растровые данные (текстовое представление для демо)
    # В реальном сценарии здесь были бы GeoTIFF файлы
    # Для демо создаем JSON с матрицами значений NBR
    
    def generate_nbr_matrix(size: int, has_fire: bool) -> list:
        """Генерация матрицы NBR значений"""
        matrix = []
        center = size // 2
        
        for i in range(size):
            row = []
            for j in range(size):
                dist_from_center = math.sqrt((i - center)**2 + (j - center)**2)
                
                if has_fire and dist_from_center < size/4:
                    # Зона гари: низкий NBR (после пожара)
                    base_val = random.uniform(-0.2, 0.1)
                    # Добавляем вариацию по степени поражения
                    if dist_from_center < size/8:
                        base_val = random.uniform(-0.3, -0.1)  # Высокая степень
                else:
                    # Здоровая растительность: высокий NBR
                    base_val = random.uniform(0.4, 0.8)
                
                row.append(round(base_val, 3))
            matrix.append(row)
        
        return matrix
    
    size = 50  # 50x50 пикселей для демо
    
    # Pre-fire NBR (здоровая растительность)
    pre_nbr = generate_nbr_matrix(size, has_fire=False)
    with open(output_dir / "pre_NBR.json", 'w') as f:
        json.dump({"matrix": pre_nbr, "size": size, "pixel_size_m": 10}, f)
    
    # Post-fire NBR (с гарью)
    post_nbr = generate_nbr_matrix(size, has_fire=True)
    with open(output_dir / "post_NBR.json", 'w') as f:
        json.dump({"matrix": post_nbr, "size": size, "pixel_size_m": 10}, f)
    
    # dNBR (разница)
    dnbr_matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            dnbr = pre_nbr[i][j] - post_nbr[i][j]
            row.append(round(dnbr, 3))
        dnbr_matrix.append(row)
    
    with open(output_dir / "dNBR.json", 'w') as f:
        json.dump({
            "matrix": dnbr_matrix,
            "size": size,
            "pixel_size_m": 10,
            "projection": "EPSG:6933",
            "ha_per_pixel": 0.01  # 10м x 10м = 100 м² = 0.01 га
        }, f)
    
    print(f"  Созданы растровые фикстуры {size}x{size} пикселей")
    print(f"  Ожидаемая площадь гари: ~{event_metadata['properties']['expected_burned_area_ha']['estimate']} га")
    
    return event_metadata


def generate_demo_payload(output_path: Path):
    """Генерация примера запроса для API"""
    payload = {
        "bbox": [
            FIXTURE_REGION["center_lon"] - 0.2,
            FIXTURE_REGION["center_lat"] - 0.2,
            FIXTURE_REGION["center_lon"] + 0.2,
            FIXTURE_REGION["center_lat"] + 0.2
        ],
        "start_date": "2026-07-01",
        "end_date": "2026-07-31",
        "sensors": ["MODIS", "VIIRS"],
        "min_confidence": "nominal",
        "mode": "demo_fixture"
    }
    
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    
    print(f"Создан демо-запрос в {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка/генерация демо-данных для Wildfire Nexus Core"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Генерировать синтетические фикстуры вместо загрузки реальных данных"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/fixtures",
        help="Директория для сохранения данных"
    )
    
    args = parser.parse_args()
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Wildfire Nexus Core - Демо данные")
    print("=" * 60)
    
    if args.offline:
        print("\nРЕЖИМ: Офлайн генерация синтетических фикстур")
        print("ВНИМАНИЕ: Все сгенерированные данные являются DEMO_FIXTURE\n")
        
        fires_dir = output_base / "fires"
        sentinel2_dir = output_base / "sentinel2"
        
        fires_dir.mkdir(parents=True, exist_ok=True)
        sentinel2_dir.mkdir(parents=True, exist_ok=True)
        
        # Генерация фикстур
        modis_real, modis_fp = generate_modis_fixture(fires_dir / "modis_demo.csv")
        viirs_count = generate_viirs_fixture(fires_dir / "viirs_demo.csv")
        event_meta = generate_sentinel2_fixture(sentinel2_dir)
        generate_demo_payload(output_base / "demo_payload.json")
        
        # README с описанием
        readme_content = f"""# Демо фикстуры Wildfire Nexus Core

## Статус: DEMO_FIXTURE

Эти данные являются **синтетическими** и предназначены только для демонстрации работы сервиса.
Не используйте для реального мониторинга пожаров.

## Содержимое

### fires/
- `modis_demo.csv` - {modis_real} точек пожара + {modis_fp} ложных срабатываний
- `viirs_demo.csv` - {viirs_count} точек VIIRS

### sentinel2/
- `demo_event.json` - метаданные синтетического события
- `pre_NBR.json` - матрица NBR до пожара
- `post_NBR.json` - матрица NBR после пожара  
- `dNBR.json` - матрица разницы dNBR

## Ожидаемые результаты

При обработке этих фикстур сервис должен:
1. Обнаружить {modis_real + viirs_count} тепловых точек
2. Отфильтровать {modis_fp} ложных срабатываний
3. Скластеризовать точки в 1 пожарное событие
4. Рассчитать площадь гари: {event_meta['properties']['expected_burned_area_ha']['min']}-{event_meta['properties']['expected_burned_area_ha']['max']} га
5. Определить степень поражения согласно порогам dNBR

## Регион

- Название: {FIXTURE_REGION['name']}
- Центр: {FIXTURE_REGION['center_lat']}, {FIXTURE_REGION['center_lon']}
- Описание: {FIXTURE_REGION['description']}

## Генерация

Дата генерации: {datetime.now().isoformat()}
Команда: python scripts/download_demo_data.py --offline
"""
        
        with open(output_base / "README.md", 'w') as f:
            f.write(readme_content)
        
        print("\n" + "=" * 60)
        print("✅ Генерация завершена успешно!")
        print("=" * 60)
        print(f"\nФайлы созданы в: {output_base.absolute()}")
        print("\nДля запуска сервиса с фикстурами:")
        print("  export OFFLINE_MODE=true")
        print("  uvicorn app.main:app --reload")
        print("\nИли через docker-compose:")
        print("  docker compose up")
        
    else:
        print("\nРЕЖИМ: Онлайн загрузка (требуется реализация)")
        print("В текущей версии поддерживается только --offline режим.")
        print("\nИспользуйте:")
        print("  python scripts/download_demo_data.py --offline")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
