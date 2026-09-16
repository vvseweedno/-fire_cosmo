"""
Модуль оценки качества сервиса пожаров.

Загружает золотой набор данных, запускает оценку через метрики,
сохраняет результаты и предоставляет API для /evaluate.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from app.services.metrics import (
    evaluate_detection,
    evaluate_burned_area,
    create_evaluation_report,
    calculate_brier_score,
    calculate_expected_calibration_error
)

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Сервис для оценки качества детекции пожаров и картирования гарей.
    
    Работает с офлайн-фикстурами из data/eval/.
    Не требует доступа к интернету.
    """
    
    def __init__(self, eval_data_dir: Path):
        self.eval_data_dir = eval_data_dir
        self.golden_set_path = eval_data_dir / "golden_set.json"
        self.burned_labels_path = eval_data_dir / "burned_labels.geojson"
        
        self._golden_set: Optional[Dict] = None
        self._burned_labels: Optional[Dict] = None
    
    def load_golden_set(self) -> Dict[str, Any]:
        """
        Загрузить золотой набор данных для оценки детекции.
        
        Returns:
            Словарь с тестовыми данными
        
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если данные некорректны
        """
        if self._golden_set is not None:
            return self._golden_set
        
        if not self.golden_set_path.exists():
            raise FileNotFoundError(
                f"Золотой набор данных не найден: {self.golden_set_path}\n"
                f"Создайте файл с тестовыми данными для оценки."
            )
        
        with open(self.golden_set_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Валидация структуры
        required_keys = ["detection_samples", "metadata"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Отсутствует обязательное поле в golden_set: {key}")
        
        self._golden_set = data
        logger.info(f"Загружен золотой набор: {len(data['detection_samples'])} образцов")
        
        return data
    
    def load_burned_labels(self) -> Optional[Dict[str, Any]]:
        """
        Загрузить эталонные полигоны гари.
        
        Returns:
            GeoJSON с эталонными полигонами или None
        """
        if self._burned_labels is not None:
            return self._burned_labels
        
        if not self.burned_labels_path.exists():
            logger.warning(f"Эталонные полигоны гари не найдены: {self.burned_labels_path}")
            return None
        
        with open(self.burned_labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._burned_labels = data
        logger.info(f"Загружено эталонных полигонов гари: {len(data.get('features', []))}")
        
        return data
    
    def evaluate_detection_from_predictions(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Оценить качество детекции по предсказаниям модели.
        
        Args:
            predictions: Список предсказаний модели
        
        Returns:
            Метрики детекции
        """
        try:
            golden_set = self.load_golden_set()
        except FileNotFoundError as e:
            logger.warning(str(e))
            return {
                "status": "insufficient_eval_data",
                "missing": ["golden_set"],
                "recommendation": "Создайте файл data/eval/golden_set.json с размеченными данными"
            }
        
        # Извлечение истинных меток и предсказаний
        y_true = []
        y_pred = []
        y_prob = []
        
        samples = golden_set["detection_samples"]
        
        # Сопоставление предсказаний с эталоном
        pred_by_id = {p.get("id"): p for p in predictions}
        
        for sample in samples:
            sample_id = sample["id"]
            true_label = sample["is_fire"]  # 0 или 1
            
            if sample_id in pred_by_id:
                pred = pred_by_id[sample_id]
                pred_label = 1 if pred.get("is_valid", False) else 0
                prob = pred.get("fire_probability", 0.5)
            else:
                # Если предсказания нет, считаем как отрицательный ответ
                pred_label = 0
                prob = 0.0
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            y_prob.append(prob)
        
        # Расчёт метрик
        metrics = evaluate_detection(y_true, y_pred, y_prob)
        metrics["n_samples"] = len(samples)
        metrics["status"] = "success"
        
        logger.info(f"Оценка детекции: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}")
        
        return metrics
    
    def evaluate_burned_area_from_predictions(
        self,
        burned_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Оценить качество картирования гарей.
        
        Args:
            burned_predictions: Предсказания площадей и масок гари
        
        Returns:
            Метрики картирования
        """
        burned_labels = self.load_burned_labels()
        
        if burned_labels is None or len(burned_labels.get("features", [])) == 0:
            return {
                "status": "insufficient_eval_data",
                "missing": ["burned_labels"],
                "recommendation": "Создайте файл data/eval/burned_labels.geojson с эталонными полигонами"
            }
        
        # Для хакатона упростим: оценим только площади
        areas_true = []
        areas_pred = []
        
        label_features = burned_labels["features"]
        
        for feature in label_features:
            event_id = feature["properties"].get("event_id")
            true_area = feature["properties"].get("area_ha")
            
            if true_area is None:
                continue
            
            # Найти предсказание для этого события
            pred_area = None
            for pred in burned_predictions:
                if pred.get("event_id") == event_id:
                    pred_area = pred.get("area_ha")
                    break
            
            if pred_area is not None:
                areas_true.append(true_area)
                areas_pred.append(pred_area)
        
        if len(areas_true) == 0:
            return {
                "status": "no_matching_predictions",
                "message": "Не найдено совпадений между предсказаниями и эталоном"
            }
        
        # Упрощённая оценка по площадям
        from app.services.metrics import calculate_area_metrics
        mae, mape = calculate_area_metrics(areas_true, areas_pred)
        
        metrics = {
            "area_mae_ha": round(mae, 2),
            "area_mape_percent": round(mape, 2),
            "n_samples": len(areas_true),
            "status": "success"
        }
        
        logger.info(f"Оценка гарей: MAE={mae:.2f} га, MAPE={mape:.1f}%")
        
        return metrics
    
    def run_full_evaluation(
        self,
        fire_predictions: List[Dict[str, Any]],
        burned_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Запустить полную оценку сервиса.
        
        Args:
            fire_predictions: Предсказания детекции очагов
            burned_predictions: Предсказания картирования гарей
        
        Returns:
            Полный отчёт с overall_score
        """
        logger.info("Запуск полной оценки сервиса...")
        
        # Оценка детекции
        detection_metrics = self.evaluate_detection_from_predictions(fire_predictions)
        
        # Оценка гарей
        burned_area_metrics = self.evaluate_burned_area_from_predictions(burned_predictions)
        
        # Калибровка (из метрик детекции)
        calibration_metrics = {}
        if "brier_score" in detection_metrics:
            calibration_metrics["brier_score"] = detection_metrics["brier_score"]
            calibration_metrics["expected_calibration_error"] = detection_metrics.get(
                "expected_calibration_error", 0.0
            )
        
        # Сводный отчёт
        report = create_evaluation_report(
            detection_metrics=detection_metrics,
            burned_area_metrics=burned_area_metrics,
            calibration_metrics=calibration_metrics if calibration_metrics else None
        )
        
        logger.info(f"Полная оценка завершена. Overall score: {report['overall_score']:.3f}")
        
        return report
    
    def get_evaluation_status(self) -> Dict[str, Any]:
        """
        Проверить готовность системы оценки.
        
        Returns:
            Статус с перечнем доступных/отсутствующих данных
        """
        status = {
            "ready": True,
            "golden_set_available": False,
            "burned_labels_available": False,
            "missing_files": []
        }
        
        if self.golden_set_path.exists():
            status["golden_set_available"] = True
        else:
            status["ready"] = False
            status["missing_files"].append(str(self.golden_set_path))
        
        if self.burned_labels_path.exists():
            status["burned_labels_available"] = True
        else:
            status["missing_files"].append(str(self.burned_labels_path))
        
        return status
