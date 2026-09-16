"""
Модуль метрик для оценки качества сервиса пожаров.

Поддерживает офлайн-фикстуры, не использует интернет.
Считает метрики детекции, картирования гарей и калибровки.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionMetrics:
    """Метрики детекции очагов"""
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class BurnedAreaMetrics:
    """Метрики картирования гарей"""
    iou: float
    dice: float
    area_mae_ha: float
    area_mape_percent: float
    pixel_accuracy: float


@dataclass
class CalibrationMetrics:
    """Метрики калибровки вероятностей"""
    brier_score: float
    expected_calibration_error: float
    reliability_diagram_data: List[Dict[str, float]]


def calculate_precision_recall_f1(
    y_true: List[int], 
    y_pred: List[int]
) -> Tuple[float, float, float]:
    """
    Рассчитать precision, recall, F1-score.
    
    Args:
        y_true: Истинные метки (0/1)
        y_pred: Предсказанные метки (0/1)
    
    Returns:
        (precision, recall, f1)
    """
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def calculate_false_positive_rate(
    y_true: List[int], 
    y_pred: List[int]
) -> float:
    """
    Рассчитать долю ложных срабатываний.
    
    FPR = FP / (FP + TN)
    """
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def calculate_iou_dice(
    mask_true: np.ndarray, 
    mask_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Рассчитать IoU и Dice coefficient для бинарных масок.
    
    Args:
        mask_true: Истинная маска гари
        mask_pred: Предсказанная маска гари
    
    Returns:
        (iou, dice)
    """
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()
    total_true = mask_true.sum()
    total_pred = mask_pred.sum()
    
    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / (total_true + total_pred) if (total_true + total_pred) > 0 else 0.0
    
    return iou, dice


def calculate_area_metrics(
    areas_true: List[float], 
    areas_pred: List[float]
) -> Tuple[float, float]:
    """
    Рассчитать MAE и MAPE для площадей в гектарах.
    
    Args:
        areas_true: Истинные площади (га)
        areas_pred: Предсказанные площади (га)
    
    Returns:
        (mae_ha, mape_percent)
    """
    if len(areas_true) != len(areas_pred) or len(areas_true) == 0:
        return 0.0, 0.0
    
    errors = [abs(at - ap) for at, ap in zip(areas_true, areas_pred)]
    mae = np.mean(errors)
    
    # MAPE с защитой от деления на ноль
    mape_values = []
    for at, ap in zip(areas_true, areas_pred):
        if at > 0:
            mape_values.append(abs(at - ap) / at * 100)
    
    mape = np.mean(mape_values) if mape_values else 0.0
    
    return mae, mape


def calculate_brier_score(
    y_true: List[int], 
    y_prob: List[float]
) -> float:
    """
    Рассчитать Brier score для вероятностных предсказаний.
    
    Brier = mean((y_true - y_prob)^2)
    
    Args:
        y_true: Истинные метки (0/1)
        y_prob: Предсказанные вероятности [0, 1]
    
    Returns:
        brier_score ∈ [0, 1] (меньше = лучше)
    """
    if len(y_true) != len(y_prob):
        return 0.0
    
    squared_errors = [(yt - yp) ** 2 for yt, yp in zip(y_true, y_prob)]
    return np.mean(squared_errors)


def calculate_expected_calibration_error(
    y_true: List[int], 
    y_prob: List[float], 
    n_bins: int = 10
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Рассчитать Expected Calibration Error (ECE) и данные для reliability diagram.
    
    ECE = Σ (|Bm|/n) * |acc(Bm) - conf(Bm)|
    
    Args:
        y_true: Истинные метки (0/1)
        y_prob: Предсказанные вероятности [0, 1]
        n_bins: Количество бинов для калибровки
    
    Returns:
        (ece, reliability_data)
    """
    if len(y_true) != len(y_prob) or len(y_true) == 0:
        return 0.0, []
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability_data = []
    
    n_total = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Найти предсказания в этом бине
        in_bin = [(yt, yp) for yt, yp in zip(y_true, y_prob) 
                  if bin_lower <= yp < bin_upper or (i == n_bins - 1 and yp == bin_upper)]
        
        if len(in_bin) > 0:
            bin_acc = np.mean([yt for yt, _ in in_bin])
            bin_conf = np.mean([yp for _, yp in in_bin])
            bin_size = len(in_bin)
            
            ece += (bin_size / n_total) * abs(bin_acc - bin_conf)
            
            reliability_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "count": bin_size
            })
    
    return ece, reliability_data


def evaluate_detection(
    y_true: List[int], 
    y_pred: List[int], 
    y_prob: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Полная оценка детекции очагов.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        y_prob: Предсказанные вероятности (опционально)
    
    Returns:
        Словарь с метриками
    """
    precision, recall, f1 = calculate_precision_recall_f1(y_true, y_pred)
    fpr = calculate_false_positive_rate(y_true, y_pred)
    
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    
    result = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "n_samples": len(y_true)
    }
    
    if y_prob is not None:
        brier = calculate_brier_score(y_true, y_prob)
        ece, reliability = calculate_expected_calibration_error(y_true, y_prob)
        result["brier_score"] = round(brier, 4)
        result["expected_calibration_error"] = round(ece, 4)
        result["reliability_diagram"] = reliability
    
    return result


def evaluate_burned_area(
    masks_true: List[np.ndarray], 
    masks_pred: List[np.ndarray],
    areas_true: List[float],
    areas_pred: List[float]
) -> Dict[str, Any]:
    """
    Полная оценка картирования гарей.
    
    Args:
        masks_true: Список истинных масок
        masks_pred: Список предсказанных масок
        areas_true: Истинные площади (га)
        areas_pred: Предсказанные площади (га)
    
    Returns:
        Словарь с метриками
    """
    if len(masks_true) != len(masks_pred) or len(masks_true) == 0:
        return {
            "status": "insufficient_data",
            "message": "Недостаточно данных для оценки"
        }
    
    ious = []
    dices = []
    
    for mt, mp in zip(masks_true, masks_pred):
        iou, dice = calculate_iou_dice(mt, mp)
        ious.append(iou)
        dices.append(dice)
    
    area_mae, area_mape = calculate_area_metrics(areas_true, areas_pred)
    
    # Pixel accuracy (усреднённый по всем маскам)
    pixel_accuracies = []
    for mt, mp in zip(masks_true, masks_pred):
        accuracy = np.mean(mt == mp)
        pixel_accuracies.append(accuracy)
    
    return {
        "iou": round(np.mean(ious), 4),
        "dice": round(np.mean(dices), 4),
        "pixel_accuracy": round(np.mean(pixel_accuracies), 4),
        "area_mae_ha": round(area_mae, 2),
        "area_mape_percent": round(area_mape, 2),
        "n_samples": len(masks_true)
    }


def create_evaluation_report(
    detection_metrics: Dict[str, Any],
    burned_area_metrics: Dict[str, Any],
    calibration_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Создать сводный отчёт об оценке.
    
    Returns:
        Полный отчёт с overall_score
    """
    # Расчёт общего scores (взвешенное среднее ключевых метрик)
    weights = {
        "detection_f1": 0.35,
        "burned_iou": 0.30,
        "calibration_ece": 0.20,
        "area_mape_inverse": 0.15
    }
    
    score_components = []
    
    # Detection F1
    if "f1" in detection_metrics:
        score_components.append(weights["detection_f1"] * detection_metrics["f1"])
    
    # Burned Area IoU
    if "iou" in burned_area_metrics:
        score_components.append(weights["burned_iou"] * burned_area_metrics["iou"])
    
    # Calibration ECE (чем меньше, тем лучше; нормализуем)
    if calibration_metrics and "expected_calibration_error" in calibration_metrics:
        ece = calibration_metrics["expected_calibration_error"]
        ece_score = max(0, 1 - ece * 10)  # ECE=0.1 → score=0, ECE=0 → score=1
        score_components.append(weights["calibration_ece"] * ece_score)
    
    # Area MAPE (чем меньше, тем лучше)
    if "area_mape_percent" in burned_area_metrics:
        mape = burned_area_metrics["area_mape_percent"]
        mape_score = max(0, 1 - mape / 100)  # MAPE=100% → score=0
        score_components.append(weights["area_mape_inverse"] * mape_score)
    
    overall_score = sum(score_components) if score_components else 0.0
    
    report = {
        "detection": detection_metrics,
        "burned_area": burned_area_metrics,
        "calibration": calibration_metrics or {},
        "overall_score": round(overall_score, 4),
        "score_breakdown": {
            "detection_contribution": round(score_components[0], 4) if len(score_components) > 0 else 0,
            "burned_area_contribution": round(score_components[1], 4) if len(score_components) > 1 else 0,
            "calibration_contribution": round(score_components[2], 4) if len(score_components) > 2 else 0,
            "area_accuracy_contribution": round(score_components[3], 4) if len(score_components) > 3 else 0
        }
    }
    
    return report
