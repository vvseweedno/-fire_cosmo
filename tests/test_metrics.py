"""
Тесты для модуля метрик.
"""

import pytest
import numpy as np
from app.services.metrics import (
    calculate_precision_recall_f1,
    calculate_false_positive_rate,
    calculate_iou_dice,
    calculate_area_metrics,
    calculate_brier_score,
    calculate_expected_calibration_error,
    evaluate_detection,
    evaluate_burned_area,
    create_evaluation_report
)


class TestPrecisionRecallF1:
    """Тесты для расчёта precision, recall, F1"""
    
    def test_perfect_prediction(self):
        """Идеальное предсказание"""
        y_true = [1, 1, 1, 0, 0]
        y_pred = [1, 1, 1, 0, 0]
        
        precision, recall, f1 = calculate_precision_recall_f1(y_true, y_pred)
        
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0
    
    def test_all_false_positives(self):
        """Все предсказания - ложные срабатывания"""
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]
        
        precision, recall, f1 = calculate_precision_recall_f1(y_true, y_pred)
        
        assert precision == 0.0
        assert recall == 0.0  # Нет истинных положительных
        assert f1 == 0.0
    
    def test_mixed_case(self):
        """Смешанный случай"""
        y_true = [1, 1, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1]
        
        # TP=2, FP=1, FN=1
        precision, recall, f1 = calculate_precision_recall_f1(y_true, y_pred)
        
        assert precision == 2/3  # 2/(2+1)
        assert recall == 2/3     # 2/(2+1)
        assert abs(f1 - 0.6667) < 0.001
    
    def test_empty_input(self):
        """Пустой вход"""
        precision, recall, f1 = calculate_precision_recall_f1([], [])
        
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0


class TestFalsePositiveRate:
    """Тесты для FPR"""
    
    def test_no_false_positives(self):
        """Нет ложных срабатываний"""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        
        fpr = calculate_false_positive_rate(y_true, y_pred)
        assert fpr == 0.0
    
    def test_all_false_positives(self):
        """Все негативные классифицированы как позитивные"""
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 1, 1]
        
        fpr = calculate_false_positive_rate(y_true, y_pred)
        assert fpr == 1.0
    
    def test_mixed(self):
        """Смешанный случай"""
        y_true = [0, 0, 0, 1, 1]
        y_pred = [1, 0, 0, 1, 1]
        
        # FP=1, TN=2 → FPR = 1/(1+2) = 1/3
        fpr = calculate_false_positive_rate(y_true, y_pred)
        assert abs(fpr - 0.3333) < 0.001


class TestIoUDice:
    """Тесты для IoU и Dice coefficient"""
    
    def test_perfect_overlap(self):
        """Идеальное совпадение"""
        mask_true = np.array([1, 1, 1, 1])
        mask_pred = np.array([1, 1, 1, 1])
        
        iou, dice = calculate_iou_dice(mask_true, mask_pred)
        
        assert iou == 1.0
        assert dice == 1.0
    
    def test_no_overlap(self):
        """Нет перекрытия"""
        mask_true = np.array([1, 1, 0, 0])
        mask_pred = np.array([0, 0, 1, 1])
        
        iou, dice = calculate_iou_dice(mask_true, mask_pred)
        
        assert iou == 0.0
        assert dice == 0.0
    
    def test_partial_overlap(self):
        """Частичное перекрытие"""
        mask_true = np.array([1, 1, 1, 0])
        mask_pred = np.array([0, 1, 1, 1])
        
        # Intersection = 2, Union = 4, Total_true = 3, Total_pred = 3
        iou, dice = calculate_iou_dice(mask_true, mask_pred)
        
        assert iou == 0.5  # 2/4
        assert abs(dice - 0.6667) < 0.001  # 2*2/(3+3)
    
    def test_empty_masks(self):
        """Пустые маски"""
        mask_true = np.array([0, 0, 0])
        mask_pred = np.array([0, 0, 0])
        
        iou, dice = calculate_iou_dice(mask_true, mask_pred)
        
        assert iou == 0.0
        assert dice == 0.0


class TestAreaMetrics:
    """Тесты для метрик площади"""
    
    def test_perfect_areas(self):
        """Идеальное совпадение площадей"""
        areas_true = [100.0, 200.0, 300.0]
        areas_pred = [100.0, 200.0, 300.0]
        
        mae, mape = calculate_area_metrics(areas_true, areas_pred)
        
        assert mae == 0.0
        assert mape == 0.0
    
    def test_constant_error(self):
        """Постоянная ошибка"""
        areas_true = [100.0, 200.0, 300.0]
        areas_pred = [110.0, 210.0, 310.0]  # +10 га
        
        mae, mape = calculate_area_metrics(areas_true, areas_pred)
        
        assert mae == 10.0
        # MAPE = (10/100 + 10/200 + 10/300) / 3 * 100 = (0.1 + 0.05 + 0.033) / 3 * 100 ≈ 6.11%
        assert abs(mape - 6.11) < 0.1
    
    def test_empty_input(self):
        """Пустой вход"""
        mae, mape = calculate_area_metrics([], [])
        
        assert mae == 0.0
        assert mape == 0.0


class TestBrierScore:
    """Тесты для Brier score"""
    
    def test_perfect_probabilities(self):
        """Идеальные вероятности"""
        y_true = [1, 1, 0, 0]
        y_prob = [1.0, 1.0, 0.0, 0.0]
        
        brier = calculate_brier_score(y_true, y_prob)
        assert brier == 0.0
    
    def test_worst_probabilities(self):
        """Худшие вероятности"""
        y_true = [1, 1, 0, 0]
        y_prob = [0.0, 0.0, 1.0, 1.0]
        
        # (1-0)^2 + (1-0)^2 + (0-1)^2 + (0-1)^2 = 4, mean = 1.0
        brier = calculate_brier_score(y_true, y_prob)
        assert brier == 1.0
    
    def test_uncertain(self):
        """Неопределённые предсказания"""
        y_true = [1, 0, 1, 0]
        y_prob = [0.5, 0.5, 0.5, 0.5]
        
        # Все ошибки = 0.25, среднее = 0.25
        brier = calculate_brier_score(y_true, y_prob)
        assert abs(brier - 0.25) < 0.001


class TestExpectedCalibrationError:
    """Тесты для ECE"""
    
    def test_perfectly_calibrated(self):
        """Идеальная калибровка"""
        y_true = [1, 1, 0, 0]
        y_prob = [1.0, 1.0, 0.0, 0.0]
        
        ece, _ = calculate_expected_calibration_error(y_true, y_prob)
        assert ece < 0.001
    
    def test_miscalibrated(self):
        """Плохая калибровка"""
        y_true = [0, 0, 0, 0]
        y_prob = [0.9, 0.9, 0.9, 0.9]  # Уверен но неверно
        
        ece, _ = calculate_expected_calibration_error(y_true, y_prob)
        assert ece > 0.5  # Большая ошибка калибровки


class TestEvaluateDetection:
    """Интеграционные тесты для evaluate_detection"""
    
    def test_full_evaluation(self):
        """Полная оценка детекции"""
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [1, 1, 0, 0, 0, 1]
        y_prob = [0.9, 0.8, 0.4, 0.2, 0.3, 0.6]
        
        metrics = evaluate_detection(y_true, y_pred, y_prob)
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "brier_score" in metrics
        assert "expected_calibration_error" in metrics
        assert metrics["n_samples"] == 6
    
    def test_without_probabilities(self):
        """Оценка без вероятностей"""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 1]
        
        metrics = evaluate_detection(y_true, y_pred)
        
        assert "precision" in metrics
        assert "brier_score" not in metrics


class TestCreateEvaluationReport:
    """Тесты для сводного отчёта"""
    
    def test_full_report(self):
        """Полный отчёт"""
        detection = {"f1": 0.85, "precision": 0.87, "recall": 0.83}
        burned = {"iou": 0.72, "area_mae_ha": 25.0, "area_mape_percent": 15.0}
        calibration = {"expected_calibration_error": 0.08}
        
        report = create_evaluation_report(detection, burned, calibration)
        
        assert "overall_score" in report
        assert "score_breakdown" in report
        assert report["overall_score"] > 0
        assert report["overall_score"] <= 1.0
    
    def test_minimal_report(self):
        """Минимальный отчёт"""
        detection = {"f1": 0.8}
        burned = {"iou": 0.6}
        
        report = create_evaluation_report(detection, burned)
        
        assert "overall_score" in report
        assert report["overall_score"] > 0
