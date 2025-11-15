"""OCR Calibration Module v1 - Advanced Calibration with Uncertainty

Performs pixel-to-mm calibration using ruler markings and optional OCR detection.
Handles multiple detection strategies with automatic fallback.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings

class AdvancedCalibrationMeasurement:
    """Advanced calibration and measurement for concrete analysis"""

    def __init__(self, ruler_length_mm: float = 100.0):
        """Initialize calibration module

        Args:
            ruler_length_mm: Assumed ruler length in mm (default: 100mm ruler)
        """
        self.ruler_length_mm = ruler_length_mm
        self.pixel_per_mm = None
        self.pixel_per_cm = None
        self.calibration_info = {}
        self.latest_measurements = {}

    def detect_ruler_markings_contours(self, scale_mask: np.ndarray) -> Dict:
        """Strategy 1: Detect ruler markings using contour analysis

        Args:
            scale_mask: Binary mask of ruler/scale

        Returns:
            Dictionary with detected markings and calibration info
        """
        contours, _ = cv2.findContours(scale_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        ruler_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(ruler_contour)

        ruler_pixel_length = max(w, h)

        if ruler_pixel_length == 0:
            return None

        pixel_per_mm = ruler_pixel_length / self.ruler_length_mm

        return {
            'strategy': 'contour_analysis',
            'ruler_pixel_length': ruler_pixel_length,
            'pixel_per_mm': pixel_per_mm,
            'pixel_per_cm': pixel_per_mm * 10,
            'bounding_box': (x, y, w, h),
            'confidence': 0.7
        }

    def detect_ruler_markings_projection(self, scale_mask: np.ndarray) -> Dict:
        """Strategy 2: Detect ruler markings using projection analysis

        Args:
            scale_mask: Binary mask of ruler/scale

        Returns:
            Dictionary with detected markings and calibration info
        """
        horizontal_projection = np.sum(scale_mask, axis=0)
        vertical_projection = np.sum(scale_mask, axis=1)

        horizontal_peaks = self._find_peaks(horizontal_projection)
        vertical_peaks = self._find_peaks(vertical_projection)

        num_horizontal = len(horizontal_peaks) if horizontal_peaks is not None else 0
        num_vertical = len(vertical_peaks) if vertical_peaks is not None else 0

        if num_horizontal > num_vertical:
            markings = horizontal_peaks
            ruler_axis = 'horizontal'
        else:
            markings = vertical_peaks
            ruler_axis = 'vertical'

        if markings is None or len(markings) < 2:
            return None

        marking_distance = markings[-1] - markings[0]

        estimated_cm_count = len(markings) - 1

        if estimated_cm_count <= 0:
            return None

        pixel_per_cm = marking_distance / estimated_cm_count

        return {
            'strategy': 'projection_analysis',
            'markings_detected': len(markings),
            'pixel_per_cm': pixel_per_cm,
            'pixel_per_mm': pixel_per_cm / 10,
            'ruler_orientation': ruler_axis,
            'marking_positions': markings.tolist() if hasattr(markings, 'tolist') else markings,
            'confidence': 0.8 if len(markings) > 5 else 0.6
        }

    def detect_ruler_markings_manual_threshold(self, image: np.ndarray,
                                               scale_mask: np.ndarray) -> Dict:
        """Strategy 3: Manual threshold-based detection

        Args:
            image: Input preprocessed image
            scale_mask: Binary mask of ruler/scale

        Returns:
            Dictionary with detected markings and calibration info
        """
        ruler_region = cv2.bitwise_and(image, image, mask=scale_mask)

        gray = cv2.cvtColor(ruler_region, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 2:
            return None

        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        marking_positions = [cv2.boundingRect(c)[0] for c in contours[:15]]

        if len(marking_positions) < 2:
            return None

        spacings = [marking_positions[i+1] - marking_positions[i] 
                   for i in range(len(marking_positions)-1)]
        avg_spacing = np.mean(spacings)

        pixel_per_cm = avg_spacing

        return {
            'strategy': 'manual_threshold',
            'markings_detected': len(marking_positions),
            'pixel_per_cm': pixel_per_cm,
            'pixel_per_mm': pixel_per_cm / 10,
            'marking_positions': marking_positions,
            'avg_marking_spacing': float(avg_spacing),
            'confidence': 0.6
        }

    def _find_peaks(self, signal: np.ndarray, threshold_factor: float = 0.5) -> Optional[np.ndarray]:
        """Find peaks in a 1D signal

        Args:
            signal: 1D signal array
            threshold_factor: Threshold for peak detection

        Returns:
            Array of peak positions or None
        """
        if signal is None or len(signal) == 0:
            return None

        threshold = threshold_factor * np.max(signal)
        peaks = np.where((signal > threshold) & 
                        (np.roll(signal, 1) < signal) & 
                        (np.roll(signal, -1) < signal))[0]

        if len(peaks) == 0:
            return None

        return peaks

    def auto_calibrate_advanced(self, image: np.ndarray,
                               scale_mask: np.ndarray) -> Optional[Dict]:
        """Automatic calibration with multiple fallback strategies

        Args:
            image: Preprocessed input image
            scale_mask: Binary mask of ruler/scale

        Returns:
            Calibration info dictionary or None
        """
        strategies = [
            ('Projection Analysis', lambda: self.detect_ruler_markings_projection(scale_mask)),
            ('Manual Threshold', lambda: self.detect_ruler_markings_manual_threshold(image, scale_mask)),
            ('Contour Analysis', lambda: self.detect_ruler_markings_contours(scale_mask))
        ]

        print("\n[Calibration] Testing detection strategies...")

        for strategy_name, strategy_func in strategies:
            try:
                print(f"  → Trying: {strategy_name}...")
                result = strategy_func()

                if result is not None:
                    print(f"  ✓ SUCCESS with {strategy_name}")
                    print(f"    Pixel/CM: {result.get('pixel_per_cm', 0):.2f}")
                    print(f"    Confidence: {result.get('confidence', 0):.1%}")

                    self.calibration_info = result
                    self.pixel_per_mm = result.get('pixel_per_mm', 1.0)
                    self.pixel_per_cm = result.get('pixel_per_cm', 10.0)

                    result['status'] = 'success'
                    return result
            except Exception as e:
                print(f"  ✗ Failed: {str(e)[:50]}")
                continue

        print("\n✗ All strategies failed!")
        return None

    def measure_concrete_block_with_uncertainty(self, concrete_mask: np.ndarray,
                                                image: np.ndarray) -> Optional[Dict]:
        """Measure concrete block dimensions with uncertainty

        Args:
            concrete_mask: Binary mask of concrete block
            image: Preprocessed image for visualization

        Returns:
            Dictionary with measurements
        """
        if self.pixel_per_mm is None:
            print("⚠ Calibration required first!")
            return None

        contours, _ = cv2.findContours(concrete_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(contour)
        area_pixels = cv2.contourArea(contour)

        width_mm = w * self.pixel_per_mm
        height_mm = h * self.pixel_per_mm
        area_mm2 = area_pixels * (self.pixel_per_mm ** 2)
        area_cm2 = area_mm2 / 100

        uncertainty = 0.05

        measurements = {
            'width_mm': float(width_mm),
            'height_mm': float(height_mm),
            'width_uncertainty_mm': float(width_mm * uncertainty),
            'height_uncertainty_mm': float(height_mm * uncertainty),
            'area_mm2': float(area_mm2),
            'area_cm2': float(area_cm2),
            'area_uncertainty_mm2': float(area_mm2 * uncertainty * 2),
            'perimeter_mm': float(cv2.arcLength(contour, True) * self.pixel_per_mm),
            'status': 'success'
        }

        self.latest_measurements = measurements
        return measurements

    def create_measurement_visualization(self, image: np.ndarray,
                                        concrete_mask: np.ndarray,
                                        output_path: Optional[str] = None) -> np.ndarray:
        """Create visualization with measurements

        Args:
            image: Input image
            concrete_mask: Mask of concrete block
            output_path: Optional save path

        Returns:
            Visualization image
        """
        vis_image = image.copy()

        contours, _ = cv2.findContours(concrete_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)

        if self.latest_measurements:
            meas = self.latest_measurements
            y_offset = 30
            texts = [
                f"Width: {meas.get('width_mm', 0):.1f} ± {meas.get('width_uncertainty_mm', 0):.1f} mm",
                f"Height: {meas.get('height_mm', 0):.1f} ± {meas.get('height_uncertainty_mm', 0):.1f} mm",
                f"Area: {meas.get('area_cm2', 0):.2f} cm²"
            ]

            for i, text in enumerate(texts):
                cv2.putText(vis_image, text, (10, y_offset + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if output_path:
            cv2.imwrite(output_path, vis_image)

        return vis_image

    def generate_advanced_report(self, output_path: Optional[str] = None) -> str:
        """Generate detailed measurement report

        Args:
            output_path: Optional save path

        Returns:
            Report string
        """
        report = "\n" + "="*70 + "\n"
        report += "ADVANCED MEASUREMENT AND CALIBRATION REPORT\n"
        report += "="*70 + "\n\n"

        report += "CALIBRATION INFORMATION\n"
        report += "-"*70 + "\n"
        if self.calibration_info:
            calib = self.calibration_info
            report += f"Strategy: {calib.get('strategy', 'N/A')}\n"
            report += f"Pixel/MM: {calib.get('pixel_per_mm', 0):.4f}\n"
            report += f"Pixel/CM: {calib.get('pixel_per_cm', 0):.2f}\n"
            report += f"Confidence: {calib.get('confidence', 0):.1%}\n"
        else:
            report += "No calibration performed\n"

        report += "\nMEASUREMENTS\n"
        report += "-"*70 + "\n"
        if self.latest_measurements:
            meas = self.latest_measurements
            report += f"Width: {meas.get('width_mm', 0):.2f} ± {meas.get('width_uncertainty_mm', 0):.2f} mm\n"
            report += f"Height: {meas.get('height_mm', 0):.2f} ± {meas.get('height_uncertainty_mm', 0):.2f} mm\n"
            report += f"Area: {meas.get('area_cm2', 0):.2f} cm²\n"
            report += f"Perimeter: {meas.get('perimeter_mm', 0):.2f} mm\n"
        else:
            report += "No measurements available\n"

        report += "\n" + "="*70 + "\n"

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report


if __name__ == "__main__":
    calibrator = AdvancedCalibrationMeasurement()
    print("Advanced Calibration Module Ready")