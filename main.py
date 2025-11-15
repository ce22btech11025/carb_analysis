"""
MAIN PIPELINE v3 DEBUG - WITH STEP-BY-STEP CARBONATION VISUALIZATION

Pipeline:
1. Image Preprocessing
2. GPU-Accelerated Segmentation (SAM2)
3. Adaptive Calibration
4. Precision Measurement
5. CARBONATION ANALYSIS with DEBUG OUTPUTS (NEW!)
6. Comprehensive Reporting
"""

import cv2
import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np

try:
    from image_preprocessor import ImagePreprocessor
    from segmentation_module import SegmentationModule
    from ocr_calibration_v1 import AdvancedCalibrationMeasurement
    from carbonation_analyzer_debug import CarbonationAnalyzer
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class ConcreteAnalysisPipelineV3Debug:
    """Complete analysis pipeline with carbonation detection and debug outputs"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create debug subdirectory for carbonation analysis
        self.carbonation_debug_dir = self.output_dir / "carbonation_debug"
        self.carbonation_debug_dir.mkdir(exist_ok=True)

        self.preprocessor = ImagePreprocessor()
        self.segmenter = SegmentationModule()
        self.calibrator = AdvancedCalibrationMeasurement()
        self.carbonation_analyzer = CarbonationAnalyzer(
            debug_output_dir=str(self.carbonation_debug_dir)
        )
        self.results = {
            'preprocessing': None,
            'segmentation': None,
            'calibration': None,
            'measurements': None,
            'carbonation': None,
            'analysis': None
        }

    def _print_header(self, title: str) -> None:
        """Print formatted header"""
        print("\n" + "="*70)
        print(f" {title}")
        print("="*70)

    def _print_subheader(self, title: str) -> None:
        """Print formatted subheader"""
        print(f"\n>>> {title} <<<\n")

    def stage_1_preprocessing(self, image_path: str) -> Optional[np.ndarray]:
        """Stage 1: Image Preprocessing"""
        self._print_header("STAGE 1: IMAGE PREPROCESSING")
        print("-" * 70)

        try:
            self.preprocessed_image = self.preprocessor.preprocess(
                image_path=image_path,
                save_path=str(self.output_dir / "01_preprocessed.jpg")
            )
            self.results['preprocessing'] = {
                'status': 'success',
                'shape': self.preprocessed_image.shape,
                'saved': str(self.output_dir / "01_preprocessed.jpg")
            }
            print("✓ Stage 1 complete")
            return self.preprocessed_image
        except Exception as e:
            print(f"✗ Stage 1 failed: {e}")
            self.results['preprocessing'] = {'status': 'failed', 'error': str(e)}
            return None

    def stage_2_segmentation(self, preprocessed_image: np.ndarray) -> Optional[Dict]:
        """Stage 2: GPU-Accelerated Segmentation"""
        self._print_header("STAGE 2: GPU-ACCELERATED SEGMENTATION")
        print("-" * 70)

        device_info = self.segmenter.get_device_info()
        print(f"Device: {device_info['device'].upper()}")
        if device_info['device'] == 'cuda':
            print(f"GPU: {device_info['device_name']}")
            print(f"Memory: {device_info['memory_gb']:.2f} GB")
        print()

        try:
            segmentation_results = self.segmenter.segment_and_extract(preprocessed_image)

            if segmentation_results.get('masks'):
                self.segmenter.visualize_segmentation(
                    preprocessed_image,
                    segmentation_results['masks'],
                    output_path=str(self.output_dir / "02_segmentation.jpg")
                )

                for name, mask in segmentation_results.get('masks', {}).items():
                    cv2.imwrite(str(self.output_dir / f"mask_{name}.jpg"), mask)

            self.results['segmentation'] = {
                'status': 'success',
                'masks_found': list(segmentation_results.get('masks', {}).keys()),
                'saved': str(self.output_dir / "02_segmentation.jpg")
            }
            print("✓ Stage 2 complete")
            return segmentation_results
        except Exception as e:
            print(f"✗ Stage 2 failed: {e}")
            self.results['segmentation'] = {'status': 'failed', 'error': str(e)}
            return None

    def stage_3_adaptive_calibration(self,
                                    preprocessed_image: np.ndarray,
                                    scale_mask: np.ndarray) -> Optional[Dict]:
        """Stage 3: Adaptive Calibration with Multiple Strategies"""
        self._print_header("STAGE 3: ADAPTIVE CALIBRATION")
        print("-" * 70)
        self._print_subheader("Multiple Detection Strategies")
        print("✓ Strategy 1: Projection-based peak detection")
        print("✓ Strategy 2: Manual threshold detection")
        print("✓ Strategy 3: Contour-based marking detection")
        print("✓ Automatic fallback between strategies")

        try:
            calibration_info = self.calibrator.auto_calibrate_advanced(
                preprocessed_image,
                scale_mask
            )

            if calibration_info is None:
                print("\n✗ Calibration failed!")
                return None

            self.results['calibration'] = calibration_info
            return calibration_info
        except Exception as e:
            print(f"\n✗ Stage 3 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['calibration'] = {'status': 'failed', 'error': str(e)}
            return None

    def stage_4_measurement(self,
                           preprocessed_image: np.ndarray,
                           segmentation_results: Dict) -> Optional[Dict]:
        """Stage 4: Precision Measurement"""
        self._print_header("STAGE 4: PRECISION MEASUREMENT")
        print("-" * 70)

        try:
            concrete_mask = segmentation_results['masks'].get('concrete_block')
            if concrete_mask is None:
                print("⚠ Concrete block not detected")
                return None

            print("\n[Measurement] Computing block dimensions...")
            measurements = self.calibrator.measure_concrete_block_with_uncertainty(
                concrete_mask,
                preprocessed_image
            )

            if measurements is None:
                print("⚠ Measurement failed")
                return None

            print("\n[Visualization] Creating analysis visualization...")
            vis_image = self.calibrator.create_measurement_visualization(
                preprocessed_image,
                concrete_mask,
                output_path=str(self.output_dir / "03_measurements.jpg")
            )

            self.results['measurements'] = measurements
            print("✓ Stage 4 complete")
            return measurements
        except Exception as e:
            print(f"✗ Stage 4 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['measurements'] = {'status': 'failed', 'error': str(e)}
            return None

    def stage_5_carbonation_analysis_debug(self,
                                          preprocessed_image: np.ndarray,
                                          segmentation_results: Dict) -> Optional[Dict]:
        """Stage 5: Carbonation Analysis with DEBUG OUTPUTS"""
        self._print_header("STAGE 5: CARBONATION ANALYSIS WITH DEBUG OUTPUTS")
        print("-" * 70)

        try:
            concrete_mask = segmentation_results['masks'].get('concrete_block')
            if concrete_mask is None:
                print("⚠ Concrete block not detected")
                return None

            print(f"\nDebug output directory: {self.carbonation_debug_dir}")
            print("Step-by-step images will be saved to: carbonation_debug/")
            print("\n[Carbonation] Analyzing non-carbonated regions...")
            print("  → Method: Phenolphthalein coloration (magenta detection)")
            print("  → Algorithm: Choi et al. (2017)")
            print("  → All stages will output debug images\n")

            carbonation_results = self.carbonation_analyzer.analyze_carbonation(
                preprocessed_image,
                concrete_mask,
                use_hsv=True
            )

            print(f"\n  ✓ Non-carbonated area: {carbonation_results['non_carbonated_percentage']:.2f}%")
            print(f"  ✓ Carbonated area: {carbonation_results['carbonated_percentage']:.2f}%")

            # Create visualization
            print("\n[Visualization] Creating carbonation visualization...")
            vis_image = self.carbonation_analyzer.create_visualization(
                preprocessed_image,
                carbonation_results['non_carbonated_mask'],
                concrete_mask,
                output_path=str(self.output_dir / "04_carbonation_analysis.jpg")
            )

            # Generate report
            pixel_to_mm = self.calibrator.pixel_per_mm
            carbonation_report = self.carbonation_analyzer.generate_carbonation_report(
                pixel_to_mm=pixel_to_mm,
                output_path=str(self.output_dir / "carbonation_report.txt")
            )

            self.results['carbonation'] = carbonation_results

            print("\n" + "="*70)
            print("✓ Stage 5 complete")
            print("="*70)
            print(f"\nDebug images saved in: {self.carbonation_debug_dir}/")
            print("\nImagefiles generated:")
            debug_files = sorted(self.carbonation_debug_dir.glob("*.jpg"))
            for i, f in enumerate(debug_files, 1):
                print(f"  {i:2d}. {f.name}")

            return carbonation_results
        except Exception as e:
            print(f"✗ Stage 5 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['carbonation'] = {'status': 'failed', 'error': str(e)}
            return None

    def stage_6_reporting(self) -> str:
        """Stage 6: Comprehensive Reporting"""
        self._print_header("STAGE 6: COMPREHENSIVE REPORTING")
        print("-" * 70)

        try:
            report = self.calibrator.generate_advanced_report(
                output_path=str(self.output_dir / "calibration_report.txt")
            )

            print("\n" + report)

            # Save JSON results
            json_path = str(self.output_dir / "results.json")
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)

            print(f"\n✓ JSON results saved to {json_path}")

            self._print_subheader("ANALYSIS SUMMARY")

            if self.results.get('calibration', {}).get('status') != 'failed':
                calib = self.results['calibration']
                print(f"Calibration Method: {calib.get('strategy', 'N/A')}")
                print(f"Pixel/CM: {calib.get('pixel_per_cm', 0):.2f}")
                print(f"Confidence: {calib.get('confidence', 0):.1%}")

            if self.results.get('measurements', {}).get('status') != 'failed':
                meas = self.results['measurements']
                print(f"\nBlock Dimensions:")
                print(f"  Width: {meas.get('width_mm', 0):.2f} mm")
                print(f"  Height: {meas.get('height_mm', 0):.2f} mm")
                print(f"  Area: {meas.get('area_cm2', 0):.2f} cm²")

            if self.results.get('carbonation', {}).get('status') != 'failed':
                carb = self.results['carbonation']
                print(f"\nCarbonation Analysis:")
                print(f"  Non-carbonated: {carb.get('non_carbonated_percentage', 0):.2f}%")
                print(f"  Carbonated: {carb.get('carbonated_percentage', 0):.2f}%")

            print("\n✓ Stage 6 complete")
            return report
        except Exception as e:
            print(f"✗ Stage 6 failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def run_pipeline(self, image_path: str) -> bool:
        """Run complete pipeline"""
        self._print_header("CONCRETE BLOCK ANALYSIS PIPELINE v3 DEBUG")
        print("Complete analysis with STEP-BY-STEP CARBONATION DEBUG OUTPUTS")
        print(f"\nInput image: {image_path}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Debug directory: {self.carbonation_debug_dir.absolute()}")

        # Stage 1
        preprocessed_image = self.stage_1_preprocessing(image_path)
        if preprocessed_image is None:
            return False

        # Stage 2
        segmentation_results = self.stage_2_segmentation(preprocessed_image)
        if segmentation_results is None:
            return False

        # Stage 3
        scale_mask = segmentation_results.get('masks', {}).get('scale')
        if scale_mask is None:
            print("✗ No scale detected!")
            return False

        calibration_info = self.stage_3_adaptive_calibration(preprocessed_image, scale_mask)
        if calibration_info is None:
            return False

        # Stage 4
        measurements = self.stage_4_measurement(preprocessed_image, segmentation_results)
        if measurements is None:
            return False

        # Stage 5 - NEW CARBONATION ANALYSIS WITH DEBUG
        carbonation = self.stage_5_carbonation_analysis_debug(preprocessed_image, segmentation_results)

        # Stage 6
        report = self.stage_6_reporting()

        self._print_header("PIPELINE COMPLETE ✓")
        print("All outputs saved to:", self.output_dir.absolute())
        print("\nDebug images for carbonation analysis saved to:")
        print(f"  {self.carbonation_debug_dir.absolute()}")
        print("\nYou can now examine each stage to find where the bug is!")

        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_v3_debug.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    pipeline = ConcreteAnalysisPipelineV3Debug(output_dir=output_dir)
    success = pipeline.run_pipeline(image_path)

    sys.exit(0 if success else 1)