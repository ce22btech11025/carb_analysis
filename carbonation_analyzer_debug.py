"""Carbonation Analysis Module with Step-by-Step Image Outputs

Detects and analyzes non-carbonated regions in concrete using phenolphthalein coloration.
Algorithm based on: Choi et al., Construction and Building Materials (2017)

Features DEBUGGING OUTPUT - saves images at every stage for troubleshooting
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

class CarbonationAnalyzer:
    """Analyzes concrete carbonation using phenolphthalein coloration with debug outputs"""

    def __init__(self, debug_output_dir: Optional[str] = None):
        """Initialize analyzer with optional debug output directory

        Args:
            debug_output_dir: Directory to save debug images at each stage
        """
        self.original_image = None
        self.concrete_mask = None
        self.carbonation_results = {}
        self.debug_dir = None

        if debug_output_dir:
            self.debug_dir = Path(debug_output_dir)
            self.debug_dir.mkdir(exist_ok=True, parents=True)
            print(f"✓ Debug output enabled: {self.debug_dir}")

    def _save_debug_image(self, image: np.ndarray, stage_name: str, description: str = "") -> None:
        """Save debug image at current stage

        Args:
            image: Image to save
            stage_name: Name of the stage (will be used in filename)
            description: Additional description
        """
        if self.debug_dir is None:
            return

        filename = f"{stage_name}.jpg"
        filepath = self.debug_dir / filename
        cv2.imwrite(str(filepath), image)
        print(f"  ✓ Debug: {filename} saved")

    def _extract_green_channel(self, image: np.ndarray) -> np.ndarray:
        """Extract green channel (complementary to magenta phenolphthalein color)

        Args:
            image: Input BGR image

        Returns:
            Green channel as grayscale image
        """
        print("\n[Stage 1/8] Extracting Green Channel...")
        green_channel = image[:, :, 1]

        self._save_debug_image(green_channel, "01_green_channel", 
                              "Green channel extracted (magenta appears dark)")

        return green_channel

    def _invert_green(self, green_channel: np.ndarray) -> np.ndarray:
        """Invert green channel to highlight magenta regions

        Args:
            green_channel: Green channel image

        Returns:
            Inverted green channel
        """
        print("[Stage 2/8] Inverting Green Channel...")
        inverted = cv2.bitwise_not(green_channel)

        self._save_debug_image(inverted, "02_inverted_green",
                              "Inverted green (magenta regions now bright)")

        return inverted

    def _binary_threshold(self, inverted: np.ndarray, threshold: int = 100) -> np.ndarray:
        """Apply binary threshold to create initial binary image

        Args:
            inverted: Inverted green channel
            threshold: Threshold value

        Returns:
            Binary image
        """
        print(f"[Stage 3/8] Applying Binary Threshold (value={threshold})...")
        _, binary = cv2.threshold(inverted, threshold, 255, cv2.THRESH_BINARY)

        self._save_debug_image(binary, "03_binary_threshold",
                              "Binary image (magenta regions = white)")

        return binary

    def _morphological_close(self, binary: np.ndarray) -> np.ndarray:
        """Apply morphological close operation (fill small holes)

        Args:
            binary: Binary image

        Returns:
            Closed binary image
        """
        print("[Stage 4/8] Morphological Close Operation (fill holes)...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        self._save_debug_image(closed, "04_morphological_close",
                              "Holes filled (close operation)")

        return closed

    def _morphological_open(self, closed: np.ndarray) -> np.ndarray:
        """Apply morphological open operation (remove small noise)

        Args:
            closed: Closed binary image

        Returns:
            Opened binary image
        """
        print("[Stage 5/8] Morphological Open Operation (remove noise)...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        self._save_debug_image(opened, "05_morphological_open",
                              "Noise removed (open operation)")

        return opened

    def _connected_components(self, opened: np.ndarray) -> Tuple[np.ndarray, int]:
        """Label connected components

        Args:
            opened: Opened binary image

        Returns:
            Tuple of (labeled image, number of labels)
        """
        print("[Stage 6/8] Connected Component Labeling...")
        num_labels, labeled_image = cv2.connectedComponents(opened)

        # Visualize labels with colors
        labeled_vis = np.uint8(255 * labeled_image / max(num_labels, 1))
        self._save_debug_image(labeled_vis, "06_connected_components",
                              f"Connected components (found {num_labels} regions)")

        return labeled_image, num_labels

    def _convex_hull_refinement(self, binary_image: np.ndarray) -> np.ndarray:
        """Apply convex hull refinement to smooth boundaries

        Args:
            binary_image: Binary image from primary detection

        Returns:
            Refined binary image with filled convex regions
        """
        print("[Stage 7/8] Convex Hull Refinement...")
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)

        print(f"  Found {len(contours)} contours")

        # Create visualization of contours before hull
        contour_vis = np.zeros_like(binary_image)
        cv2.drawContours(contour_vis, contours, -1, 255, 1)
        self._save_debug_image(contour_vis, "07a_contours_before_hull",
                              f"Detected contours ({len(contours)} regions)")

        if not contours:
            return binary_image

        # Create empty image for convex hulls
        hull_image = np.zeros_like(binary_image)

        # Draw and fill convex hulls
        for i, contour in enumerate(contours):
            hull = cv2.convexHull(contour)
            cv2.drawContours(hull_image, [hull], 0, 255, -1)

        self._save_debug_image(hull_image, "07b_convex_hull_filled",
                              "Convex hulls computed and filled")

        return hull_image

    def _apply_concrete_mask(self, refined_mask: np.ndarray, 
                           concrete_mask: np.ndarray) -> np.ndarray:
        """Apply concrete block mask to isolate results

        Args:
            refined_mask: Refined non-carbonated mask
            concrete_mask: Binary mask of concrete block

        Returns:
            Final mask with concrete boundary applied
        """
        print("[Stage 8/8] Applying Concrete Block Mask...")
        final_mask = cv2.bitwise_and(refined_mask, concrete_mask)

        self._save_debug_image(final_mask, "08_final_masked",
                              "Final mask (constrained to concrete block)")

        return final_mask

    def _segment_magenta_regions_hsv(self, image: np.ndarray, 
                                     concrete_mask: np.ndarray) -> Dict:
        """Segment magenta regions using HSV color space

        Args:
            image: Input BGR image
            concrete_mask: Binary mask of concrete block

        Returns:
            Dictionary with segmentation results
        """
        print("\n[HSV Method] Detecting Magenta Color...\n")

        # Convert to HSV
        print("[HSV-1/3] Converting to HSV color space...")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Save HSV channels for visualization
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        self._save_debug_image(h_channel, "hsv_01_hue_channel",
                              "HSV Hue channel (magenta ~130-170)")
        self._save_debug_image(s_channel, "hsv_02_saturation_channel",
                              "HSV Saturation channel")
        self._save_debug_image(v_channel, "hsv_03_value_channel",
                              "HSV Value channel")

        # Define magenta range
        print("[HSV-2/3] Creating magenta mask (H=130-170, S=40-255, V=40-255)...")
        lower_magenta = np.array([130, 40, 40])
        upper_magenta = np.array([170, 255, 255])

        magenta_mask = cv2.inRange(hsv, lower_magenta, upper_magenta)

        self._save_debug_image(magenta_mask, "hsv_04_magenta_mask",
                              "Magenta mask (before concrete constraint)")

        # Apply concrete mask
        print("[HSV-3/3] Applying concrete block constraint...")
        non_carbonated_mask = cv2.bitwise_and(magenta_mask, concrete_mask)

        self._save_debug_image(non_carbonated_mask, "hsv_05_masked_magenta",
                              "Magenta mask (within concrete block)")

        return {
            'magenta_mask': magenta_mask,
            'non_carbonated_mask': non_carbonated_mask
        }

    def _primary_detection_pipeline(self, green_channel: np.ndarray, 
                                   concrete_mask: np.ndarray,
                                   threshold: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Run complete primary detection pipeline with image outputs

        Args:
            green_channel: Green channel image
            concrete_mask: Concrete block mask
            threshold: Threshold value

        Returns:
            Tuple of (binary image, labeled image)
        """
        print("\n" + "="*60)
        print("PRIMARY DETECTION PIPELINE")
        print("="*60)

        # Stage 1: Extract (already done)
        # Stage 2: Invert
        inverted = self._invert_green(green_channel)

        # Stage 3: Binary threshold
        binary = self._binary_threshold(inverted, threshold)

        # Stage 4: Morphological close
        closed = self._morphological_close(binary)

        # Stage 5: Morphological open
        opened = self._morphological_open(closed)

        # Stage 6: Connected components
        labeled_image, num_labels = self._connected_components(opened)

        # Stage 7: Convex hull refinement
        refined_mask = self._convex_hull_refinement(opened)

        # Stage 8: Apply concrete mask
        final_mask = self._apply_concrete_mask(refined_mask, concrete_mask)

        return final_mask, labeled_image

    def analyze_carbonation(self, image: np.ndarray, concrete_mask: np.ndarray,
                           threshold: int = 100, 
                           use_hsv: bool = True) -> Dict:
        """Complete carbonation analysis pipeline with debug outputs

        Args:
            image: Input preprocessed image
            concrete_mask: Binary mask of concrete block
            threshold: Threshold for green channel binarization
            use_hsv: Use HSV-based detection (recommended)

        Returns:
            Dictionary with analysis results
        """
        self.original_image = image
        self.concrete_mask = concrete_mask

        print("\n" + "="*60)
        print("CARBONATION ANALYSIS - STEP BY STEP")
        print("="*60)

        # Save original image
        self._save_debug_image(image, "00_original_image",
                              "Original preprocessed image")

        self._save_debug_image(concrete_mask * 255, "00b_concrete_mask",
                              "Concrete block mask (binary)")

        # Stage 1: Extract green channel
        green_channel = self._extract_green_channel(image)

        # Primary detection
        if use_hsv:
            print("\n[Using HSV Detection]")
            hsv_results = self._segment_magenta_regions_hsv(image, concrete_mask)
            non_carbonated_mask = hsv_results['non_carbonated_mask']
        else:
            print("\n[Using Green Channel Detection]")
            non_carbonated_mask, _ = self._primary_detection_pipeline(
                green_channel, concrete_mask, threshold
            )

        # Apply secondary detection (convex hull refinement if not already done)
        if use_hsv:
            print("\n[Applying Convex Hull Refinement to HSV Result]...")
            refined_mask = self._convex_hull_refinement(non_carbonated_mask)
        else:
            refined_mask = non_carbonated_mask

        # Calculate areas
        print("\n" + "="*60)
        print("CALCULATING STATISTICS")
        print("="*60)

        concrete_area = np.sum(concrete_mask > 0)
        non_carbonated_area = np.sum(refined_mask > 0)
        carbonated_area = concrete_area - non_carbonated_area

        non_carbonated_percentage = (non_carbonated_area / concrete_area * 100) if concrete_area > 0 else 0
        carbonated_percentage = 100 - non_carbonated_percentage

        print(f"\n  Concrete Area: {concrete_area:,} pixels")
        print(f"  Non-Carbonated Area: {non_carbonated_area:,} pixels")
        print(f"  Carbonated Area: {carbonated_area:,} pixels")
        print(f"  Non-Carbonated: {non_carbonated_percentage:.2f}%")
        print(f"  Carbonated: {carbonated_percentage:.2f}%")

        # Store results
        self.carbonation_results = {
            'concrete_area_pixels': int(concrete_area),
            'non_carbonated_area_pixels': int(non_carbonated_area),
            'carbonated_area_pixels': int(carbonated_area),
            'non_carbonated_percentage': float(non_carbonated_percentage),
            'carbonated_percentage': float(carbonated_percentage),
            'non_carbonated_mask': refined_mask,
            'magenta_mask': hsv_results.get('magenta_mask') if use_hsv else None
        }

        print("\n✓ Carbonation analysis complete")

        return self.carbonation_results

    def create_visualization(self, image: np.ndarray, 
                            non_carbonated_mask: np.ndarray,
                            concrete_mask: np.ndarray,
                            output_path: Optional[str] = None) -> np.ndarray:
        """Create visualization with overlays and save all intermediate visualizations

        Args:
            image: Original image
            non_carbonated_mask: Mask of non-carbonated regions
            concrete_mask: Mask of concrete block
            output_path: Optional path to save visualization

        Returns:
            Visualization image
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)

        vis_image = image.copy()

        # Draw concrete block boundary (green)
        print("\n[Viz-1/4] Drawing concrete block boundary...")
        contours_concrete, _ = cv2.findContours(concrete_mask, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours_concrete, -1, (0, 255, 0), 2)
        self._save_debug_image(vis_image, "09_viz_concrete_boundary",
                              "Visualization: Concrete block boundary (green)")

        # Create mask visualization
        print("[Viz-2/4] Creating non-carbonated overlay...")
        overlay = vis_image.copy()
        overlay[non_carbonated_mask > 0] = (255, 0, 0)  # Blue for non-carbonated
        vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
        self._save_debug_image(vis_image, "10_viz_overlay",
                              "Visualization: Non-carbonated overlay (blue)")

        # Add text information
        print("[Viz-3/4] Adding text annotations...")
        if self.carbonation_results:
            results = self.carbonation_results
            text_lines = [
                f"Non-carbonated: {results['non_carbonated_percentage']:.2f}%",
                f"Carbonated: {results['carbonated_percentage']:.2f}%"
            ]

            y_offset = 30
            for i, line in enumerate(text_lines):
                cv2.putText(vis_image, line, (10, y_offset + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self._save_debug_image(vis_image, "11_viz_with_text",
                              "Visualization: With percentages")

        # Save to specified path if provided
        if output_path:
            print(f"[Viz-4/4] Saving to {output_path}...")
            cv2.imwrite(output_path, vis_image)

        print("\n✓ All visualizations complete")

        return vis_image

    def generate_carbonation_report(self, pixel_to_mm: Optional[float] = None,
                                   output_path: Optional[str] = None) -> str:
        """Generate text report of carbonation analysis

        Args:
            pixel_to_mm: Optional pixel to mm conversion ratio
            output_path: Optional path to save report

        Returns:
            Report string
        """
        if not self.carbonation_results:
            return "No carbonation analysis results available"

        results = self.carbonation_results

        report = "\n" + "="*60 + "\n"
        report += "CONCRETE CARBONATION ANALYSIS REPORT\n"
        report += "="*60 + "\n\n"

        report += "AREA MEASUREMENTS\n"
        report += "-"*60 + "\n"
        report += f"Total Concrete Area: {results['concrete_area_pixels']:,} pixels\n"
        report += f"Non-Carbonated Area: {results['non_carbonated_area_pixels']:,} pixels\n"
        report += f"Carbonated Area: {results['carbonated_area_pixels']:,} pixels\n\n"

        report += "CARBONATION PERCENTAGES\n"
        report += "-"*60 + "\n"
        report += f"Non-Carbonated: {results['non_carbonated_percentage']:.2f}%\n"
        report += f"Carbonated: {results['carbonated_percentage']:.2f}%\n\n"

        if pixel_to_mm:
            report += "DEPTH ANALYSIS\n"
            report += "-"*60 + "\n"
            concrete_area_mm2 = results['concrete_area_pixels'] * (pixel_to_mm ** 2)
            non_carbonated_area_mm2 = results['non_carbonated_area_pixels'] * (pixel_to_mm ** 2)
            report += f"Concrete Area: {concrete_area_mm2:.2f} mm²\n"
            report += f"Non-Carbonated Area: {non_carbonated_area_mm2:.2f} mm²\n\n"

        report += "METHODOLOGY\n"
        report += "-"*60 + "\n"
        report += "Algorithm: Choi et al. (2017)\n"
        report += "- Stage 1: Green channel extraction\n"
        report += "- Stage 2: Inversion (magenta→bright)\n"
        report += "- Stage 3: Binary threshold (100)\n"
        report += "- Stage 4: Morphological close (7x7, 2x)\n"
        report += "- Stage 5: Morphological open (7x7, 1x)\n"
        report += "- Stage 6: Connected component labeling\n"
        report += "- Stage 7: Convex hull refinement\n"
        report += "- Stage 8: Concrete mask application\n"
        report += "- Color: Magenta (phenolphthalein) = non-carbonated\n"
        report += "- Color: Gray/Colorless = carbonated region\n"

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report


if __name__ == "__main__":
    analyzer = CarbonationAnalyzer(debug_output_dir="debug_output")
    print("Carbonation Analyzer Module with Debug Output Ready")