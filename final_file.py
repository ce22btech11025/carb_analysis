import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from skimage import morphology, measure
from pathlib import Path

class ConcreteCarbonationAnalyzer:
    """
    Implements image-processing technique to detect carbonation regions of concrete
    sprayed with phenolphthalein solution (Choi et al., 2017)
    """
    
    def __init__(self, image_path, reference_side_mm=50):
        """
        Initialize analyzer with image and reference measurement
        
        Args:
            image_path: Path to concrete block image
            reference_side_mm: Reference side length in mm (user provides this)
        """
        self.image_path = image_path
        self.reference_side_mm = reference_side_mm
        self.original_image = cv2.imread(image_path)
        self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        self.green_channel = None
        self.binary_green = None
        self.complementary = None
        self.holes_filled = None
        self.cleaned_binary = None
        self.convex_hull_result = None
        self.final_carbonated = None
        self.pixel_to_mm_ratio = None
        self.concrete_boundary = None
        
        # Create output directory
        self.output_dir = Path("carbonation_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def save_step(self, image, step_name):
        """Save intermediate processing step as image"""
        output_path = self.output_dir / f"{step_name}.png"
        if len(image.shape) == 2:  # Grayscale
            cv2.imwrite(str(output_path), image * 255 if image.max() <= 1 else image)
        else:  # Color
            cv2.imwrite(str(output_path), image)
        print(f"✓ Saved: {step_name}.png")
        return output_path
    
    def step1_detect_concrete_boundary(self):
        """
        Step 1: Detect concrete block boundaries using color-based segmentation
        Returns pixel dimensions to establish pixel-to-mm ratio
        """
        print("\n" + "="*60)
        print("STEP 1: DETECT CONCRETE BLOCK BOUNDARY")
        print("="*60)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for concrete (light gray areas)
        # Concrete is typically light gray
        lower_concrete = np.array([0, 0, 100])
        upper_concrete = np.array([180, 50, 255])
        mask_concrete = cv2.inRange(hsv, lower_concrete, upper_concrete)
        
        # Also include the purple/magenta areas (non-carbonated)
        lower_purple = np.array([130, 50, 50])
        upper_purple = np.array([160, 255, 255])
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
        
        combined_mask = cv2.bitwise_or(mask_concrete, mask_purple)
        
        # Morphological operations to clean the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (the concrete block)
            largest_contour = max(contours, key=cv2.contourArea)
            self.concrete_boundary = largest_contour
            
            # Fit rectangle to get approximate dimensions
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get side lengths
            side1 = np.linalg.norm(box[0] - box[1])
            side2 = np.linalg.norm(box[1] - box[2])
            
            # Calculate pixel-to-mm ratio (user provides one reference side as 50mm)
            detected_pixel_side = max(side1, side2)
            self.pixel_to_mm_ratio = self.reference_side_mm / detected_pixel_side
            
            print(f"Detected pixel side length: {detected_pixel_side:.2f} pixels")
            print(f"Reference side: {self.reference_side_mm} mm")
            print(f"Pixel-to-mm ratio: {self.pixel_to_mm_ratio:.6f} mm/pixel")
            
            # Visualize boundary detection
            boundary_viz = self.original_image_rgb.copy()
            cv2.drawContours(boundary_viz, [largest_contour], 0, (0, 255, 0), 3)
            
            self.save_step(boundary_viz, "01_concrete_boundary_detected")
            self._display_step(boundary_viz, "Concrete Block Boundary Detected")
            
            return True
        else:
            print("⚠ Warning: Could not detect concrete boundary")
            return False
    
    def step2_extract_green_channel(self):
        """
        Step 2: Extract green channel from original image
        Green is complementary to magenta (non-carbonated color)
        """
        print("\n" + "="*60)
        print("STEP 2: EXTRACT GREEN CHANNEL")
        print("="*60)
        
        # Convert BGR to RGB then extract green
        self.green_channel = self.original_image_rgb[:, :, 1]  # Green channel
        
        print(f"Green channel shape: {self.green_channel.shape}")
        print(f"Green channel range: [{self.green_channel.min()}, {self.green_channel.max()}]")
        
        # Visualize
        green_viz = np.zeros_like(self.original_image_rgb)
        green_viz[:, :, 1] = self.green_channel
        self.save_step(green_viz, "02_green_channel_extracted")
        self._display_step(green_viz, "Green Channel Extracted\n(High values = Non-carbonated, Low values = Carbonated)")
        
        return self.green_channel
    
    def step3_otsu_thresholding(self):
        """
        Step 3: Apply Otsu's thresholding to binary image
        Separates carbonated (white) from non-carbonated (black)
        """
        print("\n" + "="*60)
        print("STEP 3: OTSU'S THRESHOLDING")
        print("="*60)
        
        # Apply Otsu's method
        threshold_value, self.binary_green = cv2.threshold(
            self.green_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        print(f"Otsu threshold value: {threshold_value}")
        print(f"Binary image range: [{self.binary_green.min()}, {self.binary_green.max()}]")
        
        # Convert to 0-1 range for further processing
        self.binary_green = self.binary_green.astype(np.uint8) // 255
        
        # Visualize
        binary_viz = np.zeros_like(self.original_image_rgb)
        binary_viz[self.binary_green == 1] = [255, 255, 255]
        binary_viz[self.binary_green == 0] = [0, 0, 0]
        
        self.save_step(binary_viz, "03_binary_otsu_thresholded")
        self._display_step(binary_viz, "Binary Image (Otsu Thresholding)\nWhite = Carbonated, Black = Non-carbonated")
        
        return self.binary_green
    
    def step4_fill_holes_noncarbonated(self):
        """
        Step 4: Fill holes in non-carbonated region (morphological analysis)
        Invert binary image, fill holes, then invert back
        """
        print("\n" + "="*60)
        print("STEP 4: FILL HOLES IN NON-CARBONATED REGION")
        print("="*60)
        
        # Invert to work with non-carbonated regions (white)
        inverted = 1 - self.binary_green
        
        # Fill holes using scipy
        self.holes_filled = ndimage.binary_fill_holes(inverted)
        
        print(f"Holes filled in non-carbonated region")
        print(f"Filled region shape: {self.holes_filled.shape}")
        
        # Visualize
        filled_viz = np.zeros_like(self.original_image_rgb)
        filled_viz[self.holes_filled == 1] = [255, 255, 255]  # Non-carbonated in white
        filled_viz[self.holes_filled == 0] = [0, 0, 0]        # Carbonated in black
        
        self.save_step(filled_viz, "04_holes_filled_noncarbonated")
        self._display_step(filled_viz, "After Filling Holes\nWhite = Non-carbonated, Black = Carbonated")
        
        return self.holes_filled
    
    def step5_eliminate_external_objects(self):
        """
        Step 5: Remove small objects and objects outside non-carbonated region
        Keep only the largest connected component (main non-carbonated region)
        """
        print("\n" + "="*60)
        print("STEP 5: ELIMINATE EXTERNAL OBJECTS")
        print("="*60)
        
        # Label connected components
        labeled_array, num_features = ndimage.label(self.holes_filled)
        
        print(f"Number of connected components: {num_features}")
        
        # Get size of each component
        sizes = ndimage.sum(self.holes_filled, labeled_array, range(num_features + 1))
        
        # Keep only largest component
        largest_label = np.argmax(sizes)
        self.cleaned_binary = (labeled_array == largest_label)
        
        print(f"Largest component label: {largest_label}")
        print(f"Largest component size: {sizes[largest_label]} pixels")
        print(f"Total non-carbonated area: {np.sum(self.cleaned_binary)} pixels")
        
        # Visualize
        cleaned_viz = np.zeros_like(self.original_image_rgb)
        cleaned_viz[self.cleaned_binary == 1] = [255, 255, 255]  # Non-carbonated
        cleaned_viz[self.cleaned_binary == 0] = [0, 0, 0]        # Carbonated
        
        self.save_step(cleaned_viz, "05_external_objects_eliminated")
        self._display_step(cleaned_viz, "After Eliminating External Objects\nWhite = Non-carbonated, Black = Carbonated")
        
        return self.cleaned_binary
    
    def step6_convex_hull_secondary_detection(self):
        """
        Step 6: Apply convex hull for secondary detection
        Improves accuracy by removing false detections based on convexity of carbonation
        """
        print("\n" + "="*60)
        print("STEP 6: CONVEX HULL SECONDARY DETECTION")
        print("="*60)
        
        # Find contour of non-carbonated region
        contours, _ = cv2.findContours(self.cleaned_binary.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠ No contours found")
            self.convex_hull_result = self.cleaned_binary
            return self.cleaned_binary
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute convex hull
        hull = cv2.convexHull(largest_contour)
        
        # Create mask from convex hull
        convex_mask = np.zeros_like(self.cleaned_binary)
        cv2.drawContours(convex_mask, [hull], 0, 1, -1)
        
        # Final carbonated region is inside convex hull but not in cleaned region
        self.convex_hull_result = convex_mask
        
        print(f"Convex hull applied successfully")
        print(f"Convex hull area: {np.sum(convex_mask)} pixels")
        
        # Visualize convex hull
        hull_viz = np.zeros_like(self.original_image_rgb)
        hull_viz[convex_mask == 1] = [255, 255, 255]  # Non-carbonated in convex hull
        hull_viz[self.cleaned_binary == 1] = [0, 255, 0]  # Actually non-carbonated in green
        
        self.save_step(hull_viz, "06_convex_hull_applied")
        self._display_step(hull_viz, "Convex Hull Applied (Secondary Detection)\nGreen = Confirmed Non-carbonated")
        
        return self.convex_hull_result
    
    def step7_calculate_carbonation_metrics(self):
        """
        Step 7: Calculate carbonation depth, area, and percentage
        """
        print("\n" + "="*60)
        print("STEP 7: CALCULATE CARBONATION METRICS")
        print("="*60)
        
        # Identify carbonated region (inverse of convex hull)
        self.final_carbonated = 1 - self.convex_hull_result
        
        # Calculate areas in pixels
        total_area_pixels = np.sum(self.cleaned_binary)
        carbonated_area_pixels = np.sum(self.final_carbonated)
        non_carbonated_area_pixels = np.sum(self.convex_hull_result)
        
        # Convert to mm²
        area_conversion = self.pixel_to_mm_ratio ** 2
        total_area_mm2 = total_area_pixels * area_conversion
        carbonated_area_mm2 = carbonated_area_pixels * area_conversion
        non_carbonated_area_mm2 = non_carbonated_area_pixels * area_conversion
        
        # Calculate percentages
        if total_area_pixels > 0:
            carbonation_percentage = (carbonated_area_pixels / total_area_pixels) * 100
            non_carbonation_percentage = (non_carbonated_area_pixels / total_area_pixels) * 100
        else:
            carbonation_percentage = 0
            non_carbonation_percentage = 0
        
        print(f"\nCarbonation Analysis Results:")
        print(f"{'='*50}")
        print(f"Total concrete area: {total_area_pixels:.0f} pixels = {total_area_mm2:.2f} mm²")
        print(f"Carbonated area: {carbonated_area_pixels:.0f} pixels = {carbonated_area_mm2:.2f} mm²")
        print(f"Non-carbonated area: {non_carbonated_area_pixels:.0f} pixels = {non_carbonated_area_mm2:.2f} mm²")
        print(f"\nCarbonation percentage: {carbonation_percentage:.2f}%")
        print(f"Non-carbonation percentage: {non_carbonation_percentage:.2f}%")
        print(f"{'='*50}")
        
        return {
            'total_area_pixels': total_area_pixels,
            'carbonated_area_pixels': carbonated_area_pixels,
            'non_carbonated_area_pixels': non_carbonated_area_pixels,
            'total_area_mm2': total_area_mm2,
            'carbonated_area_mm2': carbonated_area_mm2,
            'non_carbonated_area_mm2': non_carbonated_area_mm2,
            'carbonation_percentage': carbonation_percentage,
            'non_carbonation_percentage': non_carbonation_percentage
        }
    
    def step8_visualize_side_lengths(self):
        """
        Step 8: Detect quadrilateral sides and measure lengths
        """
        print("\n" + "="*60)
        print("STEP 8: MEASURE QUADRILATERAL SIDES")
        print("="*60)
        
        # Find concrete block contour
        contours, _ = cv2.findContours(
            self.cleaned_binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            print("⚠ No concrete boundary found")
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculate side lengths in mm
        sides = []
        for i in range(4):
            p1 = box[i]
            p2 = box[(i + 1) % 4]
            length_pixels = np.linalg.norm(p2 - p1)
            length_mm = length_pixels * self.pixel_to_mm_ratio
            sides.append(length_mm)
        
        print(f"\nQuadrilateral Side Lengths (mm):")
        print(f"{'='*50}")
        for i, side in enumerate(sides, 1):
            print(f"Side {i}: {side:.2f} mm")
        
        # Calculate perimeter and area
        perimeter_mm = sum(sides)
        
        # For rectangular approximation
        area_mm2 = sides[0] * sides[1] if len(sides) >= 2 else 0
        
        print(f"\nPerimeter: {perimeter_mm:.2f} mm")
        print(f"Approx Area (rectangular): {area_mm2:.2f} mm²")
        print(f"{'='*50}")
        
        # Visualize
        side_viz = self.original_image_rgb.copy()
        cv2.drawContours(side_viz, [largest_contour], 0, (0, 255, 0), 3)
        
        # Draw lines for each side and label
        for i in range(4):
            p1 = box[i]
            p2 = box[(i + 1) % 4]
            mid = ((p1 + p2) // 2).astype(int)
            cv2.line(side_viz, tuple(p1), tuple(p2), (255, 0, 0), 2)
            cv2.putText(side_viz, f"{sides[i]:.1f}mm", tuple(mid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        self.save_step(side_viz, "08_measured_sides")
        self._display_step(side_viz, "Measured Sides (mm)")
        
        return {
            'sides': sides,
            'perimeter': perimeter_mm,
            'area_rectangular': area_mm2
        }
    
    def step9_create_final_composite(self):
        """
        Step 9: Create final composite image showing all regions
        """
        print("\n" + "="*60)
        print("STEP 9: CREATE FINAL COMPOSITE VISUALIZATION")
        print("="*60)
        
        composite = self.original_image_rgb.copy()
        
        # Overlay results with transparency
        carbonated_mask = self.final_carbonated.astype(np.uint8)
        non_carbonated_mask = self.convex_hull_result.astype(np.uint8)
        
        # Red for carbonated, Green for non-carbonated
        composite[carbonated_mask == 1] = [255, 100, 100]  # Light red
        composite[non_carbonated_mask == 1] = [100, 255, 100]  # Light green
        
        # Draw boundary
        if self.concrete_boundary is not None:
            cv2.drawContours(composite, [self.concrete_boundary], 0, (0, 0, 255), 3)
        
        self.save_step(composite, "09_final_composite_result")
        self._display_step(composite, "Final Result\nRed = Carbonated, Green = Non-carbonated")
        
        return composite
    
    def _display_step(self, image, title):
        """Display intermediate step in console-friendly way"""
        print(f"\n→ {title}")
        print(f"   Image shape: {image.shape}")
    
    def run_complete_analysis(self):
        """Execute complete analysis pipeline"""
        print("\n" + "#"*60)
        print("CONCRETE CARBONATION DETECTION ANALYSIS")
        print("Based on: Choi et al., 2017")
        print("#"*60)
        
        # Execute all steps
        self.step1_detect_concrete_boundary()
        if self.pixel_to_mm_ratio is None:
            print("❌ Failed to establish pixel-to-mm ratio. Exiting.")
            return None
        
        self.step2_extract_green_channel()
        self.step3_otsu_thresholding()
        self.step4_fill_holes_noncarbonated()
        self.step5_eliminate_external_objects()
        self.step6_convex_hull_secondary_detection()
        
        metrics = self.step7_calculate_carbonation_metrics()
        sides = self.step8_visualize_side_lengths()
        composite = self.step9_create_final_composite()
        
        print("\n" + "#"*60)
        print("ANALYSIS COMPLETE")
        print(f"All outputs saved to: {self.output_dir.absolute()}")
        print("#"*60)
        
        return {
            'metrics': metrics,
            'sides': sides,
            'composite': composite,
            'pixel_to_mm_ratio': self.pixel_to_mm_ratio
        }


def main():
    """Main execution function"""
    # Example usage
    image_files = ['test_images\6_3AC_12CC.jpg', 'test_images\9-3AC-12CC.jpg', 'test_images\6_3AC_12CC.jpg', 'test_images\9-12CC.jpg']
    
    for image_file in image_files:
        if os.path.exists(image_file):
            print(f"\n\n{'*'*80}")
            print(f"Processing: {image_file}")
            print(f"{'*'*80}")
            
            analyzer = ConcreteCarbonationAnalyzer(
                image_path=image_file,
                reference_side_mm=50
            )
            results = analyzer.run_complete_analysis()
            break  # Process first available image
    else:
        print("❌ No image files found. Please provide concrete images.")


if __name__ == "__main__":
    main()
