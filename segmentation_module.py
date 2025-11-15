"""Segmentation Module using Ultralytics - WITH GPU ACCELERATION & MASK REFINEMENT

Performs precise object segmentation with automatic GPU detection and acceleration.
Uses Ultralytics SAM2 model for state-of-the-art segmentation.
Includes advanced mask cleaning to remove scale and shadows from concrete mask.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch

try:
    from ultralytics import SAM
except ImportError:
    print("WARNING: ultralytics not installed. Install with: pip install ultralytics")
    SAM = None

class SegmentationModule:
    """Performs segmentation using Ultralytics SAM2 with advanced mask refinement"""

    def __init__(self, model_path: str = "sam2_b.pt"):
        """Initialize SAM 2 with automatic GPU detection

        Args:
            model_path: Path to SAM 2 model weights
        """
        self.model_path = model_path
        self.model = None
        self.segmentation_results = {}
        self.device = self._detect_device()
        print(f"🖥️ Device detected: {self.device}")

    def _detect_device(self) -> str:
        """Detect and return available computation device

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("✓ Apple Silicon GPU (MPS) available")
        else:
            device = 'cpu'
            print("⚠ No GPU available, using CPU (slower)")
        return device

    def get_device_info(self) -> Dict:
        """Get device information for reporting

        Returns:
            Dictionary with device info
        """
        info = {'device': self.device}
        if self.device == 'cuda':
            info['device_name'] = torch.cuda.get_device_name(0)
            info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        return info

    def initialize_sam_model(self) -> None:
        """Load and initialize SAM 2 model with GPU acceleration"""
        if SAM is None:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        try:
            print(f"\n[SAM] Loading model on {self.device}...")
            self.model = SAM(self.model_path)

            if self.device == 'cuda':
                self.model.to('cuda')
                print(f"✓ SAM 2 model loaded on GPU (CUDA)")
            elif self.device == 'mps':
                self.model.to('mps')
                print(f"✓ SAM 2 model loaded on Apple GPU (MPS)")
            else:
                self.model.to('cpu')
                print(f"✓ SAM 2 model loaded on CPU")
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print(" Downloading model automatically...")
            self.model = SAM("sam2_b.pt")

            if self.device == 'cuda':
                self.model.to('cuda')
            elif self.device == 'mps':
                self.model.to('mps')

    def segment_objects(self, image: np.ndarray,
                       points: Optional[List[List[int]]] = None,
                       labels: Optional[List[int]] = None) -> Dict:
        """Segment objects using SAM 2 with GPU acceleration

        Args:
            image: Input preprocessed image
            points: Optional prompt points [[x, y], ...]
            labels: Optional point labels (1=foreground, 0=background)

        Returns:
            Segmentation results
        """
        if self.model is None:
            self.initialize_sam_model()

        print(f"\n[SAM] Running segmentation on {self.device}...")

        if points is None:
            try:
                results = self.model.predict(
                    source=image,
                    device=self.device,
                    save=False,
                    verbose=False
                )
                print(f"✓ Automatic segmentation complete: {len(results)} result(s)")
            except Exception as e:
                print(f"⚠ Error during segmentation: {e}")
                results = []
        else:
            results = self.model(
                image,
                points=points,
                labels=labels,
                device=self.device,
                save=False,
                verbose=False
            )
            print(f"✓ Guided segmentation complete with {len(points)} prompts")

        return results

    def extract_masks(self, results) -> Dict[str, np.ndarray]:
        """Extract individual masks for different objects

        Args:
            results: Segmentation results from SAM 2

        Returns:
            Dictionary of masks for different objects
        """
        masks = {}

        if hasattr(results, 'masks') and results.masks is not None:
            if hasattr(results.masks.data, 'cpu'):
                all_masks = results.masks.data.cpu().numpy()
            else:
                all_masks = results.masks.data.numpy()

            mask_areas = [np.sum(mask) for mask in all_masks]
            sorted_indices = np.argsort(mask_areas)[::-1]

            if len(sorted_indices) >= 1:
                masks['concrete_block'] = all_masks[sorted_indices[0]].astype(np.uint8) * 255
                print(f"✓ Concrete block: {mask_areas[sorted_indices[0]]:.0f} pixels")

            if len(sorted_indices) >= 2:
                masks['scale'] = all_masks[sorted_indices[1]].astype(np.uint8) * 255
                print(f"✓ Scale: {mask_areas[sorted_indices[1]]:.0f} pixels")

            for i in range(2, min(len(sorted_indices), 5)):
                masks[f'object_{i-1}'] = all_masks[sorted_indices[i]].astype(np.uint8) * 255

        return masks

    def _clean_concrete_mask(self, concrete_mask: np.ndarray, 
                            scale_mask: np.ndarray) -> np.ndarray:
        """Clean concrete mask by removing scale and shadows

        Args:
            concrete_mask: Original concrete block mask
            scale_mask: Scale/ruler mask to remove

        Returns:
            Cleaned concrete mask (only concrete block)
        """
        print("\n[Mask Refinement] Cleaning concrete block mask...")

        # Step 1: Remove scale from concrete mask
        print("  Step 1: Removing scale from concrete mask...")
        # Dilate scale mask slightly to ensure complete removal
        scale_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        scale_dilated = cv2.dilate(scale_mask, scale_kernel, iterations=2)

        # Subtract scale from concrete mask
        cleaned_mask = cv2.bitwise_and(concrete_mask, cv2.bitwise_not(scale_dilated))

        print(f"    Original concrete area: {np.sum(concrete_mask > 0):,} pixels")
        print(f"    After removing scale: {np.sum(cleaned_mask > 0):,} pixels")

        # Step 2: Morphological opening to remove small shadows
        print("  Step 2: Removing small shadows with morphological opening...")
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opened = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

        print(f"    After removing shadows: {np.sum(opened > 0):,} pixels")

        # Step 3: Morphological closing to fill small holes in concrete
        print("  Step 3: Filling small holes with morphological closing...")
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        print(f"    After filling holes: {np.sum(closed > 0):,} pixels")

        # Step 4: Keep only the largest connected component (main concrete block)
        print("  Step 4: Keeping only main concrete block...")
        num_labels, labels_im = cv2.connectedComponents(closed)

        if num_labels > 1:
            # Find the largest component (excluding background which is 0)
            component_areas = [(labels_im == i).sum() for i in range(1, num_labels)]
            largest_label = np.argmax(component_areas) + 1

            final_mask = np.zeros_like(closed)
            final_mask[labels_im == largest_label] = 255

            print(f"    Found {num_labels} components, keeping largest")
            print(f"    Final concrete area: {np.sum(final_mask > 0):,} pixels")

            return final_mask
        else:
            print(f"    No components found after cleaning")
            return closed

    def detect_scale_boundaries(self, scale_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect precise boundaries of the scale/ruler

        Args:
            scale_mask: Binary mask of the scale

        Returns:
            Tuple of (edge image, boundary info dict)
        """
        edges = cv2.Canny(scale_mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return edges, {}

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        boundary_info = {
            'x': x, 'y': y, 'width': w, 'height': h,
            'contour': largest_contour,
            'area': cv2.contourArea(largest_contour)
        }
        print(f"✓ Scale boundaries: {w}x{h} pixels")
        return edges, boundary_info

    def detect_concrete_boundaries(self, concrete_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect precise boundaries of concrete block

        Args:
            concrete_mask: Binary mask of concrete block (cleaned)

        Returns:
            Tuple of (edge image, boundary info dict)
        """
        edges = cv2.Canny(concrete_mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return edges, {}

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        boundary_info = {
            'x': x, 'y': y, 'width': w, 'height': h,
            'contour': largest_contour,
            'area': cv2.contourArea(largest_contour),
            'perimeter': cv2.arcLength(largest_contour, True)
        }
        print(f"✓ Concrete boundaries: {w}x{h} pixels")
        return edges, boundary_info

    def visualize_segmentation(self, image: np.ndarray, masks: Dict[str, np.ndarray],
                             output_path: Optional[str] = None) -> np.ndarray:
        """Create visualization of segmentation results

        Args:
            image: Original image
            masks: Dictionary of masks
            output_path: Optional save path

        Returns:
            Visualization image
        """
        vis_image = image.copy()

        colors = {
            'concrete_block': (0, 255, 0),
            'scale': (255, 0, 0)
        }

        for name, mask in masks.items():
            color = colors.get(name, (0, 0, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color, 2)

        if output_path:
            cv2.imwrite(output_path, vis_image)

        return vis_image

    def segment_and_extract(self, image: np.ndarray) -> Dict:
        """Complete segmentation pipeline with GPU acceleration and mask refinement

        Args:
            image: Preprocessed input image

        Returns:
            Dictionary containing all segmentation results
        """
        if self.model is None:
            self.initialize_sam_model()

        results = self.segment_objects(image)
        masks = self.extract_masks(results[0] if isinstance(results, list) else results)

        segmentation_data = {'masks': masks}

        # Refine concrete mask by removing scale
        if 'concrete_block' in masks and 'scale' in masks:
            print("\n[Quality Control] Refining concrete block mask...")
            masks['concrete_block'] = self._clean_concrete_mask(
                masks['concrete_block'],
                masks['scale']
            )

        if 'concrete_block' in masks:
            edges, boundary_info = self.detect_concrete_boundaries(masks['concrete_block'])
            segmentation_data['concrete_boundaries'] = boundary_info
            segmentation_data['concrete_edges'] = edges

        if 'scale' in masks:
            edges, boundary_info = self.detect_scale_boundaries(masks['scale'])
            segmentation_data['scale_boundaries'] = boundary_info
            segmentation_data['scale_edges'] = edges

        self.segmentation_results = segmentation_data
        return segmentation_data


if __name__ == "__main__":
    segmenter = SegmentationModule()
    print("Segmentation Module Ready")