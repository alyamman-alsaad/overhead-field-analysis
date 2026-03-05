import cv2
import numpy as np
import json
import argparse
from collections import defaultdict


class FieldHealthAnalyzer:
    def __init__(self):
        # BGR ranges for vegetation health classification
        self.bgr_ranges = {
            'very_healthy': (np.array([0, 0, 0]), np.array([41, 113, 95])),
            'mostly_healthy': (np.array([0, 105, 85]), np.array([46, 255, 151])),
            'mostly_unhealthy': (np.array([48, 0, 81]), np.array([91, 143, 136])),
            'very_unhealthy': (np.array([59, 0, 120]), np.array([180, 255, 255]))
        }

        # Color map for output visualization
        self.color_map = {
            'very_healthy': (0, 255, 0),  # Green
            'mostly_healthy': (0, 255, 190),  # Light Green
            'mostly_unhealthy': (0, 145, 255),  # Orange
            'very_unhealthy': (0, 0, 255),  # Red
            'unknown': (128, 128, 128)  # Gray
        }

    def find_corners(self, img):
        """Detect corners using goodFeaturesToTrack"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.2, 50)
        corners = corners.astype(int).reshape(-1, 2)
        return corners

    def organize_corners_to_grid(self, corners):
        """
        Organize detected corners into a structured grid.
        Returns a sorted list of corners by row and column.
        """
        # Sort by Y first (rows), then by X (columns)
        corners_sorted = corners[np.lexsort((corners[:, 0], corners[:, 1]))]

        # Group corners into rows (using Y-coordinate clustering)
        rows = []
        current_row = [corners_sorted[0]]
        y_threshold = 30  # Adjust based on your grid spacing

        for corner in corners_sorted[1:]:
            if abs(corner[1] - current_row[0][1]) < y_threshold:
                current_row.append(corner)
            else:
                # Sort current row by X coordinate
                current_row.sort(key=lambda c: c[0])
                rows.append(current_row)
                current_row = [corner]

        # Don't forget the last row
        current_row.sort(key=lambda c: c[0])
        rows.append(current_row)

        return rows

    def create_grid_cells(self, rows):
        """
        Create rectangular cells from organized corner points.
        Returns a list of cell polygons (4 corners each).
        """
        cells = []

        for i in range(len(rows) - 1):
            row_current = rows[i]
            row_next = rows[i + 1]

            # Match corners between adjacent rows
            for j in range(min(len(row_current), len(row_next)) - 1):
                # Try to create a cell from 4 corners
                if j < len(row_current) - 1 and j < len(row_next) - 1:
                    tl = row_current[j]  # Top-left
                    tr = row_current[j + 1]  # Top-right
                    bl = row_next[j]  # Bottom-left
                    br = row_next[j + 1]  # Bottom-right

                    # Create cell as numpy array of 4 corners
                    cell = np.array([tl, tr, br, bl], dtype=np.int32)
                    cells.append(cell)

        return cells

    def classify_cell(self, img, cell):
        """
        Apply color ranges to a single cell and determine classification.
        Returns classification and ratio data.
        """
        # Create a mask for this specific cell
        cell_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(cell_mask, [cell], 255)

        # Calculate total pixels in this cell
        total_pixels = cv2.countNonZero(cell_mask)

        # Apply each BGR range and calculate ratio
        ratios = {}
        for name, (lower, upper) in self.bgr_ranges.items():
            # Apply color range to BGR image
            color_mask = cv2.inRange(img, lower, upper)
            # Combine with cell mask to only count pixels in this cell
            combined_mask = cv2.bitwise_and(color_mask, cell_mask)
            matched_pixels = cv2.countNonZero(combined_mask)

            # Calculate ratio
            ratio = matched_pixels / total_pixels if total_pixels > 0 else 0
            ratios[name] = ratio

        # Find the classification with the highest ratio
        if sum(ratios.values()) > 0:
            classification = max(ratios, key=ratios.get)
        else:
            classification = 'unknown'

        return classification, ratios

    def segment_and_classify(self, img, cells):
        """
        For each cell, apply color ranges and determine classification.
        Returns classifications for each cell.
        """
        classifications = []

        for idx, cell in enumerate(cells):
            classification, ratios = self.classify_cell(img, cell)

            # Get cell center for reference
            center = np.mean(cell, axis=0).astype(int).tolist()

            # Get cell bounds
            min_x = int(np.min(cell[:, 0]))
            max_x = int(np.max(cell[:, 0]))
            min_y = int(np.min(cell[:, 1]))
            max_y = int(np.max(cell[:, 1]))

            classifications.append({
                'cell_id': idx,
                'classification': classification,
                'ratios': {k: float(v) for k, v in ratios.items()},  # Convert to float for JSON
                'center': center,
                'bounds': {
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y
                },
                'corners': cell.tolist()  # Convert numpy array to list for JSON
            })

        return classifications

    def draw_results(self, img, classifications, alpha=0.3):
        """
        Draw the classification results on the image with semi-transparent fills.

        Args:
            img: Input image
            classifications: List of classification results
            alpha: Transparency level (0-1), default 0.3 = 30% opacity

        Returns:
            Annotated image
        """
        output = img.copy()
        overlay = img.copy()

        for result in classifications:
            cell = np.array(result['corners'], dtype=np.int32)
            classification = result['classification']

            # Get the color for this classification
            color = self.color_map.get(classification, (128, 128, 128))

            # Fill the cell with the color on the overlay
            cv2.fillPoly(overlay, [cell], color)

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        return output

    def save_classifications_to_json(self, classifications, output_path):
        """
        Save classification results to a JSON file.

        Args:
            classifications: List of classification results
            output_path: Path to save JSON file
        """
        # Create summary statistics
        summary = defaultdict(int)
        for result in classifications:
            summary[result['classification']] += 1

        # Prepare output data
        output_data = {
            'total_cells': len(classifications),
            'summary': dict(summary),
            'cells': classifications
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved classification data to {output_path}")

    def analyze(self, image_path, output_image_path=None, output_json_path=None, alpha=0.3):
        """
        Main pipeline: load image, detect corners, segment, classify, and visualize.

        Args:
            image_path: Path to input image
            output_image_path: Path to save output image (optional)
            output_json_path: Path to save JSON classification data (optional)
            alpha: Transparency level for overlay (0-1), default 0.3

        Returns:
            Dictionary with results including original image, output image, and classifications
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        print(f"Image loaded: {img.shape}")

        # Step 1: Find corners
        print("\nDetecting corners...")
        corners = self.find_corners(img)
        print(f"Found {len(corners)} corners")

        # Step 2: Organize into grid
        print("Organizing corners into grid...")
        rows = self.organize_corners_to_grid(corners)
        print(f"Organized into {len(rows)} rows")

        # Step 3: Create cells
        print("Creating grid cells...")
        cells = self.create_grid_cells(rows)
        print(f"Created {len(cells)} cells")

        # Step 4: Classify each cell
        print("\nClassifying cells...")
        classifications = self.segment_and_classify(img, cells)

        # Print summary
        print("\n=== Classification Summary ===")
        summary = defaultdict(int)
        for result in classifications:
            summary[result['classification']] += 1
        for classification, count in sorted(summary.items()):
            print(f"  {classification}: {count} cells")

        # Step 5: Draw results
        print("\nDrawing results...")
        output_img = self.draw_results(img, classifications, alpha=alpha)

        # Step 6: Save output image if path provided
        if output_image_path:
            cv2.imwrite(output_image_path, output_img)
            print(f"Saved result image to {output_image_path}")

        # Step 7: Save JSON data if path provided
        if output_json_path:
            self.save_classifications_to_json(classifications, output_json_path)

        return {
            'original': img,
            'output': output_img,
            'classifications': classifications,
            'summary': dict(summary)
        }


# Example usage
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Analyze field health from overhead imagery using corner detection and color classification.'
    )

    parser.add_argument(
        'input_image',
        type=str,
        help='Path to input image'
    )

    parser.add_argument(
        '-o', '--output-image',
        type=str,
        default=None,
        help='Path to save output image with health overlay (optional)'
    )

    parser.add_argument(
        '-j', '--output-json',
        type=str,
        default=None,
        help='Path to save classification data as JSON (optional)'
    )

    parser.add_argument(
        '-a', '--alpha',
        type=float,
        default=0.3,
        help='Transparency level for overlay (0-1), default=0.3'
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Skip displaying the results (useful for batch processing)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Create analyzer
    analyzer = FieldHealthAnalyzer()

    # Analyze the field image
    results = analyzer.analyze(
        args.input_image,
        output_image_path=args.output_image,
        output_json_path=args.output_json,
        alpha=args.alpha
    )

    # Display results unless --no-display flag is set
    if not args.no_display:
        cv2.imshow("Original Image", results['original'])
        cv2.imshow("Health Analysis", results['output'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nAnalysis complete!")
    print(f"Total cells analyzed: {len(results['classifications'])}")

'''
 Bash Script:
 python raw-field-analysis_CG.py assets/overhead_img.jpg -o assets/raw_field_analysis.jpg -j assets/raw_health_data.json
'''