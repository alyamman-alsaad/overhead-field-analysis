import cv2
import numpy as np
import json
import argparse
from collections import defaultdict


class FieldNodeIntegrator:
    def __init__(self, node_diameter=20):
        # BGR ranges for vegetation health classification
        self.bgr_ranges = {
            'very_healthy': (np.array([0, 0, 0]), np.array([41, 113, 95])),
            'mostly_healthy': (np.array([0, 105, 85]), np.array([46, 255, 151])),
            'mostly_unhealthy': (np.array([48, 0, 81]), np.array([91, 143, 136])),
            'very_unhealthy': (np.array([59, 0, 120]), np.array([180, 255, 255]))
        }

        # Color map for field section overlay
        self.color_map = {
            'very_healthy': (0, 255, 0),  # Green
            'mostly_healthy': (0, 255, 190),  # Light Green
            'mostly_unhealthy': (0, 145, 255),  # Orange
            'very_unhealthy': (0, 0, 255),  # Red
            'unknown': (128, 128, 128)  # Gray
        }

        # Node configuration
        self.node_diameter = node_diameter

        # Map rectangle classifications to indices for node color lookup
        self.rect_class_to_idx = {
            'very_healthy': 0,
            'mostly_healthy': 1,
            'mostly_unhealthy': 2,
            'very_unhealthy': 3
        }

        # BGR colors for nodes (from original code)
        self.node_bgr_colors = [
            (1, 1, 1),  # BGR[0] - Very dark (almost black)
            (50, 51, 86),  # BGR[1] - Dark brown/purple
            (82, 82, 82),  # BGR[2] - Medium gray
            (121, 121, 121)  # BGR[3] - Light gray
        ]

        # Node color mapping: [node_class][rect_class_idx] = bgr_color_idx
        # Rows = node class (0=bacterial, 1=healthy)
        # Cols = rect class (0=very_healthy, 1=mostly_healthy, 2=mostly_unhealthy, 3=very_unhealthy)
        self.node_color_map = [
            [1, 2, 3, 3],  # Bacterial node colors
            [0, 0, 1, 2],  # Healthy node colors
        ]

    def load_json(self, filepath):
        """Load JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)

    def find_cell_for_point(self, x, y, cells_data):
        """
        Find which field cell contains the given point (x, y).
        Returns cell data or None if not found.
        """
        point = np.array([x, y])

        for cell in cells_data:
            corners = np.array(cell['corners'], dtype=np.int32)
            # Use pointPolygonTest to check if point is inside cell
            result = cv2.pointPolygonTest(corners, (float(x), float(y)), False)
            if result >= 0:  # Point is inside or on the boundary
                return cell

        return None

    def get_node_color(self, node_health_status, field_classification):
        """
        Get the BGR color for a node based on its health status and field classification.

        Args:
            node_health_status: 'healthy' or 'bacterial' (lowercase)
            field_classification: 'very_healthy', 'mostly_healthy', etc.

        Returns:
            BGR tuple for the node color
        """
        # Convert node health status to class index
        node_class = 0 if node_health_status == 'bacterial' else 1

        # Get field classification index
        if field_classification not in self.rect_class_to_idx:
            # Default to mostly_healthy if unknown
            rect_idx = 1
        else:
            rect_idx = self.rect_class_to_idx[field_classification]

        # Get color index from mapping
        color_idx = self.node_color_map[node_class][rect_idx]

        # Return the BGR color
        return self.node_bgr_colors[color_idx]

    def draw_nodes_on_image(self, img, node_coords, node_health, field_health_data):
        """
        Draw nodes on the image with colors based on node health and field section.

        Args:
            img: Input image
            node_coords: Dictionary with node IDs and their x, y coordinates
            node_health: Dictionary with node IDs and their health status
            field_health_data: Dictionary with field section health classifications

        Returns:
            Image with nodes drawn, and node placement information
        """
        img_with_nodes = img.copy()
        node_info = []

        cells_data = field_health_data.get('cells', [])

        for node_id, coords in node_coords.items():
            x = int(coords['x'])
            y = int(coords['y'])

            # Get node health status
            health_data = node_health.get(node_id, {})
            # Convert health_class to lowercase for internal processing
            health_status = health_data.get('health_class', 'Healthy').lower()

            # Find which field cell this node is in
            cell = self.find_cell_for_point(x, y, cells_data)

            if cell:
                field_classification = cell['classification']
                cell_id = cell['cell_id']
            else:
                print(f"Warning: Node {node_id} at ({x}, {y}) is not within any field cell")
                field_classification = 'mostly_healthy'  # Default
                cell_id = None

            # Get node color based on health and field classification
            node_color = self.get_node_color(health_status, field_classification)

            # Draw the node
            radius = self.node_diameter // 2
            cv2.circle(img_with_nodes, (x, y), radius, node_color, -1)

            # Store node information
            node_info.append({
                'node_id': node_id,
                'position': (x, y),
                'health_status': health_status,
                'field_classification': field_classification,
                'cell_id': cell_id,
                'color': node_color,
                'battery': health_data.get('battery', None),
                'soil_moisture': health_data.get('soil_moisture', None),
                'soil_moisture_raw': health_data.get('soil-moisture-raw', None),
                'timestamp': health_data.get('timestamp', None)
            })

            print(f"{node_id}: {health_status} node in {field_classification} section, "
                  f"color={node_color}, position=({x}, {y})")

        return img_with_nodes, node_info

    def classify_cell(self, img, cell_corners):
        """
        Apply color ranges to a single cell and determine classification.
        Returns classification and ratio data.
        """
        # Create a mask for this specific cell
        cell_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(cell_mask, [cell_corners], 255)

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

    def reclassify_field(self, img_with_nodes, field_health_data):
        """
        Re-classify field sections after nodes have been added to the image.

        Args:
            img_with_nodes: Image with nodes drawn
            field_health_data: Original field health data

        Returns:
            Updated classifications
        """
        updated_classifications = []

        for cell in field_health_data.get('cells', []):
            cell_corners = np.array(cell['corners'], dtype=np.int32)

            # Re-classify this cell with nodes present
            classification, ratios = self.classify_cell(img_with_nodes, cell_corners)

            # Get cell center for reference
            center = cell['center']

            updated_classifications.append({
                'cell_id': cell['cell_id'],
                'classification': classification,
                'original_classification': cell['classification'],
                'ratios': {k: float(v) for k, v in ratios.items()},
                'center': center,
                'bounds': cell['bounds'],
                'corners': cell['corners']
            })

        return updated_classifications

    def draw_field_overlay(self, img, classifications, alpha=0.3):
        """
        Draw the classification results on the image with semi-transparent fills.
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

    def save_updated_data(self, classifications, node_info, output_path):
        """
        Save updated classification data with node information to JSON.
        """
        # Create summary statistics
        summary = defaultdict(int)
        for result in classifications:
            summary[result['classification']] += 1

        # Prepare output data
        output_data = {
            'total_cells': len(classifications),
            'summary': dict(summary),
            'nodes': node_info,
            'cells': classifications
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved updated classification data to {output_path}")

    def analyze(self, image_path, node_coords_path, node_health_path, field_health_path,
                output_image_path=None, output_json_path=None, alpha=0.3):
        """
        Main pipeline: Integrate field nodes with field health analysis.

        Args:
            image_path: Path to input image
            node_coords_path: Path to JSON file with node coordinates
            node_health_path: Path to JSON file with node health status
            field_health_path: Path to JSON file with raw field health data
            output_image_path: Path to save output image (optional)
            output_json_path: Path to save updated JSON data (optional)
            alpha: Transparency level for overlay (0-1), default 0.3

        Returns:
            Dictionary with results
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        print(f"Image loaded: {img.shape}")

        # Load JSON data
        print("\nLoading JSON files...")
        node_coords = self.load_json(node_coords_path)
        node_health = self.load_json(node_health_path)
        field_health_data = self.load_json(field_health_path)

        print(f"Loaded {len(node_coords)} node coordinates")
        print(f"Loaded {len(node_health)} node health records")
        print(f"Loaded field health data with {field_health_data.get('total_cells', 0)} cells")

        # Draw nodes on image
        print("\n=== Drawing Field Nodes ===")
        img_with_nodes, node_info = self.draw_nodes_on_image(
            img, node_coords, node_health, field_health_data
        )

        # Re-classify field sections with nodes present
        print("\n=== Re-classifying Field Sections ===")
        updated_classifications = self.reclassify_field(img_with_nodes, field_health_data)

        # Print summary
        print("\n=== Updated Classification Summary ===")
        summary = defaultdict(int)
        for result in updated_classifications:
            summary[result['classification']] += 1
        for classification, count in sorted(summary.items()):
            original_count = field_health_data['summary'].get(classification, 0)
            print(f"  {classification}: {count} cells (was {original_count})")

        # Draw field overlay
        print("\nDrawing field health overlay...")
        output_img = self.draw_field_overlay(img_with_nodes, updated_classifications, alpha=alpha)

        # Save output image if path provided
        if output_image_path:
            cv2.imwrite(output_image_path, output_img)
            print(f"Saved result image to {output_image_path}")

        # Save JSON data if path provided
        if output_json_path:
            self.save_updated_data(updated_classifications, node_info, output_json_path)

        return {
            'original': img,
            'with_nodes': img_with_nodes,
            'output': output_img,
            'node_info': node_info,
            'classifications': updated_classifications,
            'summary': dict(summary)
        }


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Integrate field nodes with field health analysis.'
    )

    parser.add_argument(
        'input_image',
        type=str,
        help='Path to input image'
    )

    parser.add_argument(
        'node_coords',
        type=str,
        help='Path to JSON file with node coordinates'
    )

    parser.add_argument(
        'node_health',
        type=str,
        help='Path to JSON file with node health status'
    )

    parser.add_argument(
        'field_health',
        type=str,
        help='Path to JSON file with raw field health data'
    )

    parser.add_argument(
        '-o', '--output-image',
        type=str,
        default=None,
        help='Path to save output image with nodes and health overlay (optional)'
    )

    parser.add_argument(
        '-j', '--output-json',
        type=str,
        default=None,
        help='Path to save updated classification data as JSON (optional)'
    )

    parser.add_argument(
        '-a', '--alpha',
        type=float,
        default=0.3,
        help='Transparency level for overlay (0-1), default=0.3'
    )

    parser.add_argument(
        '-d', '--node-diameter',
        type=int,
        default=20,
        help='Diameter of nodes in pixels, default=20'
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Skip displaying the results (useful for batch processing)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Create integrator
    integrator = FieldNodeIntegrator(node_diameter=args.node_diameter)

    # Analyze and integrate nodes
    results = integrator.analyze(
        args.input_image,
        args.node_coords,
        args.node_health,
        args.field_health,
        output_image_path=args.output_image,
        output_json_path=args.output_json,
        alpha=args.alpha
    )

    # Display results unless --no-display flag is set
    if not args.no_display:
        cv2.imshow("Original Image", results['original'])
        cv2.imshow("With Field Nodes", results['with_nodes'])
        cv2.imshow("Final Analysis", results['output'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n=== Node Summary ===")
    for node in results['node_info']:
        print(f"{node['node_id']}: {node['health_status']} in {node['field_classification']} section")

    print("\nAnalysis complete!")

'''
 Bash Script (3):
 python aug-field-analysis_CG.py assets/overhead_img.jpg assets/field_node_coords.json assets/field_node_health.json assets/raw_health_data.json -o assets/aug_field_analysis.jpg -j assets/updated_health_data.json
'''