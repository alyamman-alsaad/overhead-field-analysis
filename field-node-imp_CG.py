#!/usr/bin/env python3
"""
Field Node Visualizer - Simplified Version

Reads field node positions and health classifications from JSON files
and visualizes them as simple colored circles on the field image.

Usage:
    python field_node_visualizer.py --image field.jpg --coords field_nodes.json \
        --health node_health.json --output annotated_field.jpg

Requirements:
    pip install opencv-python numpy --break-system-packages
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path


class FieldNodeVisualizer:
    def __init__(self, node_diameter=20):
        """
        Initialize the field node visualizer.

        Args:
            node_diameter: Diameter of nodes to draw in pixels
        """
        self.node_diameter = node_diameter

        # Color map for health classifications (BGR format)
        # YOU CAN EASILY ADJUST THESE COLORS
        self.health_colors = {
            'healthy': (0, 255, 0),  # Green
            'bacterial': (0, 0, 255),  # Red
            'fungal': (0, 165, 255),  # Orange
            'viral': (255, 0, 255),  # Magenta
            'nutrient_deficiency': (0, 255, 255),  # Yellow
            'unknown': (128, 128, 128)  # Gray
        }

    def load_field_nodes(self, coords_path):
        """
        Load field node coordinates from JSON file.

        Args:
            coords_path: Path to field_nodes.json

        Returns:
            Dictionary of node_id -> {x, y, detected_frequency, expected_frequency}
        """
        with open(coords_path, 'r') as f:
            field_nodes = json.load(f)
        return field_nodes

    def load_health_data(self, health_path):
        """
        Load node health classifications from JSON file.

        Args:
            health_path: Path to node_health.json

        Returns:
            Dictionary of node_id -> {health_status, confidence, notes}
        """
        with open(health_path, 'r') as f:
            health_data = json.load(f)
        return health_data

    def draw_node(self, image, position, health_status):
        """
        Draw a simple colored circle for the node.

        Args:
            image: Image to draw on
            position: (x, y) tuple for node center
            health_status: Health classification string
        """
        x, y = position
        radius = self.node_diameter // 2

        # Get color for this health status
        color = self.health_colors.get(health_status, self.health_colors['unknown'])

        # Draw simple filled circle
        cv2.circle(image, (x, y), radius, color, -1)

    def visualize(self, image_path, coords_path, health_path, output_path=None):
        """
        Main visualization pipeline - draws only colored circles for nodes.

        Args:
            image_path: Path to field image
            coords_path: Path to field_nodes.json
            health_path: Path to node_health.json
            output_path: Path to save annotated image (optional)

        Returns:
            Dictionary with visualization results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        print(f"Loaded image: {image.shape}")

        # Load field node coordinates
        print(f"\nLoading field node coordinates from {coords_path}...")
        field_nodes = self.load_field_nodes(coords_path)
        print(f"Loaded {len(field_nodes)} field nodes")

        # Load health data
        print(f"\nLoading health data from {health_path}...")
        health_data = self.load_health_data(health_path)
        print(f"Loaded health data for {len(health_data)} nodes")

        # Create output image
        output_image = image.copy()

        # Draw each node
        print("\n=== Drawing Nodes ===")
        for node_id in field_nodes.keys():
            # Get coordinates
            x = field_nodes[node_id]['x']
            y = field_nodes[node_id]['y']

            # Get health status (default to unknown if not in health data)
            if node_id in health_data:
                health_status = health_data[node_id]['health_status']
                confidence = health_data[node_id].get('confidence', 'N/A')
                print(f"{node_id} at ({x}, {y}): {health_status} "
                      f"(confidence: {confidence})")
            else:
                health_status = 'unknown'
                print(f"{node_id} at ({x}, {y}): No health data available")

            # Draw the node (simple colored circle)
            self.draw_node(output_image, (x, y), health_status)

        # Save output if path provided
        if output_path:
            cv2.imwrite(output_path, output_image)
            print(f"\n✓ Saved annotated image to: {output_path}")

        return {
            'original': image,
            'annotated': output_image,
            'field_nodes': field_nodes,
            'health_data': health_data
        }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize field nodes with simple colored circles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python field_node_visualizer.py --image field.jpg \
      --coords field_nodes.json --health node_health.json \
      --output annotated_field.jpg
        """
    )

    parser.add_argument('--image', required=True, help='Path to field image')
    parser.add_argument('--coords', required=True,
                        help='Path to field node coordinates JSON (from LED analyzer)')
    parser.add_argument('--health', required=True,
                        help='Path to node health classification JSON')
    parser.add_argument('--output', default=None,
                        help='Path to save annotated image (optional)')
    parser.add_argument('--node-diameter', type=int, default=20,
                        help='Diameter of nodes in pixels (default: 20)')

    args = parser.parse_args()

    # Validate input files
    for path, name in [(args.image, 'Image'), (args.coords, 'Coordinates'),
                       (args.health, 'Health data')]:
        if not Path(path).exists():
            print(f"Error: {name} file not found: {path}")
            return 1

    # Create visualizer
    visualizer = FieldNodeVisualizer(node_diameter=args.node_diameter)

    # Run visualization
    try:
        results = visualizer.visualize(
            args.image,
            args.coords,
            args.health,
            args.output
        )

        print("\n=== Visualization Complete ===")
        print(f"Total nodes visualized: {len(results['field_nodes'])}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())

'''
 Bash Script:
 python field-node-imp_CG.py --image assets/overhead_img.jpg --coords assets/field_node_coords.json --health assets/field_node_health.json --output assets/output_nodes.jpg
'''