#!/usr/bin/env python3
"""
LED Frequency Analyzer and Field Node Identifier (v3 with coordinate export)

Analyzes a video to detect pulsing LEDs, determines their frequencies,
and labels them with field node IDs. Saves coordinates to JSON for use in other scripts.

Usage:
    python analyze_led_video_v3.py --input video.mp4 --output labeled.png \
        --mapping 1.0:NODE_001 2.0:NODE_002 3.0:NODE_003 4.0:NODE_004 \
        --save-coords field_nodes.json

Requirements:
    pip install opencv-python numpy scipy --break-system-packages
"""

import cv2
import numpy as np
import argparse
import sys
import json
from pathlib import Path
from scipy.fft import fft, fftfreq


def parse_frequency_mapping(mapping_strings):
    """Parse frequency to field node ID mapping."""
    mapping = {}
    for map_str in mapping_strings:
        try:
            freq_str, node_id = map_str.split(':')
            frequency = float(freq_str)
            mapping[frequency] = node_id
        except ValueError:
            raise ValueError(f"Invalid mapping format: {map_str}. Expected 'frequency:NODE_ID'")
    return mapping


def detect_led_regions(video_path, num_leds=4, brightness_threshold=200, min_distance=50):
    """Detect bright pulsing regions that are likely LEDs."""
    cap = cv2.VideoCapture(video_path)

    frames_to_sample = 30
    accumulated = None
    count = 0

    while count < frames_to_sample:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if accumulated is None:
            accumulated = gray.astype(np.float32)
        else:
            accumulated = np.maximum(accumulated, gray.astype(np.float32))

        count += 1

    cap.release()

    _, binary = cv2.threshold(accumulated.astype(np.uint8), brightness_threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_data = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                mask = np.zeros(accumulated.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                avg_brightness = cv2.mean(accumulated, mask=mask)[0]
                contour_data.append((avg_brightness, area, cx, cy))

    contour_data.sort(reverse=True, key=lambda x: x[0])

    led_positions = []
    for brightness, area, cx, cy in contour_data:
        is_far_enough = True
        for existing_x, existing_y in led_positions:
            distance = np.sqrt((cx - existing_x) ** 2 + (cy - existing_y) ** 2)
            if distance < min_distance:
                is_far_enough = False
                break

        if is_far_enough:
            led_positions.append((cx, cy))
            print(f"    Found LED: pos({cx},{cy}), brightness {brightness:.1f}, area {area}")

        if len(led_positions) >= num_leds:
            break

    return led_positions


def extract_intensity_timeline(video_path, position, roi_size=15):
    """Extract intensity values over time for a specific position."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    x, y = position
    intensities = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        y1 = max(0, y - roi_size)
        y2 = min(frame.shape[0], y + roi_size)
        x1 = max(0, x - roi_size)
        x2 = min(frame.shape[1], x + roi_size)

        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray_roi)
        intensities.append(mean_intensity)

        frame_count += 1

    cap.release()
    timestamps = np.arange(frame_count) / fps

    return timestamps, np.array(intensities), fps


def analyze_frequency(timestamps, intensities, min_freq=0.3, max_freq=10.0):
    """Analyze the dominant frequency in the intensity signal."""
    signal_centered = intensities - np.mean(intensities)

    std = np.std(signal_centered)
    if std > 0:
        signal_normalized = signal_centered / std
    else:
        signal_normalized = signal_centered

    window = np.hanning(len(signal_normalized))
    signal_windowed = signal_normalized * window

    n = len(signal_windowed)
    dt = timestamps[1] - timestamps[0]
    fft_values = fft(signal_windowed)
    fft_freq = fftfreq(n, dt)

    freq_mask = (fft_freq >= min_freq) & (fft_freq <= max_freq)
    fft_magnitude = np.abs(fft_values[freq_mask])
    fft_freq_filtered = fft_freq[freq_mask]

    if len(fft_freq_filtered) == 0:
        print(f"    Warning: No frequencies in range {min_freq}-{max_freq} Hz")
        freq_mask = fft_freq > 0
        fft_magnitude = np.abs(fft_values[freq_mask])
        fft_freq_filtered = fft_freq[freq_mask]

    if len(fft_magnitude) > 0:
        peak_idx = np.argmax(fft_magnitude)
        dominant_frequency = fft_freq_filtered[peak_idx]

        top_indices = np.argsort(fft_magnitude)[-3:][::-1]
        print(f"    Top 3 frequency peaks:")
        for idx in top_indices:
            freq = fft_freq_filtered[idx]
            power = fft_magnitude[idx]
            print(f"      {freq:.3f} Hz (power: {power:.1f})")
    else:
        dominant_frequency = 0.0

    return dominant_frequency


def match_frequency_to_node(detected_freq, frequency_mapping, tolerance=0.2):
    """Match detected frequency to a field node ID."""
    for expected_freq, node_id in frequency_mapping.items():
        if abs(detected_freq - expected_freq) <= tolerance:
            return node_id, expected_freq
    return None, None


def draw_label(image, position, text, font_scale=0.7, thickness=2):
    """Draw text label on image with background."""
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    padding = 5
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding - 20
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding - 20

    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    cv2.putText(image, text, (x, y - 20), font, font_scale, (0, 255, 0), thickness)
    cv2.circle(image, position, 5, (0, 255, 0), 2)
    cv2.circle(image, position, 3, (0, 255, 0), -1)


def analyze_video_and_label(video_path, output_image, frequency_mapping,
                            num_leds=4, tolerance=0.2, brightness_threshold=200,
                            min_distance=50, save_coordinates=None):
    """Complete analysis: detect LEDs, analyze frequencies, and create labeled image."""
    print(f"Analyzing video: {video_path}")
    print(f"Expected LEDs: {num_leds}")
    print(f"Frequency mapping: {frequency_mapping}")
    print(f"Tolerance: ±{tolerance} Hz")
    print(f"Min LED distance: {min_distance} px\n")

    print("Step 1: Detecting LED positions...")
    led_positions = detect_led_regions(video_path, num_leds, brightness_threshold, min_distance)

    if len(led_positions) < num_leds:
        print(f"Warning: Only {len(led_positions)} LEDs detected (expected {num_leds})")
        print(f"Try adjusting --brightness or --min-distance\n")

    print(f"  Detected {len(led_positions)} LED(s)\n")

    print("Step 2: Analyzing frequencies...")
    led_data = []

    for i, position in enumerate(led_positions):
        print(f"  LED {i + 1} at {position}:")

        timestamps, intensities, fps = extract_intensity_timeline(video_path, position)
        print(f"    - {len(intensities)} frames @ {fps:.1f} fps ({len(intensities) / fps:.2f}s)")
        print(f"    - Intensity: {np.min(intensities):.1f}-{np.max(intensities):.1f} (std: {np.std(intensities):.1f})")

        detected_freq = analyze_frequency(timestamps, intensities)
        print(f"    - Dominant frequency: {detected_freq:.2f} Hz")

        node_id, matched_freq = match_frequency_to_node(detected_freq, frequency_mapping, tolerance)

        if node_id:
            print(f"    - ✓ Matched: {node_id} (expected {matched_freq} Hz)")
            led_data.append({
                'position': position,
                'detected_freq': detected_freq,
                'node_id': node_id,
                'matched_freq': matched_freq
            })
        else:
            print(f"    - ✗ No match (tolerance ±{tolerance} Hz)")
            led_data.append({
                'position': position,
                'detected_freq': detected_freq,
                'node_id': 'UNKNOWN',
                'matched_freq': None
            })
        print()

    print("Step 3: Creating labeled image...")

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not read first frame")

    labeled_image = first_frame.copy()

    for led_info in led_data:
        position = led_info['position']
        node_id = led_info['node_id']
        detected_freq = led_info['detected_freq']
        label = f"{node_id}"
        draw_label(labeled_image, position, label)

    output_path = Path(output_image)
    if output_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        output_path = output_path.with_suffix('.png')
        print(f"  Changed extension to: {output_path}")

    success = cv2.imwrite(str(output_path), labeled_image)
    if not success:
        raise RuntimeError(f"Failed to save: {output_path}")

    print(f"  ✓ Saved: {output_path}\n")

    # Save coordinates to JSON file if requested
    field_nodes = {}
    if save_coordinates:
        for led_info in led_data:
            node_id = led_info['node_id']
            x, y = led_info['position']
            field_nodes[node_id] = {
                'x': int(x),
                'y': int(y),
                'detected_frequency': float(led_info['detected_freq']),
                'expected_frequency': float(led_info['matched_freq']) if led_info['matched_freq'] else None
            }

        # Save to JSON file
        json_path = Path(save_coordinates)
        with open(json_path, 'w') as f:
            json.dump(field_nodes, f, indent=2)

        print(f"Step 4: Saving field node coordinates...")
        print(f"  ✓ Coordinates saved to: {json_path}")
        print(f"\nField Node Data:")
        for node_id, data in field_nodes.items():
            print(f"  {node_id}: ({data['x']}, {data['y']}) @ {data['detected_frequency']:.2f} Hz")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i, led_info in enumerate(led_data):
        print(f"LED {i + 1}: {led_info['position']}")
        print(f"  Detected: {led_info['detected_freq']:.3f} Hz")
        print(f"  Node ID: {led_info['node_id']}")
        if led_info['matched_freq']:
            error = abs(led_info['detected_freq'] - led_info['matched_freq'])
            print(f"  Expected: {led_info['matched_freq']} Hz")
            print(f"  Error: {error:.3f} Hz ({error / led_info['matched_freq'] * 100:.1f}%)")
        print()

    return field_nodes


def main():
    parser = argparse.ArgumentParser(
        description="Analyze video to detect pulsing LEDs and identify field nodes",
        epilog="""
Example:
  python analyze_led_video_v3.py -i video.mp4 -o labeled.png \
      -m 1.0:FN_1 2.0:FN_2 3.0:FN_3 4.0:FN_4 \
      --save-coords field_nodes.json
        """
    )

    parser.add_argument('-i', '--input', required=True, help='Input video file')
    parser.add_argument('-o', '--output', required=True, help='Output image file')
    parser.add_argument('-m', '--mapping', nargs='+', required=True,
                        metavar='FREQ:ID', help='Frequency:NodeID pairs (e.g., 1.0:FN_1 2.0:FN_2)')
    parser.add_argument('--num-leds', type=int, default=4, help='Expected LEDs (default: 4)')
    parser.add_argument('--tolerance', type=float, default=0.2, help='Freq tolerance Hz (default: 0.2)')
    parser.add_argument('--brightness', type=int, default=200, help='Brightness threshold (default: 200)')
    parser.add_argument('--min-distance', type=int, default=50, help='Min LED distance px (default: 50)')
    parser.add_argument('--save-coords', type=str, default=None,
                        help='Save field node coordinates to JSON file (e.g., field_nodes.json)')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        frequency_mapping = parse_frequency_mapping(args.mapping)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        field_nodes = analyze_video_and_label(
            args.input, args.output, frequency_mapping,
            num_leds=args.num_leds, tolerance=args.tolerance,
            brightness_threshold=args.brightness, min_distance=args.min_distance,
            save_coordinates=args.save_coords
        )

        # Return field_nodes for programmatic use
        return field_nodes

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

'''
 Bash Script (2):
 python frequency-analysis_CG.py --input assets/overhead_video.mp4  --output assets/mapped_nodes.png  --mapping 1.0:FN_1 2.0:FN_2 3.0:FN_3 4.0:FN_4 --save-coords assets/field_node_coords.json
'''