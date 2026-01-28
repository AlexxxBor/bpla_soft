import numpy as np
import json
import os
from datetime import datetime
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional
import math


class BeeHivePlacementSystem:
    def __init__(self, calibration_ref: Optional[Dict[str, Tuple[int, int, int]]] = None):
        self.calibration_ref = calibration_ref or {
            'grass': (46, 139, 87),
            'water': (30, 144, 255),
            'road': (128, 128, 128),
            'crop': (154, 205, 50)
        }
        self._ensure_directories()

    def _ensure_directories(self):
        os.makedirs('results', exist_ok=True)

    def _convert_to_serializable(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (tuple, list)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        return obj

    def apply_darkening(self, image: np.ndarray, factor: float = 0.95) -> np.ndarray:
        darkened = image.astype(np.float32) * factor
        darkened = np.clip(darkened, 0, 255)
        return darkened.astype(np.uint8)

    def calibrate_colors(self, image: np.ndarray) -> np.ndarray:
        calibrated = image.copy()
        hsv_img = self._rgb_to_hsv(image)

        green_mask = (
                (hsv_img[:, :, 0] >= 35) & (hsv_img[:, :, 0] <= 85) &
                (hsv_img[:, :, 1] >= 0.3) & (hsv_img[:, :, 1] <= 1.0) &
                (hsv_img[:, :, 2] >= 0.2) & (hsv_img[:, :, 2] <= 1.0)
        )
        if np.any(green_mask):
            current_mean = np.mean(image[green_mask], axis=0)
            target = np.array(self.calibration_ref['grass'])
            calibrated[green_mask] = np.clip(
                image[green_mask] * (target / (current_mean + 1e-8)),
                0, 255
            ).astype(np.uint8)

        blue_water = (
                (hsv_img[:, :, 0] >= 185) & (hsv_img[:, :, 0] <= 245) &
                (hsv_img[:, :, 1] >= 0.3) & (hsv_img[:, :, 1] <= 0.9) &
                (hsv_img[:, :, 2] >= 0.2) & (hsv_img[:, :, 2] <= 0.8)
        )
        dark_water = (
                (hsv_img[:, :, 1] < 0.4) &
                (hsv_img[:, :, 2] < 0.5)
        )
        water_mask = blue_water | dark_water

        if np.any(water_mask):
            target = np.array(self.calibration_ref['water'])
            calibrated[water_mask] = np.clip(
                image[water_mask] * 0.7 + target * 0.3,
                0, 255
            ).astype(np.uint8)

        return calibrated

    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        rgb = rgb.astype('float32') / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        mx = np.max(rgb, axis=2)
        mn = np.min(rgb, axis=2)
        df = mx - mn

        h = np.zeros_like(r)
        s = np.zeros_like(r)
        v = mx

        mask = df != 0
        h[mask & (mx == r)] = ((g[mask & (mx == r)] - b[mask & (mx == r)]) / df[mask & (mx == r)]) % 6
        h[mask & (mx == g)] = ((b[mask & (mx == g)] - r[mask & (mx == g)]) / df[mask & (mx == g)]) + 2
        h[mask & (mx == b)] = ((r[mask & (mx == b)] - g[mask & (mx == b)]) / df[mask & (mx == b)]) + 4
        h = h * 60
        s[mx != 0] = df[mx != 0] / (mx[mx != 0] + 1e-8)

        return np.stack((h, s, v), axis=2)

    def calculate_ndvig(self, image: np.ndarray) -> np.ndarray:
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        eps = 1e-8
        ndvig = (g - r) / (g + r + eps)
        return np.clip(ndvig, -1, 1)

    def calculate_exg(self, image: np.ndarray) -> np.ndarray:
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)
        eps = 1e-8
        denom = r + g + b + eps
        exg = (2 * g - r - b) / denom
        return np.clip(exg, -1, 1)

    def _apply_morphology(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return self._morphology_close(mask, kernel)

    def _morphology_close(self, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        dilated = np.zeros_like(mask)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                if kernel[i, j]:
                    shifted = np.roll(np.roll(mask, i - kernel.shape[0] // 2, axis=0), j - kernel.shape[1] // 2, axis=1)
                    if i - kernel.shape[0] // 2 < 0:
                        shifted[:abs(i - kernel.shape[0] // 2)] = 0
                    if j - kernel.shape[1] // 2 < 0:
                        shifted[:, :abs(j - kernel.shape[1] // 2)] = 0
                    dilated = np.maximum(dilated, shifted)

        eroded = np.ones_like(mask)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                if kernel[i, j]:
                    shifted = np.roll(np.roll(dilated, i - kernel.shape[0] // 2, axis=0), j - kernel.shape[1] // 2, axis=1)
                    if i - kernel.shape[0] // 2 < 0:
                        shifted[:abs(i - kernel.shape[0] // 2)] = 1
                    if j - kernel.shape[1] // 2 < 0:
                        shifted[:, :abs(j - kernel.shape[1] // 2)] = 1
                    eroded = np.minimum(eroded, shifted)

        return eroded

    def detect_danger_zones(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        hsv_img = self._rgb_to_hsv(image)

        road_mask = (
                (hsv_img[:, :, 1] < 0.3) &
                (hsv_img[:, :, 2] > 0.2) &
                (hsv_img[:, :, 2] < 0.8)
        )
        road_mask = self._apply_morphology(road_mask, kernel_size=7)

        blue_water = (
                (hsv_img[:, :, 0] >= 185) & (hsv_img[:, :, 0] <= 245) &
                (hsv_img[:, :, 1] >= 0.3) & (hsv_img[:, :, 1] <= 0.9) &
                (hsv_img[:, :, 2] >= 0.2) & (hsv_img[:, :, 2] <= 0.8)
        )
        dark_water = (
                (hsv_img[:, :, 1] < 0.4) &
                (hsv_img[:, :, 2] < 0.5)
        )
        water_mask = blue_water | dark_water
        water_mask = self._apply_morphology(water_mask, kernel_size=9)

        crop_mask = (
                (hsv_img[:, :, 0] >= 40) & (hsv_img[:, :, 0] <= 80) &
                (hsv_img[:, :, 1] >= 0.4) & (hsv_img[:, :, 2] >= 0.4)
        )
        crop_mask = self._apply_morphology(crop_mask, kernel_size=7)

        return {
            'roads': road_mask.astype(np.uint8) * 255,
            'water': water_mask.astype(np.uint8) * 255,
            'crops': crop_mask.astype(np.uint8) * 255
        }

    def calculate_distance_map(self, binary_mask: np.ndarray, max_distance: int = 100) -> np.ndarray:
        h, w = binary_mask.shape
        distance_map = np.zeros((h, w), dtype=np.float32)

        if not np.any(binary_mask > 0):
            return np.ones((h, w), dtype=np.float32)

        current_mask = binary_mask.copy()
        distance_value = 1.0

        while max_distance > 0 and np.any(current_mask == 0):
            kernel = np.ones((3, 3), np.uint8)
            dilated = np.zeros_like(current_mask)
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    if kernel[i, j]:
                        shifted = np.roll(np.roll(current_mask, i - 1, axis=0), j - 1, axis=1)
                        dilated = np.maximum(dilated, shifted)

            new_pixels = (dilated > 0) & (current_mask == 0)
            distance_map[new_pixels] = distance_value
            current_mask = dilated
            distance_value += 0.1
            max_distance -= 1

        max_val = np.max(distance_map)
        if max_val > 0:
            distance_map = distance_map / max_val
        return 1.0 - distance_map

    def generate_honey_base_map(self, ndvig_map: np.ndarray, exg_map: np.ndarray,
                                danger_masks: Dict[str, np.ndarray]) -> np.ndarray:
        h, w = ndvig_map.shape
        forest_mask = (ndvig_map > 0.3) & (exg_map > 0.1)

        combined_danger = np.zeros((h, w), dtype=np.uint8)
        for mask in danger_masks.values():
            combined_danger = np.maximum(combined_danger, mask)

        distance_map = self.calculate_distance_map(combined_danger)
        base_potential = (ndvig_map + exg_map) / 2
        base_potential = np.clip(base_potential, -1, 1)
        distance_factor = np.clip(distance_map * 1.5, 0, 1)
        honey_potential = base_potential * distance_factor

        honey_base_map = np.zeros((h, w), dtype=np.float32)
        honey_base_map[forest_mask] = honey_potential[forest_mask]
        honey_base_map[honey_base_map < 0.2] = 0

        max_value = np.max(honey_base_map)
        if max_value > 0:
            honey_base_map = honey_base_map / max_value
        else:
            honey_base_map = np.zeros((h, w), dtype=np.float32)

        return honey_base_map

    def find_multiple_hive_locations(self, honey_base_map: np.ndarray, num_locations: int = 5) -> List[Tuple[int, int]]:
        h, w = honey_base_map.shape
        if np.max(honey_base_map) == 0:
            return []

        threshold = np.percentile(honey_base_map[honey_base_map > 0], 75)
        high_potential_zones = honey_base_map >= threshold

        if not np.any(high_potential_zones):
            high_potential_zones = honey_base_map > 0

        ys, xs = np.where(high_potential_zones)
        if len(ys) == 0:
            return []

        values = honey_base_map[high_potential_zones]
        sorted_indices = np.argsort(values)[::-1]
        sorted_ys = ys[sorted_indices]
        sorted_xs = xs[sorted_indices]

        selected_points = []
        min_distance = max(h, w) // 10

        for i in range(len(sorted_ys)):
            y, x = sorted_ys[i], sorted_xs[i]
            if all(math.sqrt((y - py) ** 2 + (x - px) ** 2) > min_distance for py, px in selected_points):
                selected_points.append((y, x))
                if len(selected_points) >= num_locations:
                    break

        return selected_points

    def create_combined_overlay_map(self, image: np.ndarray, ndvig_map: np.ndarray,
                                    exg_map: np.ndarray, honey_base_map: np.ndarray,
                                    optimal_locations: List[Tuple[int, int]]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'results/combined_analysis_map_{timestamp}.png'

        overlay_img = image.copy().astype(np.float32)
        overlay_img = np.clip(overlay_img * 1.1, 0, 255)

        ndvig_overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
        ndvig_colormap = plt.get_cmap('RdYlGn')
        ndvig_min, ndvig_max = np.min(ndvig_map), np.max(ndvig_map)
        ndvig_range = ndvig_max - ndvig_min

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                value = ndvig_map[i, j]
                norm_value = (value - ndvig_min) / (ndvig_range + 1e-8) if ndvig_range > 0 else 0.5
                color = ndvig_colormap(norm_value)
                ndvig_overlay[i, j, 0] = color[0] * 255
                ndvig_overlay[i, j, 1] = color[1] * 255
                ndvig_overlay[i, j, 2] = color[2] * 255
                ndvig_overlay[i, j, 3] = 0.5 * 255

        exg_overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
        exg_colormap = plt.get_cmap('RdYlGn')
        exg_min, exg_max = np.min(exg_map), np.max(exg_map)
        exg_range = exg_max - exg_min

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                value = exg_map[i, j]
                norm_value = (value - exg_min) / (exg_range + 1e-8) if exg_range > 0 else 0.5
                color = exg_colormap(norm_value)
                exg_overlay[i, j, 0] = color[0] * 255
                exg_overlay[i, j, 1] = color[1] * 255
                exg_overlay[i, j, 2] = color[2] * 255
                exg_overlay[i, j, 3] = 0.5 * 255

        honey_overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
        honey_colormap = plt.get_cmap('YlGn')

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                value = honey_base_map[i, j]
                if value > 0.1:
                    color = honey_colormap(value)
                    honey_overlay[i, j, 0] = color[0] * 255
                    honey_overlay[i, j, 1] = color[1] * 255
                    honey_overlay[i, j, 2] = color[2] * 255
                    honey_overlay[i, j, 3] = 0.5 * 255

        def overlay_with_alpha(base, overlay):
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                base[:, :, c] = base[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
            return base

        overlay_img = overlay_with_alpha(overlay_img, ndvig_overlay)
        overlay_img = overlay_with_alpha(overlay_img, exg_overlay)
        overlay_img = overlay_with_alpha(overlay_img, honey_overlay)
        overlay_img = np.clip(overlay_img * 1.1, 0, 255)

        h, w = overlay_img.shape[:2]
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
        ]

        for i, (y, x) in enumerate(optimal_locations):
            marker_size = max(10, min(h, w) // 20)
            for dy in range(-marker_size, marker_size + 1):
                for dx in range(-marker_size, marker_size + 1):
                    if 0 <= y + dy < h and 0 <= x + dx < w:
                        if abs(dy) <= 1 or abs(dx) <= 1:
                            overlay_img[y + dy, x + dx, 0] = colors[i % len(colors)][0]
                            overlay_img[y + dy, x + dx, 1] = colors[i % len(colors)][1]
                            overlay_img[y + dy, x + dx, 2] = colors[i % len(colors)][2]
                        elif abs(dy) <= 3 or abs(dx) <= 3:
                            overlay_img[y + dy, x + dx, 0] = 255
                            overlay_img[y + dy, x + dx, 1] = 255
                            overlay_img[y + dy, x + dx, 2] = 255

            try:
                pil_img = Image.fromarray(overlay_img.astype(np.uint8))
                draw = ImageDraw.Draw(pil_img)
                draw.text((x + 20, y - 20), f"Location {i + 1}", fill=(255, 255, 255))
                draw.text((x + 20, y - 5), f"({x}, {y})", fill=(255, 255, 255))
                overlay_img = np.array(pil_img)
            except:
                pass

        legend_height = 150
        legend_width = 300
        if h > legend_height + 20 and w > legend_width + 20:
            legend = np.ones((legend_height, legend_width, 3), dtype=np.float32) * 255
            legend[40:45, :, :] = 0
            legend[80:85, :, :] = 0
            pil_legend = Image.fromarray(legend.astype(np.uint8))
            draw = ImageDraw.Draw(pil_legend)
            try:
                draw.text((10, 5), "COMBINED ANALYSIS MAP", fill=(0, 0, 0))
                draw.text((10, 50), "NDVIG: Red-Yellow-Green scale", fill=(0, 0, 0))
                draw.text((10, 90), "ExG: Red-Yellow-Green scale", fill=(0, 0, 0))
                draw.text((10, 130), "Honey Base: Yellow-Green scale", fill=(0, 0, 0))
            except:
                pass
            legend = np.array(pil_legend)
            overlay_img[h - legend_height - 10:h - 10, w - legend_width - 10:w - 10] = legend

        overlay_img = overlay_img.astype(np.uint8)
        Image.fromarray(overlay_img).save(output_path)

    def visualize_results(self, image: np.ndarray, ndvig_map: np.ndarray, exg_map: np.ndarray,
                          honey_base_map: np.ndarray, danger_masks: Dict[str, np.ndarray],
                          optimal_locations: List[Tuple[int, int]],
                          output_dir: str = 'results') -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.figure(figsize=(10, 8))
        plt.imshow(ndvig_map, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='NDVIG Value')
        plt.title('NDVIG Vegetation Index Map')
        plt.savefig(os.path.join(output_dir, f'ndvig_map_{timestamp}.png'))
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.imshow(exg_map, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='ExG Value')
        plt.title('ExG Vegetation Index Map')
        plt.savefig(os.path.join(output_dir, f'exg_map_{timestamp}.png'))
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.imshow(honey_base_map, cmap='YlGn', vmin=0, vmax=1)
        plt.colorbar(label='Honey Base Potential')
        plt.title('Honey Base Potential Map')
        plt.savefig(os.path.join(output_dir, f'honey_base_map_{timestamp}.png'))
        plt.close()

        # Удалена карта опасных зон

        overlay_img = image.copy()
        colors = np.zeros((honey_base_map.shape[0], honey_base_map.shape[1], 4), dtype=np.uint8)
        for i in range(honey_base_map.shape[0]):
            for j in range(honey_base_map.shape[1]):
                value = honey_base_map[i, j]
                if value > 0.1:
                    colors[i, j, 0] = int(255 * (1 - value))
                    colors[i, j, 1] = int(255 * value)
                    colors[i, j, 2] = 0
                    colors[i, j, 3] = int(128 * value)

        for c in range(3):
            overlay_img[:, :, c] = np.where(
                colors[:, :, 3] > 0,
                (overlay_img[:, :, c] * (255 - colors[:, :, 3]) + colors[:, :, c] * colors[:, :, 3]) // 255,
                overlay_img[:, :, c]
            )

        h, w = overlay_img.shape[:2]
        point_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
        ]

        for i, (y, x) in enumerate(optimal_locations):
            marker_size = max(5, min(h, w) // 40)
            y1, y2 = max(0, y - marker_size), min(h, y + marker_size + 1)
            x1, x2 = max(0, x - marker_size), min(w, x + marker_size + 1)
            overlay_img[y1:y2, x] = point_colors[i % len(point_colors)]
            overlay_img[y, x1:x2] = point_colors[i % len(point_colors)]
            pil_img = Image.fromarray(overlay_img)
            draw = ImageDraw.Draw(pil_img)
            try:
                draw.text((x + 10, y - 20), f"Location {i + 1}", fill=(255, 255, 255))
                draw.text((x + 10, y - 5), f"({x}, {y})", fill=(255, 255, 255))
            except:
                pass
            overlay_img = np.array(pil_img)

        pil_img.save(os.path.join(output_dir, f'honey_base_overlay_{timestamp}.png'))

    def process_image(self, image_path: str) -> Dict[str, Any]:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        darkened_img = self.apply_darkening(img_array, factor=0.95)
        calibrated_img = self.calibrate_colors(darkened_img)
        ndvig_map = self.calculate_ndvig(calibrated_img)
        exg_map = self.calculate_exg(calibrated_img)
        danger_masks = self.detect_danger_zones(calibrated_img)
        honey_base_map = self.generate_honey_base_map(ndvig_map, exg_map, danger_masks)
        optimal_locations = self.find_multiple_hive_locations(honey_base_map, num_locations=5)
        self.visualize_results(calibrated_img, ndvig_map, exg_map, honey_base_map, danger_masks, optimal_locations)
        self.create_combined_overlay_map(calibrated_img, ndvig_map, exg_map, honey_base_map, optimal_locations)

        results = {
            "timestamp": str(datetime.now()),
            "image_path": image_path,
            "optimal_locations": optimal_locations,
            "danger_zones": {
                zone_type: int(np.sum(mask > 0))
                for zone_type, mask in danger_masks.items()
            }
        }

        serializable_results = self._convert_to_serializable(results)
        output_json_path = os.path.join('results', f'complete_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(output_json_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)

        return serializable_results


if __name__ == "__main__":
    system = BeeHivePlacementSystem()
    test_image_path = "tst.png"

    if not os.path.exists(test_image_path):
        print(f"Файл изображения не найден: {test_image_path}")
        exit(1)

    results = system.process_image(test_image_path)
    print("\nАнализ завершен успешно!")