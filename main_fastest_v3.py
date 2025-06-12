#!/usr/bin/env python3
"""
High-quality line art colorization with optimized performance
Maintains original algorithm quality while reducing processing time
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from kiseki.logging import logger, Profiler
from concurrent.futures import ThreadPoolExecutor

# Define named constants
LINE_ART_THRESHOLD =150
BINARY_THRESHOLD =220
RADIUS_VALUES = [20,10,5]
METHODS = ["max", "mean", "mean"]

def get_ball_structuring_element(radius: int) -> np.ndarray:
 """Optimized ball structuring element creation"""
 size =2 * radius +1
 return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def exclude_area(image: np.ndarray, radius: int) -> np.ndarray:
 """Efficient boundary exclusion with precomputation"""
 kernel = get_ball_structuring_element(radius)
 return cv2.erode(image, kernel)

def trapped_ball_fill_single(image: np.ndarray, seed_point: tuple, radius: int) -> np.ndarray:
 """Optimized single trapped ball fill with reduced buffer usage"""
 ball = get_ball_structuring_element(radius)

 # Create working buffers
 pass1 = np.full(image.shape,255, dtype=np.uint8)
 im_inv = cv2.bitwise_not(image)

 # First flood fill with border handling
 mask = cv2.copyMakeBorder(im_inv,1,1,1,1, cv2.BORDER_CONSTANT,0)
 cv2.floodFill(pass1, mask, seed_point,0, flags=4)

 # Morphological operations
 pass1 = cv2.dilate(pass1, ball)
 mask = cv2.copyMakeBorder(pass1,1,1,1,1, cv2.BORDER_CONSTANT,0)
 pass2 = np.full_like(pass1,255)
 cv2.floodFill(pass2, mask, seed_point,0, flags=4)

 return cv2.erode(pass2, ball)

def get_unfilled_points(image: np.ndarray, step: int =5) -> list:
 """Efficient seed point selection with grid sampling"""
 points = []
 h, w = image.shape

 for y in range(0, h, step):
 for x in range(0, w, step):
 if image[y, x] ==255:
 points.append((x, y))

 return points

def trapped_ball_fill_multi(
 image: np.ndarray, radius: int, method: str = "mean", max_iter: int =1000
) -> list:
 """Optimized multi-fill with efficient seed selection and batch processing"""
 unfill_area = image.copy()
 filled_area = []
 filled_area_size = []

 # Precompute excluded area once
 excluded = exclude_area(unfill_area, radius)

 # Get seed points with adaptive step size
 step = max(1, radius //3)
 points = get_unfilled_points(excluded, step)

 with ThreadPoolExecutor() as executor:
 for seed_point in points:
 if unfill_area[seed_point[1], seed_point[0]] !=255:
continue

 # Process fill
 fill = trapped_ball_fill_single(unfill_area, seed_point, radius)
 fill_points = np.where(fill ==0)

 # Skip small fills immediately
 if len(fill_points[0]) <10:
continue

 # Update area
 unfill_area[fill ==0] =0
 filled_area.append(fill_points)
 filled_area_size.append(len(fill_points[0]))

 if len(filled_area) >= max_iter:
 break

 # Filter by size if needed
 if filled_area and method != "none":
 sizes = np.array(filled_area_size)
 if method == "max":
 threshold = np.max(sizes)
 elif method == "mean":
 threshold = np.mean(sizes)
 elif method == "median":
 threshold = np.median(sizes)
 else:
 threshold =0

 return [fill for fill, size in zip(filled_area, sizes) if size >= threshold]

 return filled_area

def flood_fill_multi(image: np.ndarray, max_iter: int =20000) -> list:
 """Optimized flood fill with efficient seed selection"""
 unfill_area = image.copy()
 filled_area = []
 points = get_unfilled_points(unfill_area,10)

 with ThreadPoolExecutor() as executor:
 for seed_point in points:
 if unfill_area[seed_point[1], seed_point[0]] !=255:
continue

 # Process fill
 pass1 = np.full(unfill_area.shape,255, dtype=np.uint8)
 mask = cv2.copyMakeBorder(
 cv2.bitwise_not(unfill_area),1,1,1,1, cv2.BORDER_CONSTANT,0
 )
 cv2.floodFill(pass1, mask, seed_point,0, flags=4)

 # Update area
 unfill_area[pass1 ==0] =0
 filled_area.append(np.where(pass1 ==0))

 if len(filled_area) >= max_iter:
 break

 return filled_area

def merge_fill(fillmap: np.ndarray, max_iter: int =3) -> np.ndarray:
 """Optimized region merging with efficient area processing"""
 result = fillmap.copy()
 h, w = fillmap.shape

 for _ in range(max_iter):
 unique_ids = np.unique(result)
 regions = []

 # Collect region info
 for region_id in unique_ids:
 if region_id ==0:
continue

 points = np.where(result == region_id)
 area = len(points[0])
 if area ==0:
continue

 regions.append(
 {
 "id": region_id,
 "points": points,
 "area": area,
 "rect": (
 min(points[1]),
 min(points[0]),
 max(points[1]),
 max(points[0]),
 ),
 }
 )

 # Sort by area to process small regions first
 regions.sort(key=lambda x: x["area"])

 # Process regions
 for region in regions:
 if region["area"] >500:
continue

 # Get border points
 x1, y1, x2, y2 = region["rect"]
 border_rect = (
 max(0, x1 -2),
 max(0, y1 -2),
 min(w, x2 +3),
 min(h, y2 +3),
 )

 # Create local patch
 local_patch = np.zeros(
 (border_rect[3] - border_rect[1], border_rect[2] - border_rect[0]),
 dtype=np.uint8,
 )
 local_y = region["points"][0] - border_rect[1]
 local_x = region["points"][1] - border_rect[0]
 local_patch[local_y, local_x] =255

 # Find border pixels
 kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
 border = cv2.dilate(local_patch, kernel) - local_patch

 # Get neighbors
 global_y, global_x = np.where(border)
 global_y += border_rect[1]
 global_x += border_rect[0]
 neighbors = result[global_y, global_x]
 neighbors = neighbors[neighbors !=0]
 neighbors = neighbors[neighbors != region["id"]]

 if len(neighbors) ==0:
continue

 # Find most common neighbor
 values, counts = np.unique(neighbors, return_counts=True)
 new_id = values[np.argmax(counts)]

 # Apply merge
 result[region["points"]] = new_id

 if len(np.unique(result)) == len(unique_ids):
 break

 return result

def thinning(fillmap: np.ndarray, binary: np.ndarray, max_iter: int =10) -> np.ndarray:
 """Quality-focused thinning with early termination"""
 result = fillmap.copy()
 line_mask = (binary ==0) & (result ==0)
 kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

 for _ in range(max_iter):
 # Create influence map
 influence = cv2.dilate(result.astype(np.uint8), kernel)

 # Update only line pixels
 update_mask = line_mask & (influence >0)
 if not np.any(update_mask):
 break

 result[update_mask] = influence[update_mask]
 line_mask = (binary ==0) & (result ==0)

 return result

def process_image(image_path: str, output_dir: str, resize: int = None) -> None:
 """Quality-focused processing pipeline with performance optimizations"""
 logger.info(f"Processing: {image_path}")
 start_time = time.time()

 # Load and prepare image
 img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

 if img is None:
 logger.error(f"Error loading image: {image_path}")
 return

 # Handle resizing
 if resize:
 h, w = img.shape[:2]
 scale = resize / max(h, w)
 img = cv2.resize(img, (int(w * scale), int(h * scale)))

 # Extract line art
 if img.shape[2] ==4:
 alpha = img[:, :,3]
 mask = alpha >100
 line_art = np.full(img.shape[:2],255, dtype=np.uint8)
 line_art[~mask] =0
 else:
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 _, line_art = cv2.threshold(gray, LINE_ART_THRESHOLD,255, cv2.THRESH_BINARY_INV)

 # Prepare binary image
 _, binary = cv2.threshold(line_art, BINARY_THRESHOLD,255, cv2.THRESH_BINARY_INV)
 cv2.imwrite(str(Path(output_dir) / "line_art.png"), line_art)

 # Multi-scale trapped-ball filling
 result = binary.copy()
 fills = []

 # Use progressively smaller radii
 for radius, method in zip(RADIUS_VALUES, METHODS):
 logger.info(f"Processing radius: {radius}")
 fill = trapped_ball_fill_multi(result, radius, method)
 fills.extend(fill)

 # Update result
 mask = np.zeros_like(result, dtype=bool)
 for f in fill:
 mask[f] = True
 result[mask] =0

 # Final flood fill
 logger.info("Processing flood fill")
 flood_fills = flood_fill_multi(result)
 fills.extend(flood_fills)

 # Build and merge fill map
 logger.info("Building fill map")
 fillmap = np.zeros_like(result, dtype=np.int32)
 for idx, fill in enumerate(fills):
 fillmap[fill] = idx +1

 logger.info("Merging regions")
 merged = merge_fill(fillmap)

 # Quality thinning
 logger.info("Thinning")
 final = thinning(merged, binary)

 # Output results
 logger.info("Generating output")
 output_dir = Path(output_dir)
 output_dir.mkdir(exist_ok=True, parents=True)

 # Save fill map
 cv2.imwrite(str(output_dir / "fill_map.png"), final.astype(np.uint16))

 # Create colored visualization
 colored = np.zeros((*binary.shape,3), dtype=np.uint8)
 unique_ids = np.unique(final)
 color_map = {}

 for region_id in unique_ids:
 if region_id ==0:
 color_map[region_id] = [0,0,0]
continue

 color = np.random.randint(50,255, size=3).tolist()
 color_map[region_id] = color
 mask = final == region_id
 colored[mask] = color

 # Apply original lines
 line_mask = binary ==0
 colored[line_mask] = [0,0,0]

 cv2.imwrite(str(output_dir / "colored.png"), colored)

 # Performance stats
 total_time = time.time() - start_time
 logger.info(f"Total processing time: {total_time:.2f} seconds")
 logger.info(f"Results saved to {output_dir}/")

def main() -> None:
 parser = argparse.ArgumentParser(
 description="Quality-Optimized Line Art Colorization"
 )
 parser.add_argument("input", help="Input image path")
 parser.add_argument("-o", "--output", default="output", help="Output directory")
 parser.add_argument(
 "--resize", type=int, help="Resize to max dimension (maintains aspect ratio)"
 )

 args = parser.parse_args()

 if not Path(args.input).exists():
 logger.error(f"Input file not found: {args.input}")
 return

 process_image(args.input, args.output, args.resize)
 logger.info("Processing complete")

if __name__ == "__main__":
 with Profiler("Quality Line Art Colorization"):
 main()
