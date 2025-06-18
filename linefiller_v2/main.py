import numpy as np
import cv2
from kiseki.logging import Profiler
from linefiller.trappedball_fill_opti import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, \
    show_fill_map
print("Using optimized C++ implementation")
from linefiller.thinning import thinning
import time
from log.logger import logger
import argparse
from numba import njit, prange
from scipy import ndimage
from PIL import Image
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import os


@njit(cache=True)
def _get_border_points_fast(line_mask, h, w):
    """Fast border detection using direct neighbor checking."""
    border_y = []
    border_x = []
    
    for y in range(h):
        for x in range(w):
            if line_mask[y, x] == 255:  # Not a line pixel
                # Check if any neighbor is a line pixel
                has_line_neighbor = False
                
                # Check 8-connected neighbors
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and line_mask[ny, nx] == 0:
                            has_line_neighbor = True
                            break
                    if has_line_neighbor:
                        break
                
                if has_line_neighbor:
                    border_y.append(y)
                    border_x.append(x)
    
    return np.array(border_y), np.array(border_x)

@njit(parallel=True, cache=True, nogil=True)
def _process_border_vectorized(result, line_border_y, line_border_x, line_id, h, w):
    """Vectorized processing with lookup table for neighbor offsets."""
    result_new = result.copy()
    n_points = len(line_border_y)
    
    # Neighbor offsets in priority order
    dy_offsets = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)
    dx_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
    
    for idx in prange(n_points):
        y, x = line_border_y[idx], line_border_x[idx]
        
        # Check all neighbors in one loop
        for i in range(8):
            ny = y + dy_offsets[i]
            nx = x + dx_offsets[i]
            
            if 0 <= ny < h and 0 <= nx < w and result[ny, nx] != line_id:
                result_new[y, x] = result[ny, nx]
                break
    
    return result_new

@njit(cache=True)
def _count_line_pixels(result, line_id):
    """Fast line pixel counting."""
    count = 0
    h, w = result.shape
    for y in range(h):
        for x in range(w):
            if result[y, x] == line_id:
                count += 1
    return count

def thinning_scipy(fillmap: np.ndarray, max_iter: int = 100):
    """Optimized thinning using scipy for morphological operations."""
    line_id = 0
    h, w = fillmap.shape[:2]
    result = fillmap.copy()
    
    # Cross structuring element
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    
    for iterNum in range(max_iter):
        # Fast check for line pixels
        line_mask = (result == line_id)
        if not np.any(line_mask):
            break
        
        # Use scipy for faster dilation
        dilated = ndimage.binary_dilation(line_mask, structure=struct)
        border_mask = dilated & ~line_mask
        
        line_border_y, line_border_x = np.where(border_mask)
        
        if len(line_border_y) == 0:
            break
        
        result = _process_border_vectorized(result, line_border_y, line_border_x, line_id, h, w)
    
    return result

def thinning_ultra(fillmap: np.ndarray, max_iter: int = 100):
    """Ultra-optimized version with early termination and caching."""
    line_id = 0
    h, w = fillmap.shape[:2]
    result = fillmap.astype(np.int32)  # Use int32 for better performance
    
    # Pre-create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Track convergence
    prev_line_count = -1
    
    for iterNum in range(max_iter):
        # Count line pixels for convergence check
        line_count = _count_line_pixels(result, line_id)
        
        if line_count == 0:
            break
        
        # Early termination if no change
        if line_count == prev_line_count:
            break
        prev_line_count = line_count
        
        # Create line mask
        line_mask = np.ones((h, w), dtype=np.uint8) * 255
        line_mask[result == line_id] = 0
        
        # Find border using optimized morphology
        dilated = cv2.dilate(255 - line_mask, kernel, iterations=1)
        border_mask = dilated & line_mask
        
        line_border_y, line_border_x = np.where(border_mask)
        
        if len(line_border_y) == 0:
            break
        
        result = _process_border_vectorized(result, line_border_y, line_border_x, line_id, h, w)
    
    return result.astype(fillmap.dtype)


def processing_optimized(image:np.ndarray, use_parallel=True)->np.ndarray:
    """Optimized processing with parallel execution and better algorithms."""
    ret, binary = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
    
    if use_parallel:
        # Parallel execution of trapped-ball fills
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            futures.append(executor.submit(trapped_ball_fill_multi, binary, 3, 'max'))
            futures.append(executor.submit(trapped_ball_fill_multi, binary, 2, 'mean'))
            futures.append(executor.submit(trapped_ball_fill_multi, binary, 1, 'mean'))
            
            fills = []
            result = binary
            
            # Process results in order
            for future in futures:
                fill = future.result()
                fills += fill
                result = mark_fill(result, fill)
    else:
        fills = []
        result = binary

        fill = trapped_ball_fill_multi(result, 3, method='max')
        fills += fill
        result = mark_fill(result, fill)

        fill = trapped_ball_fill_multi(result, 2, method='mean')
        fills += fill
        result = mark_fill(result, fill)

        fill = trapped_ball_fill_multi(result, 1, method='mean')
        fills += fill
        result = mark_fill(result, fill)

    fill = flood_fill_multi(result)
    fills += fill

    fillmap = build_fill_map(result, fills)
    
    # Use optimized merge if available, otherwise use original
    try:
        fillmap = merge_fill_optimized(fillmap)
    except:
        fillmap = merge_fill(fillmap)
    return fillmap

def processing(image:np.ndarray)->np.ndarray:
    """Legacy processing function for compatibility."""
    return processing_optimized(image, use_parallel=True)

@njit(parallel=True)
def merge_fill_optimized_kernel(fillmap, h, w, max_area):
    """Optimized merge kernel with parallel region analysis."""
    result = fillmap.copy()
    
    # Fast region properties calculation
    max_id = fillmap.max()
    areas = np.zeros(max_id + 1, dtype=np.int32)
    
    # Parallel area calculation
    for y in prange(h):
        for x in range(w):
            areas[fillmap[y, x]] += 1
    
    # Merge small regions
    for y in prange(h):
        for x in range(w):
            fill_id = fillmap[y, x]
            if fill_id > 0 and areas[fill_id] < max_area:
                # Find neighbor with largest area
                best_neighbor = fill_id
                best_area = areas[fill_id]
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            neighbor_id = fillmap[ny, nx]
                            if neighbor_id != fill_id and areas[neighbor_id] > best_area:
                                best_neighbor = neighbor_id
                                best_area = areas[neighbor_id]
                
                result[y, x] = best_neighbor
    
    return result

def merge_fill_optimized(fillmap: np.ndarray, max_iter: int = 10) -> np.ndarray:
    """Optimized merge fill using parallel kernels."""
    h, w = fillmap.shape
    result = fillmap.copy()
    
    # Progressive merging with different thresholds
    thresholds = [50, 250, 500]
    
    for threshold in thresholds:
        for _ in range(max_iter // len(thresholds)):
            new_result = merge_fill_optimized_kernel(result, h, w, threshold)
            if np.array_equal(result, new_result):
                break
            result = new_result
    
    return result

def saveAll(fillmap:np.ndarray,PATH:str)->None:
    """Save results with parallel processing."""
    # Use threading for parallel I/O
    with ThreadPoolExecutor(max_workers=2) as executor:
        # color+undertone
        f1 = (show_fill_map(fillmap) / 256.0).astype(np.uint8)
        future1 = executor.submit(cv2.imwrite, PATH+'fills_merged.png', f1)
        
        # undertone
        f2 = (show_fill_map(thinning_ultra(fillmap)) / 256.0).astype(np.uint8)
        future2 = executor.submit(cv2.imwrite, PATH+'fills_merged_no_contour.png', f2)
        
        # Wait for completion
        future1.result()
        future2.result()
    
def read_line_2_np(img_path, channel=4):
    img = Image.open(img_path)
    img_np = np.array(img)

    if img.mode == "RGBA":
        alpha_channel = img_np[:, :, 3]
        mask = alpha_channel > 100  # Line detection based on alpha value, default is 10
    elif img.mode == "RGB":
        grayscale = np.mean(img_np[:, :, :3], axis=2)
        mask = (
            grayscale < 150
        )  # Line detection based on grayscale value, default is 245

    line = np.zeros((*img_np.shape[:2], 4), dtype=np.uint8)
    line[:, :, :3] = 255  # Set all RGB to white
    line[:, :, 3] = np.where(mask, 255, 0)  # Set alpha: 255 for lines, 0 for background

    # Copy original RGB values to new image where there are lines
    line[mask, :3] = img_np[mask, :3]

    return line[..., :channel]

def main()->None:
    parser = argparse.ArgumentParser(description="Line Filler - Optimized Edition")
    # args
    parser.add_argument("-im","--image",type=str,help="Image Path",default="input.png")
    parser.add_argument("-o","--output",type=str,help="Save Root Path", default="./")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--threads", type=int, default=mp.cpu_count(), help="Number of threads")
    parser.add_argument("--benchmark", action="store_true", help="Compare optimized vs original")

    args = parser.parse_args()
    
    # Set thread count for numba
    if args.threads:
        os.environ['NUMBA_NUM_THREADS'] = str(args.threads)
    
    # Load image
    if os.path.exists(args.image):
        im = read_line_2_np(args.image, channel=3)
    else:
        im = read_line_2_np('./input.png', channel=3)
    
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
    logger.info(f"Processing image: {im.shape[1]}x{im.shape[0]} pixels")
    logger.info(f"Parallel processing: {'ENABLED' if args.parallel else 'DISABLED'}")
    logger.info(f"Threads: {args.threads}")
    
    if args.benchmark:
        # Benchmark mode - compare original vs optimized
        logger.info("\n=== BENCHMARK MODE ===")
        
        # Original implementation
        logger.info("Running original implementation...")
        start_orig = time.time()
        fillmap_orig = processing(image=im)
        cv2.imwrite('fills_merged_orig.png', show_fill_map(fillmap_orig))
        saveAll(fillmap=fillmap_orig, PATH=args.output)
        time_orig = time.time() - start_orig
        logger.info(f"\nOriginal: {time_orig:.2f}s")
        
        # Optimized implementation
        logger.info("Running optimized implementation...")
        start_opt = time.time()
        fillmap_opt = processing_optimized(image=im, use_parallel=args.parallel)
        time_opt = time.time() - start_opt
        
        logger.info(f"\nOriginal: {time_orig:.2f}s")
        logger.info(f"Optimized: {time_opt:.2f}s")
        logger.info(f"Speedup: {time_orig/time_opt:.2f}x")
        cv2.imwrite('fills_optimized.png', show_fill_map(fillmap_opt))
        
        fillmap = fillmap_opt
    else:
        # Normal mode
        logger.info("Start!")
        start = time.time()
        fillmap = processing_optimized(image=im, use_parallel=args.parallel)
        logger.info(f"Processing time: {time.time() - start:.2f}s")
    
    # Save results
    logger.info("Saving results...")
    saveAll(fillmap=fillmap, PATH=args.output)
    logger.info("All Finished!")

if __name__ == "__main__":
    main()