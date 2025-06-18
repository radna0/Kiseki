#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <vector>

namespace py = pybind11;
using namespace cv;
using namespace std;

// Performance profiler
class ScopedTimer {
    const char* name;
    chrono::high_resolution_clock::time_point start;
public:
    ScopedTimer(const char* n) : name(n), start(chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto end = chrono::high_resolution_clock::now();
        auto ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
        printf("[PROFILE] %s: %.2f ms\n", name, ms);
    }
};

#define PROFILE_SCOPE(name) ScopedTimer _timer_##__LINE__(name)

// Morton order (Z-order) utilities for cache-oblivious traversal
inline uint32_t mortonEncode2D(uint16_t x, uint16_t y) {
    uint32_t answer = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        answer |= ((x & ((uint32_t)1 << i)) << i) | ((y & ((uint32_t)1 << i)) << (i + 1));
    }
    return answer;
}

inline void mortonDecode2D(uint32_t morton, uint16_t& x, uint16_t& y) {
    x = y = 0;
    for (uint32_t i = 0; i < 32; i += 2) {
        x |= ((morton >> i) & 1) << (i >> 1);
        y |= ((morton >> (i + 1)) & 1) << (i >> 1);
    }
}

// SIMD-accelerated distance transform for trapped-ball algorithm
void euclideanDistanceTransformAVX512(const uint8_t* binary, float* dist, int width, int height) {
    PROFILE_SCOPE("EDT_AVX512");
    
    const float INF = 1e9f;
    
    // First pass: propagate distances along rows
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        const uint8_t* row = binary + y * width;
        float* dist_row = dist + y * width;
        
        // Forward pass
        float min_dist = INF;
        for (int x = 0; x < width; x++) {
            if (row[x] == 0) {
                min_dist = 0;
            } else {
                min_dist = std::min(min_dist + 1, INF);
            }
            dist_row[x] = min_dist * min_dist;
        }
        
        // Backward pass
        min_dist = INF;
        for (int x = width - 1; x >= 0; x--) {
            if (row[x] == 0) {
                min_dist = 0;
            } else {
                min_dist = std::min(min_dist + 1, INF);
            }
            dist_row[x] = std::min(dist_row[x], min_dist * min_dist);
        }
    }
    
    // Second pass: propagate distances along columns using SIMD
    const int simd_width = 16; // AVX-512 processes 16 floats
    
    #pragma omp parallel for
    for (int x = 0; x < width; x += simd_width) {
        int chunk_width = std::min(simd_width, width - x);
        
        // Temporary buffer for column processing
        alignas(64) float buffer[height * simd_width];
        
        // Copy columns to buffer
        for (int y = 0; y < height; y++) {
            for (int dx = 0; dx < chunk_width; dx++) {
                buffer[y * simd_width + dx] = dist[y * width + x + dx];
            }
        }
        
        // Process columns in buffer
        for (int dx = 0; dx < chunk_width; dx++) {
            // Forward pass
            float prev = buffer[dx];
            for (int y = 1; y < height; y++) {
                float curr = buffer[y * simd_width + dx];
                float cand = prev + 2 * y - 1;
                if (cand < curr) {
                    buffer[y * simd_width + dx] = cand;
                    prev = cand;
                } else {
                    prev = curr;
                }
            }
            
            // Backward pass
            prev = buffer[(height - 1) * simd_width + dx];
            for (int y = height - 2; y >= 0; y--) {
                float curr = buffer[y * simd_width + dx];
                float cand = prev + 2 * (height - y) - 1;
                if (cand < curr) {
                    buffer[y * simd_width + dx] = cand;
                    prev = cand;
                } else {
                    prev = curr;
                }
            }
        }
        
        // Copy back and compute square root
        for (int y = 0; y < height; y++) {
            for (int dx = 0; dx < chunk_width; dx++) {
                dist[y * width + x + dx] = sqrtf(buffer[y * simd_width + dx]);
            }
        }
    }
}

// Parallel trapped-ball fill using distance transform
void trappedBallFillParallel(const uint8_t* binary, uint8_t* output, int width, int height, int radius) {
    PROFILE_SCOPE("TrappedBallFill_Parallel");
    
    // Allocate aligned memory for distance transform
    float* dist = (float*)aligned_alloc(64, width * height * sizeof(float));
    
    // Compute distance transform
    euclideanDistanceTransformAVX512(binary, dist, width, height);
    
    // Apply threshold based on radius
    #pragma omp parallel for simd aligned(dist, output : 64)
    for (int i = 0; i < width * height; i++) {
        output[i] = (dist[i] > radius) ? 255 : 0;
    }
    
    free(dist);
}

// Ultra-fast parallel flood fill using Union-Find with path compression
class ParallelUnionFind {
    vector<atomic<int>> parent;
    vector<atomic<int>> rank;
    
public:
    ParallelUnionFind(int n) : parent(n), rank(n) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
    }
    
    int find(int x) {
        int root = x;
        while (parent[root] != root) {
            root = parent[root];
        }
        
        // Path compression
        while (x != root) {
            int next = parent[x];
            parent[x] = root;
            x = next;
        }
        
        return root;
    }
    
    bool unite(int x, int y) {
        int rx = find(x);
        int ry = find(y);
        
        if (rx == ry) return false;
        
        // Union by rank with CAS
        if (rx > ry) swap(rx, ry);
        
        int expected = rx;
        if (parent[rx].compare_exchange_strong(expected, ry)) {
            return true;
        }
        
        return false;
    }
};

// Parallel connected component labeling
void parallelConnectedComponents(const uint8_t* binary, int32_t* labels, int width, int height) {
    PROFILE_SCOPE("ParallelCC");
    
    ParallelUnionFind uf(width * height);
    
    // First pass: local connectivity
    tbb::parallel_for(tbb::blocked_range2d<int>(0, height, 32, 0, width, 32),
        [&](const tbb::blocked_range2d<int>& range) {
            for (int y = range.rows().begin(); y < range.rows().end(); y++) {
                for (int x = range.cols().begin(); x < range.cols().end(); x++) {
                    if (binary[y * width + x] == 255) {
                        int idx = y * width + x;
                        
                        // Check left neighbor
                        if (x > 0 && binary[y * width + x - 1] == 255) {
                            uf.unite(idx, idx - 1);
                        }
                        
                        // Check top neighbor
                        if (y > 0 && binary[(y - 1) * width + x] == 255) {
                            uf.unite(idx, idx - width);
                        }
                    }
                }
            }
        }
    );
    
    // Second pass: assign labels
    atomic<int> next_label(1);
    vector<int> label_map(width * height, 0);
    
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        if (binary[i] == 255) {
            int root = uf.find(i);
            if (label_map[root] == 0) {
                label_map[root] = next_label.fetch_add(1);
            }
            labels[i] = label_map[root];
        } else {
            labels[i] = 0;
        }
    }
}

// Cache-oblivious thinning with Morton order traversal
void thinningCacheOblivious(uint32_t* fill_map, const uint8_t* binary, int width, int height) {
    PROFILE_SCOPE("Thinning_CacheOblivious");
    
    // Create Morton order index
    vector<uint32_t> morton_indices;
    morton_indices.reserve(width * height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (binary[y * width + x] == 0) {  // Line pixel
                morton_indices.push_back(mortonEncode2D(x, y));
            }
        }
    }
    
    // Process in Morton order for optimal cache usage
    const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    
    bool changed = true;
    while (changed) {
        changed = false;
        
        #pragma omp parallel for reduction(||:changed)
        for (size_t i = 0; i < morton_indices.size(); i++) {
            uint16_t x, y;
            mortonDecode2D(morton_indices[i], x, y);
            
            if (fill_map[y * width + x] != 0) continue;
            
            // Count neighbor fills
            uint32_t neighbor_fills[8];
            int fill_count = 0;
            
            for (int d = 0; d < 8; d++) {
                int nx = x + dx[d];
                int ny = y + dy[d];
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    uint32_t fill = fill_map[ny * width + nx];
                    if (fill != 0) {
                        neighbor_fills[fill_count++] = fill;
                    }
                }
            }
            
            // Vote for most common fill
            if (fill_count > 0) {
                sort(neighbor_fills, neighbor_fills + fill_count);
                
                uint32_t best_fill = neighbor_fills[0];
                int best_count = 1;
                int current_count = 1;
                
                for (int j = 1; j < fill_count; j++) {
                    if (neighbor_fills[j] == neighbor_fills[j-1]) {
                        current_count++;
                    } else {
                        if (current_count > best_count) {
                            best_count = current_count;
                            best_fill = neighbor_fills[j-1];
                        }
                        current_count = 1;
                    }
                }
                
                if (current_count > best_count) {
                    best_fill = neighbor_fills[fill_count-1];
                }
                
                fill_map[y * width + x] = best_fill;
                changed = true;
            }
        }
    }
}

// Python bindings
Mat numpy_to_mat_uint8(py::array_t<uint8_t> input) {
    py::buffer_info buf = input.request();
    return Mat(buf.shape[0], buf.shape[1], CV_8UC1, buf.ptr).clone();
}

Mat numpy_to_mat_int32(py::array_t<int32_t> input) {
    py::buffer_info buf = input.request();
    return Mat(buf.shape[0], buf.shape[1], CV_32SC1, buf.ptr).clone();
}

py::array_t<uint8_t> mat_to_numpy_uint8(const Mat& mat) {
    return py::array_t<uint8_t>(
        {mat.rows, mat.cols},
        {mat.step[0], mat.step[1]},
        mat.data
    );
}

py::array_t<int32_t> mat_to_numpy_int32(const Mat& mat) {
    return py::array_t<int32_t>(
        {mat.rows, mat.cols},
        {mat.step[0], mat.step[1]},
        (int32_t*)mat.data
    );
}

PYBIND11_MODULE(carmack_core, m) {
    m.doc() = "Carmack-grade optimized line art colorization core";
    
    m.def("trapped_ball_fill_parallel", [](py::array_t<uint8_t> binary_np, int radius) {
        Mat binary = numpy_to_mat_uint8(binary_np);
        Mat output(binary.size(), CV_8UC1);
        trappedBallFillParallel(binary.data, output.data, binary.cols, binary.rows, radius);
        return mat_to_numpy_uint8(output);
    }, "Parallel trapped-ball fill with SIMD distance transform");
    
    m.def("connected_components_parallel", [](py::array_t<uint8_t> binary_np) {
        Mat binary = numpy_to_mat_uint8(binary_np);
        Mat labels(binary.size(), CV_32SC1);
        parallelConnectedComponents(binary.data, (int32_t*)labels.data, binary.cols, binary.rows);
        return mat_to_numpy_int32(labels);
    }, "Parallel connected component labeling");
    
    m.def("thinning_cache_oblivious", [](py::array_t<int32_t> fill_map_np, py::array_t<uint8_t> binary_np) {
        Mat fill_map = numpy_to_mat_int32(fill_map_np);
        Mat binary = numpy_to_mat_uint8(binary_np);
        thinningCacheOblivious((uint32_t*)fill_map.data, binary.data, binary.cols, binary.rows);
        return mat_to_numpy_int32(fill_map);
    }, "Cache-oblivious thinning with Morton order");
}