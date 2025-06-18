#include <omp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <execution>

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

// Optimized distance transform using separable filters
void euclideanDistanceTransform(const uint8_t* binary, float* dist, int width, int height) {
    ScopedTimer timer("EDT_Optimized");
    
    const float INF = 1e9f;
    
    // First pass: horizontal distances
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
    
    // Second pass: vertical distances
    #pragma omp parallel for
    for (int x = 0; x < width; x++) {
        vector<float> column(height);
        
        // Copy column
        for (int y = 0; y < height; y++) {
            column[y] = dist[y * width + x];
        }
        
        // Forward pass
        float prev = column[0];
        for (int y = 1; y < height; y++) {
            float curr = column[y];
            float cand = prev + 2 * y - 1;
            if (cand < curr) {
                column[y] = cand;
                prev = cand;
            } else {
                prev = curr;
            }
        }
        
        // Backward pass
        prev = column[height - 1];
        for (int y = height - 2; y >= 0; y--) {
            float curr = column[y];
            float cand = prev + 2 * (height - y) - 1;
            if (cand < curr) {
                column[y] = cand;
                prev = cand;
            } else {
                prev = curr;
            }
        }
        
        // Copy back and compute square root
        for (int y = 0; y < height; y++) {
            dist[y * width + x] = sqrtf(column[y]);
        }
    }
}

// Parallel trapped-ball fill using distance transform
void trappedBallFillParallel(const uint8_t* binary, uint8_t* output, int width, int height, int radius) {
    ScopedTimer timer("TrappedBallFill_Parallel");
    
    // Allocate memory for distance transform
    vector<float> dist(width * height);
    
    // Compute distance transform
    euclideanDistanceTransform(binary, dist.data(), width, height);
    
    // Apply threshold based on radius
    #pragma omp parallel for simd
    for (int i = 0; i < width * height; i++) {
        output[i] = (dist[i] > radius) ? 255 : 0;
    }
}

// Parallel connected component labeling using Union-Find
class ParallelUnionFind {
    vector<atomic<int>> parent;
    
public:
    ParallelUnionFind(int n) : parent(n) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
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
        while (true) {
            int rx = find(x);
            int ry = find(y);
            
            if (rx == ry) return false;
            
            if (rx > ry) swap(rx, ry);
            
            int expected = rx;
            if (parent[rx].compare_exchange_weak(expected, ry)) {
                return true;
            }
        }
    }
};

// Parallel connected component labeling
void parallelConnectedComponents(const uint8_t* binary, int32_t* labels, int width, int height) {
    ScopedTimer timer("ParallelCC");
    
    ParallelUnionFind uf(width * height);
    
    // First pass: local connectivity
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
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

// Optimized thinning
void thinningOptimized(uint32_t* fill_map, const uint8_t* binary, int width, int height) {
    ScopedTimer timer("Thinning_Optimized");
    
    vector<pair<int, int>> line_pixels;
    
    // Find all line pixels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (binary[y * width + x] == 0) {
                line_pixels.push_back({x, y});
            }
        }
    }
    
    const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    
    bool changed = true;
    int iterations = 0;
    
    while (changed && iterations < 100) {
        changed = false;
        iterations++;
        
        #pragma omp parallel for reduction(||:changed)
        for (size_t i = 0; i < line_pixels.size(); i++) {
            int x = line_pixels[i].first;
            int y = line_pixels[i].second;
            
            if (fill_map[y * width + x] != 0) continue;
            
            // Count neighbor fills
            int fill_counts[256] = {0};
            
            for (int d = 0; d < 8; d++) {
                int nx = x + dx[d];
                int ny = y + dy[d];
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    uint32_t fill = fill_map[ny * width + nx];
                    if (fill > 0 && fill < 256) {
                        fill_counts[fill]++;
                    }
                }
            }
            
            // Find most common fill
            int best_fill = 0;
            int best_count = 0;
            
            for (int j = 1; j < 256; j++) {
                if (fill_counts[j] > best_count) {
                    best_count = fill_counts[j];
                    best_fill = j;
                }
            }
            
            if (best_fill > 0) {
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
    m.doc() = "Optimized line art colorization core";
    
    m.def("trapped_ball_fill_parallel", [](py::array_t<uint8_t> binary_np, int radius) {
        Mat binary = numpy_to_mat_uint8(binary_np);
        Mat output(binary.size(), CV_8UC1);
        trappedBallFillParallel(binary.data, output.data, binary.cols, binary.rows, radius);
        return mat_to_numpy_uint8(output);
    }, "Parallel trapped-ball fill with distance transform");
    
    m.def("connected_components_parallel", [](py::array_t<uint8_t> binary_np) {
        Mat binary = numpy_to_mat_uint8(binary_np);
        Mat labels(binary.size(), CV_32SC1);
        parallelConnectedComponents(binary.data, (int32_t*)labels.data, binary.cols, binary.rows);
        return mat_to_numpy_int32(labels);
    }, "Parallel connected component labeling");
    
    m.def("thinning_optimized", [](py::array_t<int32_t> fill_map_np, py::array_t<uint8_t> binary_np) {
        Mat fill_map = numpy_to_mat_int32(fill_map_np);
        Mat binary = numpy_to_mat_uint8(binary_np);
        thinningOptimized((uint32_t*)fill_map.data, binary.data, binary.cols, binary.rows);
        return mat_to_numpy_int32(fill_map);
    }, "Optimized thinning");
}