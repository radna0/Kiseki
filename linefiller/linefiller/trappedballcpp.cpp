#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>

using namespace cv;
using namespace std;

namespace py = pybind11;

//--------------------------------------------------------------
// Helper structure to store fill information.
struct FillInfo {
  int id;
  vector<Point> pts; // All pixel coordinates belonging to the fill.
  int area;          // Number of pixels.
  Rect rect;         // Bounding rectangle.
};


// define pair<vector<int>, vector<int>>
using PointPair = pair<vector<int>, vector<int>>;
//--------------------------------------------------------------
// Helper: get a ball-shaped structuring element
Mat getBallStructuringElement(int radius) {
  return getStructuringElement(MORPH_ELLIPSE,
                               Size(2 * radius + 1, 2 * radius + 1));
}

// Helper function: Convert a NumPy array (grayscale uint8) to a Mat.
Mat numpy_uint8_to_cv_mat(py::array_t<uint8_t> input) {
  py::buffer_info buf = input.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Input image must be 2-dimensional (grayscale).");
  }
  int rows = static_cast<int>(buf.shape[0]);
  int cols = static_cast<int>(buf.shape[1]);
  // Create a Mat that shares the data. Cloning to own the data.
  Mat mat(rows, cols, CV_8UC1, buf.ptr);
  return mat.clone();
}

// Helper function: Convert a NumPy array (int32) to a Mat.
Mat numpy_int32_to_cv_mat(py::array_t<int> input) {
  py::buffer_info buf = input.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Input image must be 2-dimensional (int32).");
  }
  int rows = static_cast<int>(buf.shape[0]);
  int cols = static_cast<int>(buf.shape[1]);
  // Create a Mat that shares the data. Cloning to own the data.
  Mat mat(rows, cols, CV_32SC1, buf.ptr);
  return mat.clone();
}

pair<vector<int>, vector<int>> ultraFastWhere(const Mat &fill, int target) {
  // Ensure the input is continuous and of type can be CV_8UC1 or CV_32SC1.
  CV_Assert(fill.isContinuous());

  int n_rows = fill.rows;
  int n_cols = fill.cols;

  // First pass: count the number of target pixels in each row.
  vector<int> row_counts(n_rows, 0);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_rows; i++) {
    int count = 0;
    const uchar *row_ptr = fill.ptr<uchar>(i);
    for (int j = 0; j < n_cols; j++) {
      if (row_ptr[j] == static_cast<uchar>(target))
        count++;
    }
    row_counts[i] = count;
  }

  // Compute the total number of target pixels.
  int total = accumulate(row_counts.begin(), row_counts.end(), 0);

  // Compute prefix sum offsets.
  // offsets[i] will give the starting index in the output arrays for row i.
  vector<int> offsets(n_rows + 1, 0);
  partial_sum(row_counts.begin(), row_counts.end(), offsets.begin() + 1);

  // Allocate the output arrays of exact size.
  vector<int> rows(total), cols(total);

// Second pass: fill in the indices.
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_rows; i++) {
    int offset = offsets[i];
    int cnt = 0;
    const uchar *row_ptr = fill.ptr<uchar>(i);
    for (int j = 0; j < n_cols; j++) {
      if (row_ptr[j] == static_cast<uchar>(target)) {
        rows[offset + cnt] = i;
        cols[offset + cnt] = j;
        cnt++;
      }
    }
  }

  return make_pair(rows, cols);
}

vector<Point> get_unfilled_point(const Mat &image) {
  // Use ultraFastWhere with target value 255.
  auto indices = ultraFastWhere(image, 255);
  const vector<int> &row_indices = indices.first;  // y-values
  const vector<int> &col_indices = indices.second; // x-values

  // Combine into a vector of Points: each row is (x, y)
  vector<Point> points;
  points.reserve(row_indices.size());
  for (size_t i = 0; i < row_indices.size(); i++) {
    points.push_back(Point(col_indices[i], row_indices[i]));
  }
  return points;
}

/* //--------------------------------------------------------------
// Helper: get all points in image where pixel == 255
vector<Point> getUnfilledPoints(const Mat& image) {
    """Get points belong to unfilled(value==255) area.

    # Arguments
        image: an image.

    # Returns
        an array of points.
    """

}
 */
//--------------------------------------------------------------
// Helper: Exclude areas near the boundary by erosion
Mat excludeArea(const Mat &image, int radius) {
  Mat eroded;
  Mat se = getBallStructuringElement(radius);
  morphologyEx(image, eroded, MORPH_ERODE, se, Point(-1, -1), 1);
  return eroded;
}

//--------------------------------------------------------------
// Helper: create a full 255 image (uchar)
Mat full255(Size size) { return Mat(size, CV_8UC1, Scalar(255)); }

//--------------------------------------------------------------
// Single flood fill operation.
Mat floodFillSingle(const Mat &im, Point seed) {
  Mat pass = full255(im.size());
  Mat im_inv;
  bitwise_not(im, im_inv);
  Mat mask;
  copyMakeBorder(im_inv, mask, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
  Rect ccomp;
  floodFill(pass, mask, seed, Scalar(0), &ccomp, Scalar(0), Scalar(0), 4);
  return pass;
}

//--------------------------------------------------------------
// Perform a single trapped-ball fill operation.
// Note: This uses OpenCV floodFill and morphology operations.
Mat trappedBallFillSingle(const Mat &image, Point seed, int radius) {
  Mat ball = getBallStructuringElement(radius);

  // Create two temporary images filled with 255.
  Mat pass1 = full255(image.size());
  Mat pass2 = full255(image.size());

  // Invert image
  Mat im_inv;
  bitwise_not(image, im_inv);

  // Floodfill on pass1
  Mat mask1;
  copyMakeBorder(im_inv, mask1, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
  Rect ccomp;
  // floodFill expects a mask that is 2 pixels larger than image.
  floodFill(pass1, mask1, seed, Scalar(0), &ccomp, Scalar(0), Scalar(0), 4);

  // Dilate pass1 to disconnect fill areas between gaps.
  dilate(pass1, pass1, ball, Point(-1, -1), 1);

  // Prepare mask2 and floodfill on pass2.
  Mat mask2;
  copyMakeBorder(pass1, mask2, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
  floodFill(pass2, mask2, seed, Scalar(0), &ccomp, Scalar(0), Scalar(0), 4);

  // Erode pass2 for leak-proof fill.
  erode(pass2, pass2, ball, Point(-1, -1), 1);

  return pass2;
}

//-------------------------------------------------------------------
// flood_fill_multi: Perform multiple flood fill operations until all valid
// areas are filled. This function mimics the Python version and returns a
// vector of fills, where each fill is represented as a vector of cv::Point (the
// points that were filled).
vector<pair<vector<int>, vector<int>>> floodFillMulti(const Mat &image,
                                                      int max_iter = 20000) {

  // Copy the image into unfill_area.
  Mat unfill_area = image.clone();
  vector<pair<vector<int>, vector<int>>> filled_area;

  for (int iter = 0; iter < max_iter; iter++) {
    // Get unfilled points (pixels equal to 255) using ultraFastWhere.
    auto indices = get_unfilled_point(unfill_area);
    // if length of indices is 0, break
    if (indices.size() == 0) {
      break;
    }
    // index first and second element of indices[0]
    int seed_col = indices[0].x;
    int seed_row = indices[0].y;
    Point seed(seed_col, seed_row); // Note: (x, y) = (col, row)

    // Perform a single flood fill.
    Mat fill = floodFillSingle(unfill_area, seed);

    // Update the unfilled area by bitwise AND of unfill_area and fill.
    Mat new_unfill;
    bitwise_and(unfill_area, fill, new_unfill);
    unfill_area = new_unfill;

    // Get the points that were filled (pixels equal to 0) using ultraFastWhere.
    auto fill_indices = ultraFastWhere(fill, 0);
    filled_area.push_back(fill_indices);
  }

  return filled_area;
}

//--------------------------------------------------------------
// Mark fill areas on an image (set the fill points to 0)
void markFill(Mat &image, const vector<vector<Point>> &fills) {
  for (const auto &fill : fills) {
    for (const auto &pt : fill) {
      image.at<uchar>(pt) = 0;
    }
  }
}

//--------------------------------------------------------------
// Build a fill map where each fill is given an id (lines are id 0)
Mat buildFillMap(const Mat &image, const vector<vector<Point>> &fills) {
  Mat result = Mat::zeros(image.size(), CV_32SC1);
  // Each fill is assigned an id (starting from 1)
  for (size_t i = 0; i < fills.size(); i++) {
    for (const auto &pt : fills[i]) {
      result.at<int>(pt) = static_cast<int>(i) + 1;
    }
  }
  return result;
}

//--------------------------------------------------------------
// Visualize fill map with random colors.
Mat showFillMap(const Mat &fillmap) {
  // Get maximum id.
  double minVal, maxVal;
  minMaxLoc(fillmap, &minVal, &maxVal);
  int numColors = static_cast<int>(maxVal) + 1;

  // Generate random colors.
  vector<Vec3b> colors(numColors);
  colors[0] = Vec3b(0, 0, 0); // id 0 is black.
  std::mt19937 rng(12345);
  for (int i = 1; i < numColors; i++) {
    colors[i] = Vec3b(rng() % 256, rng() % 256, rng() % 256);
  }

  Mat colorMap(fillmap.size(), CV_8UC3, Scalar(0, 0, 0));
  for (int y = 0; y < fillmap.rows; y++) {
    for (int x = 0; x < fillmap.cols; x++) {
      int id = fillmap.at<int>(y, x);
      colorMap.at<Vec3b>(y, x) = colors[id];
    }
  }
  return colorMap;
}

//--------------------------------------------------------------
// Get bounding rectangle for a set of points.
Rect getBoundingRect(const vector<Point> &pts) { return boundingRect(pts); }

//--------------------------------------------------------------
// Get a valid bounding rect including a border of radius r.
Rect getBorderBoundingRect(int h, int w, const Point &p1, const Point &p2,
                           int r) {
  int x1 = max(p1.x - r, 0);
  int y1 = max(p1.y - r, 0);
  int x2 = min(p2.x + r + 1, w);
  int y2 = min(p2.y + r + 1, h);
  return Rect(Point(x1, y1), Point(x2, y2));
}

//--------------------------------------------------------------
// Get border points of a fill area and its approximated polygon.
// This function mimics Python's get_border_point.
// - 'pts' are the points belonging to the fill (global coordinates).
// - 'rect' is the bounding rectangle of those points (as returned by
// getBoundingRect).
// - max_height and max_width are the image dimensions.
pair<vector<Point>, vector<Point>> getBorderPoint(const vector<Point> &pts,
                                                  const Rect &rect,
                                                  int max_height,
                                                  int max_width) {
  // In the Python version, rect is a 4-tuple (min_x, min_y, max_x, max_y).
  // Here, our rect is a cv::Rect with (x, y, width, height).
  // We can compute the top-left and bottom-right as follows:
  Point tl = rect.tl();
  Point br = Point(rect.x + rect.width - 1, rect.y + rect.height - 1);

  // Get a local bounding rectangle (with a border of 2 pixels)
  Rect borderRect = getBorderBoundingRect(max_height, max_width, tl, br, 2);

  // Create a local fill image (initially zeros) of size (borderRect.height,
  // borderRect.width)
  Mat fillImg = Mat::zeros(borderRect.height, borderRect.width, CV_8UC1);

  // Move each point into the local coordinate system and set that pixel to 255.
  for (const auto &pt : pts) {
    Point localPt = pt - borderRect.tl();
    if (localPt.x >= 0 && localPt.x < fillImg.cols && localPt.y >= 0 &&
        localPt.y < fillImg.rows)
      fillImg.at<uchar>(localPt) = 255;
  }

  // Find external contours in the local fill image.
  vector<vector<Point>> contours;
  findContours(fillImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  vector<Point> approxShape;
  if (!contours.empty()) {
    approxPolyDP(contours[0], approxShape, 0.02 * arcLength(contours[0], true),
                 true);
  }

  // Use a cross-shaped structuring element for dilation.
  Mat cross = getStructuringElement(MORPH_CROSS, Size(3, 3));
  Mat dilated;
  dilate(fillImg, dilated, cross, Point(-1, -1), 1);
  Mat borderMask = dilated - fillImg;

  // Get the border pixel locations from the border mask.
  vector<Point> borderPts;
  findNonZero(borderMask, borderPts);

  // Transform border points back to global coordinates.
  for (auto &pt : borderPts) {
    pt += borderRect.tl();
  }

  return make_pair(borderPts, approxShape);
}

//--------------------------------------------------------------
// Merge fill areas iteratively.
// Equivalent to Python's merge_fill.
// fillmap: an image (CV_32SC1) where each pixel is a fill id.
// max_iter: maximum number of iterations.
Mat mergeFill(const Mat &fillmap, int max_iter = 10) {
  int max_height = fillmap.rows;
  int max_width = fillmap.cols;
  Mat result = fillmap.clone();

  for (int iter = 0; iter < max_iter; iter++) {
    // Set result at pixels where fillmap is 0 to 0.
    // (This ensures that the lines (id 0) remain unchanged.)
    for (int y = 0; y < fillmap.rows; y++) {
      for (int x = 0; x < fillmap.cols; x++) {
        if (fillmap.at<int>(y, x) == 0)
          result.at<int>(y, x) = 0;
      }
    }

    // Get the unique fill ids in the result.
    set<int> fillIDs;
    for (int y = 0; y < result.rows; y++) {
      const int *row = result.ptr<int>(y);
      for (int x = 0; x < result.cols; x++) {
        fillIDs.insert(row[x]);
      }
    }

    // Build a vector of FillInfo for each fill id.
    vector<FillInfo> fills;
    for (int id : fillIDs) {
      vector<Point> pts;
      for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
          if (result.at<int>(y, x) == id)
            pts.push_back(Point(x, y));
        }
      }
      if (!pts.empty()) {
        FillInfo info;
        info.id = id;
        info.pts = pts;
        info.area = static_cast<int>(pts.size());
        info.rect = getBoundingRect(pts);
        fills.push_back(info);
      }
    }

    // Process each fill (ignore id == 0, which represents lines).
    for (const auto &f : fills) {
      if (f.id == 0)
        continue;

      // Get border points and the approximated contour shape.
      auto borderInfo = getBorderPoint(f.pts, f.rect, max_height, max_width);
      vector<Point> border_points = borderInfo.first;
      vector<Point> approx_shape = borderInfo.second;

      // Get the fill id values on the border.
      vector<int> pixel_ids;
      for (const auto &pt : border_points) {
        int val = result.at<int>(pt);
        if (val != 0)
          pixel_ids.push_back(val);
      }
      // Get unique non-zero ids from the border pixels.
      set<int> idSet(pixel_ids.begin(), pixel_ids.end());
      vector<int> ids(idSet.begin(), idSet.end());

      int new_id = f.id;
      if (ids.empty()) {
        // If no neighboring fill ids, and the area is small, set new_id to 0.
        if (f.area < 5)
          new_id = 0;
      } else {
        // Otherwise, choose the first neighbor (you might choose the one with
        // largest contact).
        new_id = *ids.begin();
      }

      // Apply various conditions to update the fill:
      if (approx_shape.size() == 1 || f.area == 1) {
        for (const auto &pt : f.pts)
          result.at<int>(pt) = new_id;
      }
      if ((approx_shape.size() == 2 || approx_shape.size() == 3 ||
           approx_shape.size() == 4 || approx_shape.size() == 5) &&
          f.area < 500) {
        for (const auto &pt : f.pts)
          result.at<int>(pt) = new_id;
      }
      if (f.area < 250 && ids.size() == 1) {
        for (const auto &pt : f.pts)
          result.at<int>(pt) = new_id;
      }
      if (f.area < 50) {
        for (const auto &pt : f.pts)
          result.at<int>(pt) = new_id;
      }
    }

    // Check if the unique fill ids remain unchanged.
    set<int> newIDs;
    for (int y = 0; y < result.rows; y++) {
      const int *row = result.ptr<int>(y);
      for (int x = 0; x < result.cols; x++) {
        newIDs.insert(row[x]);
      }
    }
    if (fillIDs.size() == newIDs.size())
      break;
  }
  return result;
}

// Pybind11 module definition
PYBIND11_MODULE(trappedballcpp, m) {
  m.doc() = "Trapped-ball fill algorithm"; // optional module docstring
  // define fast_where with ultraFastWhere
  // convert the NumPy array to cv::Mat
  m.def(
      "fast_where",
      [](py::array_t<uint8_t> array, int target) {
        // Convert numpy array to cv::Mat
        cv::Mat image = numpy_uint8_to_cv_mat(array);
        auto result = ultraFastWhere(image, target);
        // Convert the result to a Python tuple of two NumPy arrays.
        py::array_t<int> rows(result.first.size());
        py::array_t<int> cols(result.second.size());
        py::buffer_info rows_buf = rows.request();
        py::buffer_info cols_buf = cols.request();
        memcpy(rows_buf.ptr, result.first.data(),
               result.first.size() * sizeof(int));
        memcpy(cols_buf.ptr, result.second.data(),
               result.second.size() * sizeof(int));
        return py::make_tuple(rows, cols);
      },
      py::arg("image"), py::arg("target"), "Ultra-fast where");
  m.def("build_fill_map", &buildFillMap, "Build fill map");
  m.def("show_fill_map", &showFillMap, "Show fill map");
  // Wrap flood_fill_multi with a lambda that converts the NumPy array 32int to
  // cv::Mat.
  /* m.def("trapped_ball_fill_multi",
      [](py::array_t<uint8_t> image, int radius, const string &method, int
     max_iter) {
          // Convert numpy array to cv::Mat
          cv::Mat im = numpy_uint8_to_cv_mat(image);
          auto result = trappedBallFillMulti(im, radius, method, max_iter);
          // Convert the result to a Python list of list of tuples.
          // Create a Python list to hold the result.
          py::list py_result;
          int len = result.size();
          for (int i = 0; i < len; i++) {
              const auto& fill = result[i];
              // Convert the fill to a tuple of two NumPy arrays.
              py::array_t<int> rows(fill.first.size());
              py::array_t<int> cols(fill.second.size());
              py::buffer_info rows_buf = rows.request();
              py::buffer_info cols_buf = cols.request();
              memcpy(rows_buf.ptr, fill.first.data(), fill.first.size() *
     sizeof(int)); memcpy(cols_buf.ptr, fill.second.data(), fill.second.size() *
     sizeof(int)); py_result.append(py::make_tuple(rows, cols));
          }

          return py_result;
      },
      py::arg("image"), py::arg("radius") = 4, py::arg("method") = "mean",
     py::arg("max_iter") = 1000, "Trapped-ball fill algorithm"); */
  m.def(
      "flood_fill_multi",
      [](py::array_t<uint8_t> array, int max_iter) {
        // Convert numpy array to cv::Mat
        cv::Mat image = numpy_uint8_to_cv_mat(array);
        auto result = floodFillMulti(image, max_iter);

        // Create a Python list to hold the result.
        py::list py_result;
        int len = result.size();
        for (int i = 0; i < len; i++) {
          const auto &fill = result[i];
          // Convert the fill to a tuple of two NumPy arrays.
          py::array_t<int> rows(fill.first.size());
          py::array_t<int> cols(fill.second.size());
          py::buffer_info rows_buf = rows.request();
          py::buffer_info cols_buf = cols.request();
          memcpy(rows_buf.ptr, fill.first.data(),
                 fill.first.size() * sizeof(int));
          memcpy(cols_buf.ptr, fill.second.data(),
                 fill.second.size() * sizeof(int));
          py_result.append(py::make_tuple(rows, cols));
        }

        return py_result;
      },
      py::arg("image"), py::arg("max_iter") = 20000, "Flood-fill algorithm");

  m.def(
      "merge_fill",
      [](py::array_t<int> fillmap, int max_iter) {
        cv::Mat image = numpy_int32_to_cv_mat(fillmap);
        auto result = mergeFill(image, max_iter);
        // Convert the result to a Python list of list of tuples.
        py::list py_result;
        for (int y = 0; y < result.rows; y++) {
          py::list py_row;
          for (int x = 0; x < result.cols; x++) {
            py_row.append(result.at<int>(y, x));
          }
          py_result.append(py_row);
        }
        return py_result;
      },
      py::arg("fillmap"), py::arg("max_iter") = 10, "Merge fill areas");
}
