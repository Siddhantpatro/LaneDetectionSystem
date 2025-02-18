#include "..\inc\LaneDetector.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <random>

constexpr double M_PI{3.14159265358979323846};

// Calling the constructor and destructor
LaneDetector::LaneDetector() {}
LaneDetector::~LaneDetector() {}


cv::Mat LaneDetector::detectLanes(cv::Mat& frame)
{
    cv::Mat gray = applyGrayscale(frame);
    cv::Mat blurred = applyGaussianBlur(gray);
    cv::Mat edges = applyEdgeDetection(blurred);

    cv::imshow("edge Detection", edges);

    cv::Mat roi = regionOfInterest(edges);

    cv::imshow("Region of Interest", roi);

    std::vector<cv::Vec4i> lines = ransacLineFitting(roi);
    drawLines(frame, lines);
    return frame;
}

// Grayscale conversion from RGB using the conversion formula
cv::Mat LaneDetector::applyGrayscale(const cv::Mat& frame) {
    cv::Mat gray(frame.rows, frame.cols, CV_8UC1);
    for (int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            cv::Vec3b color = frame.at<cv::Vec3b>(y, x);
            uchar blue  = color[0];
            uchar green = color[1];
            uchar red   = color[2];

            // Grayscale conversion formula
            uchar grayValue = static_cast<uchar>(0.299 * red + 0.587 * green + 0.114 * blue);
            gray.at<uchar>(y, x) = grayValue;
        }
    }
    return gray;
}

// Gaussian blur (5x5 Kernel)
cv::Mat LaneDetector::applyGaussianBlur(const cv::Mat& grayFrame) {
    int kernelSize = 5;   // Kernel or matrix size
    double sigma   = 1.5;  // Standard Deviation
    int halfSize = kernelSize / 2;
    cv::Mat blurred(grayFrame.size(), grayFrame.type());

    // Create gaussian kernel
    double kernel[5][5];
    double sum = 0.0;
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            double value = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[i + halfSize][j + halfSize] = value;
            sum += value;
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    // Apply Gaussian blur
    // Loop through each pixel of the image and for each pixel apply the kernel so as to get a blurred output
    for (int y = halfSize; y < grayFrame.rows - halfSize; y++) {
        for (int x = halfSize; x < grayFrame.cols - halfSize; x++) {
            double sum = 0.0;
            for (int ky = -halfSize; ky <= halfSize; ky++) {
                for (int kx = -halfSize; kx <= halfSize; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

                    sum += grayFrame.at<uchar>(ny, nx) * kernel[ky + halfSize][kx + halfSize];
                }
            }
            blurred.at<uchar> (y, x) = static_cast<uchar> (sum);
        }
    }
    return blurred;
}

// Manual edge detection (Canny method)
cv::Mat LaneDetector::applyEdgeDetection(const cv::Mat& blurredFrame) {
    cv::Mat edges = blurredFrame.clone();

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 1; y < blurredFrame.rows - 1; y++) {
        for (int x = 1; x < blurredFrame.cols - 1; x++) {
            int grad_x = 0, grad_y = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = blurredFrame.at<uchar>(y + ky, x + kx);
                    grad_x += pixel * Gx[ky + 1][kx + 1];
                    grad_y += pixel * Gy[ky + 1][kx + 1];
                }
            }
            int magnitude = static_cast<int>(sqrt(grad_x * grad_x + grad_y * grad_y));
            edges.at<uchar>(y, x) = magnitude > 100 ? 255 : 0;
        }
    }
    return edges;
}

// Region of Interest (Masking) 
cv::Mat LaneDetector::regionOfInterest(const cv::Mat& edges) {
    
    // Step 1: Create a black mask
    cv::Mat mask = cv::Mat::zeros(edges.size(), edges.type());

    // cv::Mat mask = cv::Mat::zeros(720, 1280, CV_8UC1);
    // cv::Point p1(100, 100), p2(400, 300);

    // Step 2: Define the four points of the trapezoid
    cv::Point points[4] = {
        cv::Point(200, edges.rows), 
        cv::Point(1100, edges.rows),
        cv::Point(600, 400), 
        cv::Point(500, 400)
    };

    // Step 3: Function to draw a line using Bresenham's Line Algorithm
    auto drawLine = [&](cv::Mat& img, cv::Point p1, cv::Point p2) {
        int x1 = p1.x, y1 = p1.y;
        int x2 = p2.x, y2 = p2.y;

        int dx = abs(x2 - x1), dy = abs(y2 - y1);
         
        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;

        int err = dx - dy;

        while (true) {
            img.at<uchar>(y1, x1) = 255; // Draw pixel
            if (x1 == x2 && y1 == y2) break;
            int e2 = 2 * err;

            if (e2 > -dy) {
                err -= dy; 
                x1 += sx;
            }

            if (e2 < dx) {
                err += dx;
                y1 += sy;
            }
        }
    };

    // Step 4: Draw the four edges of the trapezoid
    drawLine(mask, points[0], points[1]);
    drawLine(mask, points[1], points[2]);
    drawLine(mask, points[2], points[3]);
    drawLine(mask, points[3], points[0]);

    // drawLine(mask, p1, p2);

    // Step 5: Fill the polygon manually
    for (int y = 0; y < mask.rows; y++) {
        bool inside = false;
        for (int x = 0; x < mask.cols; x++) {
            if (mask.at<uchar>(y, x) == 255) {
                inside = !inside;
            }
            if (inside) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }

    if (edges.empty()) {
        std::cout << "Error: Edges image is empty!" << std::endl;
    }

    if (mask.empty()) {
        std::cout << "Error: Mask image is empty!" << std::endl;
    }

    std::cout << "Edges size: " << edges.size() << std::endl;
    std::cout << "Mask Size: " << mask.size() << std::endl;

    std::cout << "Edges size: " << edges.type() << std::endl;
    std::cout << "Mask type: " << mask.type() << std::endl;

    // Step 6: Perform bitwise "AND"
    cv::Mat maskedEdges = cv::Mat::zeros(edges.size(), edges.type());
    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {
            if (mask.at<uchar>(y, x) == 255 && edges.at<uchar>(y, x) > 0) {
                maskedEdges.at<uchar>(y, x) = edges.at<uchar>(y, x);
            }else {
                maskedEdges.at<uchar>(y, x) = 0;
            }
        }
    }

    // Step 7: return the masked images
    return maskedEdges;
}   

// RANSAC line fitting method
std::vector<cv::Vec4i> LaneDetector::ransacLineFitting(const cv::Mat& roi) {
    std::vector<cv::Vec4i> lines;
    std::vector<cv::Point> edgePoints;

    for (int y = 0; y < roi.rows; y++) {
        for (int x = 0; x < roi.cols; x++) {
            if (roi.at<uchar>(y, x) > 0) {
                edgePoints.push_back(cv::Point(x, y));
            }
        }
    }

    // Parameters for RANSAC
    const int MAX_ITERATIONS = 100;
    const double DISTANCE_THRESHOLD = 2.0;
    const int MIN_INLIERS = 50;
    int maxInliers = 0;
    cv::Vec4i bestLine;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        // Randomly select two points
        std::uniform_int_distribution<> dis(0, edgePoints.size() - 1);
        cv::Point p1 = edgePoints[dis(gen)];
        cv::Point p2 = edgePoints[dis(gen)];

        if (p1 == p2) continue;

        // Line equation: ax + by + c = 0
        double a = p1.y - p2.y;
        double b = p2.x - p1.x;
        double c = p1.x * p2.y - p2.x * p1.y;

        // Count inliers
        int inliers = 0;
        for (const auto& p : edgePoints) {
            double dist = std::fabs(a * p.x + b * p.y + c) / std::sqrt(a * a + b * b);
            if (dist < DISTANCE_THRESHOLD) {
                inliers++;
            }
        }

        if (inliers > maxInliers) {
            maxInliers = inliers;
            bestLine = cv::Vec4i(
                static_cast<int>(p1.x), static_cast<int>(p1.y),
                static_cast<int>(p2.x), static_cast<int>(p2.y)
            );
        }
    }

    // If enough inliers are found, add the line to the result
    if (maxInliers > MIN_INLIERS) {
        lines.push_back(bestLine);
    }

    return lines;
}

// Line drawing
void LaneDetector::drawLines(cv::Mat& frame, const std::vector<cv::Vec4i>& lines) {
    bool laneDeviation = false;
    int frameCenter = frame.cols / 2;
    int laneCenter = 0;
    int detectedLines = 0;
    for (const auto& line : lines) {
        cv::line(frame, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 5);
        laneCenter += (line[0] + line[2]) / 2;
        detectedLines++;
    }
    if (detectedLines > 0) {
        laneCenter /= detectedLines;
        int deviation = laneCenter - frameCenter;
        if (std::abs(deviation) > 50) {
            laneDeviation = true;
            std::string alertText = "Lane Departure!";
            cv::putText(frame, alertText, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
    }
}

