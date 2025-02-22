// Header files included
#include "..\inc\LaneDetector.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <random>

// Pi value
constexpr double M_PI{3.14159265358979323846};

// Constructor and Destructor for the LaneDetector class
LaneDetector::LaneDetector() {}
LaneDetector::~LaneDetector() {}

/* 
@brief Detects and tracks lanes in the provided frame using a series of image processing steps.

This function processes the input frame to detect lanes using a series of image processing techniques:

1. **applyGrayscale(frame)**: Converts the input frame into a grayscale image, as lane detection works better on single-channel images, simplifying the processing.
2. **applyGaussianBlur(gray)**: Applies a Gaussian blur to the grayscale image to reduce noise and detail, helping to focus on the larger patterns like lane edges.
3. **applyEdgeDetection(blurred)**: Performs edge detection (likely using Canny or Sobel), which highlights the edges of the lanes. This is a crucial step to distinguish the lanes from the rest of the image.
4. **regionOfInterest(edges)**: Defines a region of interest (ROI) on the edge-detected image to focus on the area where lanes are likely to appear, removing irrelevant parts of the frame (such as the sky or road boundaries).
5. **applyHoughTransform(roi)**: Applies the Hough Transform to detect straight lines in the region of interest. These lines are likely to represent the lanes.
6. **trackLanes(lines)**: Tracks and filters the detected lines to identify the most likely lane markings. This function might apply further logic, such as line grouping or tracking, to ensure the lanes are consistently detected.
7. **drawLines(frame, trackedLines)**: Draws the tracked lane lines onto the original frame to visualize the result.

Finally, the function returns the frame with the detected and tracked lane lines drawn on it.

@param frame Input frame (image) on which lane detection will be performed.
@return cv::Mat The input frame with detected lanes drawn on it.
*/

cv::Mat LaneDetector::detectLanes(cv::Mat& frame)
{
    cv::Mat gray = applyGrayscale(frame);
    cv::Mat blurred = applyGaussianBlur(gray);
    cv::Mat edges = applyEdgeDetection(blurred);
    cv::Mat roi = regionOfInterest(edges);
    std::vector<cv::Vec4i> lines = applyHoughTransform(roi);
    std::vector<cv::Vec4i> trackedLines = trackLanes(lines);
    drawLines(frame, trackedLines);
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
    cv::Mat mask = cv::Mat::zeros(edges.size(), CV_8UC1); // Ensure it's 8-bit single channel

    // Step 2: Define the three points of the region of interest (ROI) polygon

    /*@brief The below code defines a triangular region of interest (ROI).
       For a trapezoidal region, one needs to add a fourth point to the polygon.
       @example 
        Bottom-left corner   (200, edges.rows),
        Bottom-right corner  (1100, edges.rows),    
        Top-right corner     (600, 400),            
        Top-left corner      (500, 400)             

       The current configuration represents a triangle with the following points:
       - Bottom-left corner   (200, edges.rows)
       - Bottom-right corner  (1100, edges.rows)
       - Top corner           (550, 250) 
    */

    cv::Point points[3] = {
        cv::Point(200, edges.rows),     // Bottom-left
        cv::Point(1100, edges.rows),    // Bottom-right
        cv::Point(550, 250),            // Top
    };

    // Step 3: Convert points into an array for cv::fillPoly()
    std::vector<cv::Point> roiPolygon(points, points + 3);
    std::vector<std::vector<cv::Point>> fillCont = {roiPolygon};

    // Step 4: Fill the polygon with white (255) to create the ROI mask
    cv::fillPoly(mask, fillCont, cv::Scalar(255));

    // Debugging checks
    if (edges.empty()) {
        std::cout << "Error: Edges image is empty!" << std::endl;
    }
    if (mask.empty()) {
        std::cout << "Error: Mask image is empty!" << std::endl;
    }

    // Step 5: Apply the mask using bitwise AND
    cv::Mat maskedEdges;
    cv::bitwise_and(edges, mask, maskedEdges);

    return maskedEdges;
}

// Detect lines in the ROI using custom Hough Transform implementation
std::vector<cv::Vec4i> LaneDetector::applyHoughTransform(const cv::Mat& maskedEdges){
    std::vector<cv::Vec4i> lines;

    // Apply Hough Line Transform using openCV built in function
    cv::HoughLinesP(maskedEdges, lines, 1, CV_PI / 180, 50, 50, 150.0);

    return lines;

    /*
    @brief Below is the code for Manual Hough Transform implementation. The result was not as good as the above CV HoughLinesP function.

    int width = maskedEdges.cols;
    int height = maskedEdges.rows;

    int rhoMax = sqrt(width * width + height * height);
    int rhoBins = 2 * rhoMax;  
    int thetaBins = 180;  

    int threshold = 50;

    std::vector<std::vector<int>> accumulator(rhoBins, std::vector<int>(thetaBins, 0));

    // **Step 1: Voting in Accumulator**
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (maskedEdges.at<uchar>(y, x) > 0) {  
                std::cout << maskedEdges.at<uchar>(y, x) << endl;
                for (int theta = 0; theta < thetaBins; theta++) {
                    double thetaRad = theta * CV_PI / 180.0;
                    int rho = cvRound(x * cos(thetaRad) + y * sin(thetaRad)) + rhoMax;
                    if (rho >= 0 && rho < rhoBins) {
                        accumulator[rho][theta]++;
                    }
                }
            }
        }
    }

    // **Step 2: Extract Local Maxima (Strong Peaks)**
    for (int rho = 1; rho < rhoBins - 1; rho++) {
        for (int theta = 1; theta < thetaBins - 1; theta++) {
            int currentValue = accumulator[rho][theta];

            if (currentValue > threshold &&
                currentValue >= accumulator[rho - 1][theta - 1] &&
                currentValue >= accumulator[rho - 1][theta] &&
                currentValue >= accumulator[rho - 1][theta + 1] &&
                currentValue >= accumulator[rho][theta - 1] &&
                currentValue >= accumulator[rho][theta + 1] &&
                currentValue >= accumulator[rho + 1][theta - 1] &&
                currentValue >= accumulator[rho + 1][theta] &&
                currentValue >= accumulator[rho + 1][theta + 1]) {

                // **Step 3: Convert (rho, theta) to Cartesian**
                double thetaRad = theta * CV_PI / 180.0;
                double rhoVal = rho - rhoMax;
                cv::Point pt1, pt2;
                double a = cos(thetaRad);
                double b = sin(thetaRad);
                double x0 = a * rhoVal;
                double y0 = b * rhoVal;
                pt1.x = cvRound(x0 + 1000 * (-b));
                pt1.y = cvRound(y0 + 1000 * (a));
                pt2.x = cvRound(x0 - 1000 * (-b));
                pt2.y = cvRound(y0 - 1000 * (a));

                lines.push_back(cv::Vec4i(pt1.x, pt1.y, pt2.x, pt2.y));
            }
        }
    } 
    return lines;

    Note: This manual implementation is a simple version of the Hough Transform, 
        and the results may not be as robust as OpenCV's built-in HoughLinesP function.
    */
}

// Tracks and smooths lane lines over multiple frames
// Averages the positions of the lines to reduce fitter
std::vector<cv::Vec4i> LaneDetector::trackLanes(const std::vector<cv::Vec4i>& lines) {
    if (lines.empty()) {
        return previousLines;  // Return last known lines if none detected
    }

    std::vector<cv::Vec4i> averagedLines;
    
    // Store lanes for left and right separately
    std::vector<cv::Vec4i> leftLanes, rightLanes;

    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1];
        int x2 = line[2], y2 = line[3];

        double slope = (y2 - y1) / static_cast<double>(x2 - x1 + 1e-6); // Avoid division by zero

        // If slope is negative, it's a left lane; otherwise, it's a right lane
        if (slope < -0.2) {  // Threshold to filter near-horizontal lines
            leftLanes.push_back(line);
        } else if (slope > 0.2) {
            rightLanes.push_back(line);
        }
    }

    // Function to compute averaged lane
    auto averageLane = [](const std::vector<cv::Vec4i>& laneLines) -> cv::Vec4i {
        if (laneLines.empty()) return {0, 0, 0, 0};

        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        for (const auto& l : laneLines) {
            x1 += l[0]; y1 += l[1];
            x2 += l[2]; y2 += l[3];
        }

        int count = laneLines.size();
        return {x1 / count, y1 / count, x2 / count, y2 / count};
    };

    cv::Vec4i leftLane = averageLane(leftLanes);
    cv::Vec4i rightLane = averageLane(rightLanes);

    if (leftLane != cv::Vec4i{0, 0, 0, 0}) averagedLines.push_back(leftLane);
    if (rightLane != cv::Vec4i{0, 0, 0, 0}) averagedLines.push_back(rightLane);

    // Smooth over previous frames
    if (!previousLines.empty() && previousLines.size() == averagedLines.size()) {
        for (size_t i = 0; i < averagedLines.size(); ++i) {
            averagedLines[i][0] = (averagedLines[i][0] + previousLines[i][0]) / 2;
            averagedLines[i][1] = (averagedLines[i][1] + previousLines[i][1]) / 2;
            averagedLines[i][2] = (averagedLines[i][2] + previousLines[i][2]) / 2;
            averagedLines[i][3] = (averagedLines[i][3] + previousLines[i][3]) / 2;
        }
    }

    previousLines = averagedLines;
    return averagedLines;
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

