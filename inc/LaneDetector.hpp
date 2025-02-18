#ifndef LANE_DETECTOR_HPP
#define LANE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Class for Detection of a lane 
class LaneDetector {
    public:
        // Constructor declaration
        LaneDetector();

        // Destructor declaration
        ~LaneDetector();

        // Function declaration of type cv::Mat
        cv::Mat detectLanes(Mat& frame);
        
    private:
        cv::Mat applyGrayscale(const cv::Mat& frame);                          // apply grayscale to each image frame captured from the video
        cv::Mat applyGaussianBlur(const cv::Mat& gaussianFrame);               // Blur the gray scaled image
        cv::Mat applyEdgeDetection(const cv::Mat& blurredFrame);               // Detect edges from the received blurred image
        cv::Mat regionOfInterest(const cv::Mat& edges);                        // Find the region of interest after receiving the edge detected image
        std::vector<cv::Vec4i> ransacLineFitting(const cv::Mat& maskedEdges);  // RANSAC Transform
        void drawLines(cv::Mat& frame, const std::vector<cv::Vec4i>& lines);   // Draw lines
};

#endif
