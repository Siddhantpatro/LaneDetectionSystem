#ifndef LANE_DETECTOR_HPP
#define LANE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// Class for Detection of a lane 
class LaneDetector {
    public:
        // Constructor declaration
        LaneDetector();

        // Destructor declaration
        ~LaneDetector();

        // Function declaration of type cv::Mat
        cv::Mat detectLanes(cv::Mat& frame);
        
    private:
        cv::Mat applyGrayscale(const cv::Mat& frame);                             // apply grayscale to each image frame captured from the video
        cv::Mat applyGaussianBlur(const cv::Mat& gaussianFrame);                  // Blur the gray scaled image
        cv::Mat applyEdgeDetection(const cv::Mat& blurredFrame);                  // Detect edges from the received blurred image
        cv::Mat regionOfInterest(const cv::Mat& edges, const cv::Mat& frame);     // Find the region of interest after receiving the edge detected image
        std::vector<cv::Vec4i> applyHoughTransform(const cv::Mat& maskedEdges);   // Hough Transform
        std::vector<cv::Vec4i> trackLanes(const std::vector<cv::Vec4i>& lines);   // Tracks and smooths lane lines over multiple frames
        void drawLines(cv::Mat& frame, const std::vector<cv::Vec4i>& lines);      // Draw lines

        std::vector<cv::Vec4i> previousLines;  // Store previously detected lines for tracking
};

#endif
