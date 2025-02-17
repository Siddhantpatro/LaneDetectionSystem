#include <opencv2/opencv.hpp>
#include "inc/LaneDetector.hpp"

int main()
{
    LaneDetector laneDetector;

    std::string filename = "C:/LaneDepartureSystem/videos/test_video.mp4";

    cv::VideoCapture cap(filename);

    if (!cap.isOpened())
    {
        std::cout << "Error opening video!" << filename << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat output = laneDetector.detectLanes(frame);
        cv::imshow("Lane Detection", output);

        if(cv::waitKey(30) >= 0) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}