#ifndef PROCESSING_H_
#define PROCESSING_H_


#include "ProcessingHelper.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "vector"
#include "math.h"

class Processor
{
private:
	static void checkAndFillRect(cv::Rect & rect, cv::Mat & src, cv::Mat & dest, double frThresh, cv::Mat & final, std::vector<cv::Rect> & contourCache);

public:
	static std::vector<cv::Rect> findAndFillContours(cv::Mat & src);
	static void checkAndFill(cv::Mat & src, double fillRatio, cv::Mat & final, cv::Rect & rect, std::vector<cv::Rect> & contourCache);
        static void checkAverageHeightAndWidth(cv::Mat & src, std::vector<cv::Rect> & contourCache);
};

#endif

