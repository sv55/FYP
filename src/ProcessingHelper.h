#ifndef PROCESSING_HELPER_H
#define PROCESSING_HELPER_H

#include "GenericHelpers.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "vector"

//This class is not meant to be initialized
class ProcessingHelper
{
        public:
                static void fill(cv::Mat & dest, cv::Rect & rect, uchar fillValue);
                static void superImpose(std::vector<cv::Mat> & images, cv::Mat & dest);
                static void applyContrastStretch(std::vector<cv::Mat> & images, std::vector<cv::Mat> & dest);
                static void performAND(cv::Mat & img1, cv::Mat & img2, cv::Mat & dest);
                static void sobel(cv::Mat & src, cv::Mat & grad_x, cv::Mat & grad_y); 
                static void removeDotNoise(cv::Mat & src);
                static void findLines(cv::Mat & src, cv::Mat & dest, cv::Scalar bg, cv::Scalar fillValue, int thickness);
                static void removeLines(cv::Mat & orig, cv::Mat & filterImg, cv::Mat & dest);
        private:
                static bool isWidthBad(cv::Rect & rect);
                static bool isHeightBad(cv::Rect & rect);
                static bool isDotNoise(cv::Rect & rect);
                ProcessingHelper();
};

#endif
