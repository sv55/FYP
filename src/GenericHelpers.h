#ifndef GENERIC_HELPERS_H
#define GENERIC_HELPERS_H

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "vector"
#include "cstring"
#include "sstream"

class GenericHelpers
{
        public:
                static cv::Mat readImg(std::string & name);
                static void Display(std::string name, cv::Mat & img, bool WRITE_TO_FILE); 
                static void displayVector(std::vector<cv::Mat> & images, bool WRITE_TO_FILE); 
                static bool isBlackWhite(cv::Mat & src);                static void fill(cv::Mat & dest, cv::Rect & rect, uchar fillValue);
                static void superImpose(std::vector<cv::Mat> & images, cv::Mat & dest);
                static void applyContrastStretch(std::vector<cv::Mat> & images, std::vector<cv::Mat> & dest);
                static void performAND(cv::Mat & img1, cv::Mat & img2, cv::Mat & dest);
                static void sobel(cv::Mat & src, cv::Mat & grad_x, cv::Mat & grad_y); 
                static void removeDotNoise(cv::Mat & src);
                static void findLines(cv::Mat & src, cv::Mat & dest, cv::Scalar bg, cv::Scalar fillValue, int thickness);
                static void removeLines(cv::Mat & orig, cv::Mat & filterImg, cv::Mat & dest);
                static void cleanDots(cv::Mat & src, cv::Mat & dest);
                static void drawRectOnImg(cv::Mat & src, std::vector<cv::Rect> roi);
                static void fixGaps(cv::Mat & src);
         private:
                static bool isWidthBad(cv::Rect & rect);
                static bool isHeightBad(cv::Rect & rect);
                static bool isDotNoise(cv::Rect & rect);
};

#endif
