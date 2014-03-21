#ifndef PREPROCESSING_H_
#define PREPROCESSING_H_

#include "opencv/cv.h"
#include "opencv/highgui.h"

class PreProcessor
{
        public:
	        static void toGrayScale(cv::Mat & image);
                static void contrastStretch(cv::Mat & image);
        	static void resize(cv::Mat & image, int & basesize);
};
#endif
