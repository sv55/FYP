#ifndef SWTMarshaller_H_
#define SWTMarshaller_H_

#include "StrokeWidthTransform.h"
#include "PreProcessing.h"
#include "GenericHelpers.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "vector"
#include "utility"
#include "set"

class SWTMarshaller
{
        public:
                static std::pair<double, double> performSWT(cv::Mat & src, cv::Mat & gaborImg);
};

#endif
