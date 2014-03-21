#ifndef STROKE_WIDTH_TRANSFORM_H_
#define STROKE_WIDTH_TRANSFORM_H_

#include "GenericHelpers.h"
#include "PreProcessing.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "vector"
#include "utility"
#include "cstdlib"
#include "cmath"
#include "sstream"
#include "utility"
#include "stack"
#include "algorithm"
#include "sstream"

struct connectedComponent
{
        std::vector<cv::Point> strokes;
};
class SWTransform
{
        public:
                SWTransform(cv::Mat & srcColor, int direction, double thresholdRatio);
                void calculateStrokeWidth(cv::Mat & edgeImg);
                void findStrokeConnectedComponents();
                void varianceFilter();
                void findBoundingBoxes();
                void boundingBoxHeuristics();
                void findPredominantComponent();
                std::vector<cv::Rect> getRoi() const
                {
                        return roi;
                }
                cv::Mat getStrokeWidthImage() const { return strokeW; }
        private:
                static void nextCoOrdinate(float curRow, float curCol, float & nextRow, float & nextCol, double angle, int step, int direction);
                void doStrokeWidth(cv::Mat & edgeImg, bool isMedian);

                int direction;
                double thresholdRatio;
                int maxStrokeWidth;
                int initialSW;

                cv::Mat origColor;
                cv::Mat src; //grayscale source
                cv::Mat theta;
                cv::Mat strokeW;
                cv::Mat labelledImg;

                std::vector<cv::Rect> roi; //Region of interest
                std::vector<cv::Rect> discardedRoi;
                std::vector<cv::Point> startPoints;
                std::vector< std::vector<cv::Point> > rays;
                std::vector<connectedComponent> ccs;
};

#endif
