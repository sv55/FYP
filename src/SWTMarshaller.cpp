#include "SWTMarshaller.h"

#define DEBUG 0
double getAverageStrokeWidth(cv::Mat & strokeW)
{
        double tot = 0;
        double cnt = 0;
        for(int i = 0; i < strokeW.rows; ++i)
        {
                for(int j = 0; j < strokeW.cols; ++j)
                {
                        if(strokeW.at<float>(i, j) > 0.0)
                        {
                                tot += static_cast<double>(strokeW.at<float>(i, j));
                                cnt += 1.0;
                        }
                }
        }
        if(cnt == 0.0) return 0.0;

        return tot/cnt;
}
std::pair<double, double> SWTMarshaller::performSWT(cv::Mat & src, cv::Mat & gaborImg)
{
        std::pair<double, double> strokeWidths;
        cv::Mat srcColor = src.clone();
        if(src.channels() == 3)
        {
                PreProcessor::toGrayScale(src);
        }
        cv::Mat cannyImg;
        cv::Canny(src, cannyImg, 50, 120);
        if(DEBUG)
        {
                GenericHelpers::Display("ImgCanny", cannyImg, 1);
        }

        //cv::Mat cleanEdgeImg = cv::Mat::zeros(cannyImg.size(), CV_8U); 
        cv::Mat cleanEdgeImg = cannyImg.clone();
        cv::Mat temp;
       
        int direction = -1;
        double thresholdRatio = 3.0;
        SWTransform * marshaller = new SWTransform(srcColor, direction, thresholdRatio);
        marshaller->calculateStrokeWidth(cleanEdgeImg);
        marshaller->findStrokeConnectedComponents();
        if(DEBUG)
        {
                temp = src.clone();
                GenericHelpers::drawRectOnImg(temp, marshaller->getRoi());
                GenericHelpers::Display("BareCC", temp, 1);
                std::cout<<"DEBUG: SWT -----------------------------------------\n";
        }
        marshaller->varianceFilter();
        if(DEBUG)
        {
                temp = src.clone();
                GenericHelpers::drawRectOnImg(temp, marshaller->getRoi());
                GenericHelpers::Display("AfterVarianceFilter", temp, 1);
                std::cout<<"DEBUG: SWT -----------------------------------------\n";
        }
        marshaller->findBoundingBoxes();
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT -----------------------------------------\n";
        }
        marshaller->boundingBoxHeuristics();
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT -----------------------------------------\n";
        }
        marshaller->findPredominantComponent();
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT -----------------------------------------\n";
                temp = src.clone();
                GenericHelpers::drawRectOnImg(temp, marshaller->getRoi());
                GenericHelpers::Display("FinalSWT-1", temp, 1);
        }

        cv::Mat strokeW1 = marshaller->getStrokeWidthImage().clone();
        double dRet1 = getAverageStrokeWidth(strokeW1);
        delete marshaller;
        
        if(DEBUG)
        {
                std::cout<<"\n";
        }
        
        direction = 1;
        marshaller = new SWTransform(srcColor, direction, thresholdRatio);
        marshaller->calculateStrokeWidth(cleanEdgeImg);
        marshaller->findStrokeConnectedComponents();
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT -----------------------------------------\n";
        }
        marshaller->varianceFilter();
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT -----------------------------------------\n";
        }
        marshaller->findBoundingBoxes();
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT -----------------------------------------\n";
        }
        marshaller->boundingBoxHeuristics();
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT -----------------------------------------\n";
        }
        marshaller->findPredominantComponent();
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT -----------------------------------------\n";
                temp = src.clone();
                GenericHelpers::drawRectOnImg(temp, marshaller->getRoi());
                GenericHelpers::Display("FinalSWT1", temp, 1);
        }

        cv::Mat strokeW2 = marshaller->getStrokeWidthImage().clone();
        double dRet2 = getAverageStrokeWidth(strokeW2);
        delete marshaller;

        strokeWidths.first = dRet1;
        strokeWidths.second = dRet2;
        return strokeWidths;
}
