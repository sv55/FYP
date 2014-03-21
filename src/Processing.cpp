#include "Processing.h"
#include "GenericHelpers.h"
#include "queue"
#include "utility"
#include "vector"
#include "stack"
#include "cstdlib"
#include "sstream"
#include "sys/wait.h"
#include "sys/types.h"
#include<sstream>

using namespace std;

/*!
 * Find each contour in the image and fill it based on the bounding rectangle of the contour
 */
std::vector< cv::Rect > Processor::findAndFillContours(cv::Mat& src)
{
	//Cache all the contours and return it to main
	std::vector< cv::Rect > contourCache;

	GenericHelpers::Display("Img_BeforeContour", src, 1);
	std::vector< std::vector< cv::Point> > contours;
	std::vector< cv::Vec4i > hierarchy;
	std::cout << "Finding the contours...\n";
	//Find all the contours
	cv::findContours(src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	//Fill all the contours
	std::cout << "Filling the contours...\n";
	cv::Mat contourFilled = cv::Mat::zeros( src.size(), CV_8UC1 );

	for( int i = 0; i < (int)contours.size(); i++ )
	{
		stringstream ss;
		int tt = 100 + (int)(((float)i / (float)contours.size() - 1) * 100);
		ss << tt << "% Completed";
		std::cout << ss.str();
		cv::Scalar s(255);
		cv::Mat fill = cv::Mat::zeros( src.size(), CV_8UC1);
		cv::drawContours( fill, contours, i, s, -1, 8, hierarchy, 0, cv::Point() );
		cv::Rect contourRect = cv::boundingRect(contours[i]);
		checkAndFill(fill, 0.01, contourFilled, contourRect, contourCache);
		for(int m = 0; m < ss.str().size(); ++m)
			cout << '\b';
	}
	src = contourFilled;
	GenericHelpers::Display("Img_AfterContour", src, 1);
	return contourCache;
}

/*!
 * Check each contour for the Heuristic we defined and decide whether to \
 * mark it as text region or not
 */
void Processor::checkAndFill(cv::Mat & src, double fr, cv::Mat & final, cv::Rect & rect, std::vector<cv::Rect> & contourCache)
{
	cv::Mat result = cv::Mat::zeros( src.size(), CV_8UC1 );
	checkAndFillRect(rect, src, result, fr, final, contourCache);
	src = result;
}

/*! 
 * This function checks the fill ratio, aspect ratio and height ratio of each rectangle 
 * and eliminates false text regions
 * frThresh: fill ration threshold
 */
void Processor::checkAndFillRect(cv::Rect & rect, cv::Mat & src, cv::Mat & dest, double frThresh, cv::Mat & final, std::vector<cv::Rect> & contourCache)
{
	int top = rect.y;
	int bottom = rect.y + rect.height;
	int left = rect.x;
	int right = rect.x + rect.width;
	double totalPixels = rect.area();
	double white = 0;

	//Count white pixels in my rectangle
	for(int i = top; i < bottom; ++i)
	{
		for(int j = left; j < right; ++j)
		{
			if((int)src.at<uchar>(i, j) > 0)
			{
				white += 1;
			}
		}
	}
	double fillRatio = white / totalPixels;
	int fillValue = 0;
	//Is fillRatio greater than the threshold?
	if(fillRatio >= frThresh)
	{
		fillValue = 255;
	}

	//A rectangle cannot be this small in width and have text in it
	//if(rect.width <= 12) fillValue = 0;

	//Height threshold
	//TODO: check if this is works on all resolutions 
	double heightRatio = (double) rect.height / (double)src.rows;
	if(heightRatio >= 0.075) fillValue = 0;

	//Area threshold
	//Just 60 pixels in area? Nah can't be a text-region
	if(rect.area() <= 60) fillValue = 0;

	if(rect.width >= (int)((double)src.cols * 0.5)) fillValue = 0;

	if(rect.height <= 15) fillValue = 0;

	//Fill the rect with fillValue
	ProcessingHelper::fill(dest, rect, fillValue);

	//OR(|) this rectangle with the final matrix
	if(fillValue != 0)
	{
		for(int i = top; i < bottom; ++i)
		{
			for(int j = left; j < right; ++j)
			{
				final.at<uchar>(i, j) = std::max((uchar)final.at<uchar>(i, j), (uchar)dest.at<uchar>(i, j));
			}
		}
		contourCache.push_back(rect);
	}
}
void Processor::checkAverageHeightAndWidth(cv::Mat & src, std::vector<cv::Rect> & contourCache)
{
	double avgWidth = 0, avgHeight = 0;
	for(int i = 0; i < contourCache.size(); ++i)
	{
		cv::Rect rect = contourCache[i];
		int top = rect.y;
		int bottom = rect.y + rect.height;
		int left = rect.x;
		int right = rect.x + rect.width;

		avgWidth += (right - left);
		avgHeight += (bottom - top);
	}
	avgWidth /= (double)(contourCache.size());
	avgHeight /= (double)(contourCache.size());

	//85% of the width/height of the whole page
	double maxAllowedWidth = src.cols * 0.85;
	double maxAllowedHeight = src.rows * 0.85;

	for(int i = 0; i < contourCache.size(); ++i)
	{
		cv::Rect rect = contourCache[i];
		int top = rect.y;
		int bottom = rect.y + rect.height;
		int left = rect.x;
		int right = rect.x + rect.width;

		int width = right - left;
		int height = bottom - top;

		cv::Mat temp = src.clone();
		temp = cv::Mat(temp, rect);
		int whitePixels = cv::countNonZero(temp);
		int totPixels = (right - left + 1) * (bottom - top + 1);
		double pixelThreshold = totPixels * 0.70;
		//Remove this rectangle since its width/height is more than the allowed width/height
		if(width >= (int)maxAllowedWidth || height >= (int)maxAllowedHeight || whitePixels < pixelThreshold)
		{
			ProcessingHelper::fill(src, rect, 0);
		}
	}
	GenericHelpers::Display("ImgFinal", src, 1);
}
