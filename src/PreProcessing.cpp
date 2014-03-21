#include "PreProcessing.h"

/*!
 * Initial conversions performed on the image are
 * 1. Color to Grayscale
 * 2. Contrast Stretching 
 */
void PreProcessor::toGrayScale(cv::Mat & image)
{
	cv::Mat src;

	//Convert image to grayscale
	cv::cvtColor(image, src, CV_RGB2GRAY);

        image = src;
}

void PreProcessor::contrastStretch(cv::Mat & src)
{
        cv::Mat original = src.clone();
        assert(src.channels() == 1);
	//Find the minimum and maximum pixel values in the image
	double maxVal, minVal;
	cv::minMaxLoc(src, &minVal, &maxVal);

	//Change the image to 8bit, 1channel representation by contrast stretching
	src.convertTo(original, CV_8UC1, 255.0/(maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
}

/*!
 * Resize by cubic interpolation
 * Right now, resizing only the columns
 */
void PreProcessor::resize(cv::Mat & source, int & baseColSize)
{
        assert(source.cols > baseColSize);

	int rows = source.rows;
	int cols = source.cols;
	double aspectRatio = (double)rows/(double)cols;
        
        /*!
         * To maintain the aspect ratio
         */
        int rem = cols - baseColSize;
        cols -= rem;
        rows = ((double)cols * aspectRatio);
	cv::resize(source, source, cv::Size(cols, rows), 0, 0, cv::INTER_CUBIC);
}
