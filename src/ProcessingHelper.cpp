#include "ProcessingHelper.h"

/*! 
 * Supporting function (not a member function of class)
 * Performs contrast stretch on a single image
 * Reason for not being a member of class: Its advisable to define template functions where they are declared(StackOverflow)
 */
template <typename T> void contrastStretch(cv::Mat& src, cv::Mat& dest)
{
	int rows = src.rows;
	int cols = src.cols;
	double minValue, maxValue;
	cv::minMaxLoc(src, &minValue, &maxValue);

	for(int i = 0; i < rows; ++i)
	{
		for(int j = 0; j < cols; ++j)
		{
			cv::Scalar s = src.at<T>(i, j);
			float pixValue = std::abs(s[0]);
			float destValue = ((pixValue - minValue) / (maxValue - minValue)) * 255.0; //TODO: check if the hard-coded value of 255.0 is correct
			dest.at<uchar>(i,j) = (uchar)(destValue);
		}
	}
}
/*!
 * Fill the "rect" portion of "src" with "fillValue"
 */
void ProcessingHelper::fill(cv::Mat & dest, cv::Rect & rect, uchar fillValue)
{

        int top = rect.y;
        int bottom = rect.y + rect.height;
        int left = rect.x;
        int right = rect.x + rect.width;

        assert(top >= 0 && bottom <= dest.rows && left >= 0 && right <= dest.cols);

        for(int i = top; i <= bottom; ++i)
        {
                for(int j = left; j <= right; ++j)
                {
                        dest.at<uchar>(i, j) = fillValue;
                }
        }
}

/*!
 * Perform an "AND" of two images
 * img1 and img2 should have the same depth and size
 * destult is stored in dest
 */
void ProcessingHelper::performAND(cv::Mat & img1, cv::Mat & img2, cv::Mat & dest)
{
        assert(img1.size() == img2.size());
        assert(img1.depth() == img2.depth());

        for(int i = 0; i < img1.rows; ++i)
        {
                for(int j = 0; j < img2.cols; ++j)
                {
                        dest.at<uchar>(i, j) = std::min(img1.at<uchar>(i, j), img2.at<uchar>(i, j));
                }
        }
}

/*!
 * Super-imposes all the images in images into a single image
 * Every cv::Mat in images should be 8bit 1 channel
 * Dest - 8 bit 1 channel image
 */
void ProcessingHelper::superImpose(std::vector<cv::Mat> & images, cv::Mat & dest)
{
        assert(images.size() > 0);
        assert(dest.depth() == CV_8U && dest.channels() == 1);

	int rows = images[0].rows;
	int cols = images[0].cols;
	int numImages = images.size();
        
        cv::Mat siImg(cv::Size(cols, rows), CV_16UC1, cv::Scalar(0)); //super-imposed Image

	//Sum up the pixel values of all images
	//unsigned short is used to avoid overflows
	for(int i = 0; i < numImages; ++i)
	{
		for(int j = 0; j < rows; ++j)
		{
			for(int k = 0; k < cols; ++k) 
                        {
				siImg.at<unsigned short>(j, k) += (unsigned short)images[i].at<uchar>(j,k);
			}
		}
 	}

	//Average the sum
	for(int i = 0; i < rows; ++i) 
        {
		for(int j = 0; j < cols; ++j) 
                {
			siImg.at<unsigned short>(i,j) = (int)(siImg.at<unsigned short>(i,j)/numImages);
		}
	}

	//Bring the unsgined short to 8bit - uchar using a contrast stretch
	contrastStretch<unsigned short>(siImg, dest);
}



/*!
 * Contrast stretch and then threshold a vector of images
 */
void ProcessingHelper::applyContrastStretch(std::vector<cv::Mat> & images, std::vector<cv::Mat> & dest)
{
	for(int i = 0; i < (int)images.size();++i)
	{
		cv::Mat contImg(cv::Size(images[i].cols, images[i].rows), CV_8UC1, cv::Scalar(0));
		contrastStretch <float> (images[i], contImg);
		cv::threshold(contImg, contImg, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		dest.push_back(contImg);
	}
}

/*!
 * Applies Sobel edge detector
 */
void ProcessingHelper::sobel(cv::Mat & src, cv::Mat & grad_x, cv::Mat & grad_y) 
{
        //depth of both gradient images are 16S to avoid overflow
        //Sobel(source, dest, depth, n_th-orderivative-x, n_th-orderivative-y, kernel_size, scale, delta, border_type)
        cv::Sobel(src, grad_x, CV_32F, 1, 0, 3); 
        cv::Sobel(src, grad_y, CV_32F, 0, 1, 3);

        /*cv::Mat x, y;
        cv::convertScaleAbs(grad_x, x);
        cv::convertScaleAbs(grad_y, y);
        grad_x = x.clone();
        grad_y = y.clone();*/
}

void ProcessingHelper::removeDotNoise(cv::Mat & src) 
{
 	std::vector< std::vector< cv::Point> > contours;
	std::vector< cv::Vec4i > hierarchy;

	//Find all the contours
	cv::findContours(src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	//Fill all the contours
	cv::Mat contourFilled = cv::Mat::zeros( src.size(), CV_8UC1 );

	for( int i = 0; i < (int)contours.size(); i++ )
	{
                cv::Rect rect = cv::boundingRect(contours[i]);
                if(isDotNoise(rect)) 
                {
                       fill(src, rect, 0);
                }
        }
}

bool ProcessingHelper::isWidthBad(cv::Rect & rect) 
{
        if(rect.width < 5) return true;
        return false;
}
bool ProcessingHelper::isHeightBad(cv::Rect & rect) 
{
        if(rect.width < 5) return true;
        return false;
}
bool ProcessingHelper::isDotNoise(cv::Rect & rect) 
{
        if(isWidthBad(rect) || isHeightBad(rect)) return true;
        
        return false;
}

/*!
 * Find lines in a given image
 * the lines will be drawn on dest
 * dest will have Scalar bg as the background
 * dest will have lines drawn in Scalar fillValue
 */
void ProcessingHelper::findLines(cv::Mat & src, cv::Mat & dest, cv::Scalar bg, cv::Scalar fillValue, int thickness)
{
        cv::vector<cv::Vec4i> lines;
        cv::Mat lineImg(src.size(), CV_8UC1, bg);

        cv::Mat canny;
        cv::Canny(src, canny, 50, 100, 3); 
        cv::HoughLinesP(canny, lines, 1, CV_PI/180, 50, 80, 3);
        for( size_t i = 0; i < lines.size(); i++ )
        {
                cv::Vec4i l = lines[i];
                cv::line(lineImg, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), fillValue, thickness, CV_AA);
        }
        dest = lineImg.clone();
}
                
void ProcessingHelper::removeLines(cv::Mat & orig, cv::Mat & filterImg, cv::Mat & dest)
{
        dest = filterImg.clone();

        int k = 0;
        cv::Mat temp = orig.clone();
        cv::Mat houghImg;
        findLines(temp, houghImg, cv::Scalar(255), cv::Scalar(0), 2);
        GenericHelpers::Display("lines", houghImg, 1);
        while(cv::countNonZero(houghImg) != (houghImg.rows * houghImg.cols) && k < 10)
        {
                for(int i = 0; i < dest.rows; ++i)
                {
                        for(int j = 0; j < dest.cols; ++j)
                        {
                                if(houghImg.at<uchar>(i, j) == 0)
                                {
                                        dest.at<uchar>(i, j) = 0;
                                }
                        }
                }
                temp = dest.clone();
                findLines(temp, houghImg, cv::Scalar(255), cv::Scalar(0), 2);
                GenericHelpers::Display("lines", houghImg, 1);
                ++k;
        }
}
