#include "GenericHelpers.h"

cv::Mat GenericHelpers::readImg(std::string & name)
{
        char *imgName = (char *)name.c_str();
        cv::Mat ret = cv::imread(imgName, CV_LOAD_IMAGE_COLOR);
        return ret;
}

/*
 * Write img to file denoted by name
 * Currently a default jpg extension is added
 * WRITE_TO_FILE specifies whether the image should be displyed on screen or written to file
 * TODO: Check if name already has an extension, if it has, dont add .jpg
 */
void GenericHelpers::Display(std::string name, cv::Mat & img, bool WRITE_TO_FILE = true) 
{
        assert(name.length() > 0);

	if(WRITE_TO_FILE)
	{
		std::string res = "";
		name += ".jpg";
		res += name;
		char * newName = (char *)(res.c_str());
		cv::imwrite(newName, img);
	}
	else
	{
		char * newName = (char *)name.c_str();
		cv::namedWindow(newName);
		cv::imshow(newName, img);
                cv::waitKey(0);
	}
}

/*!
 * Used to write/display of vector of images
 */
void GenericHelpers::displayVector(std::vector<cv::Mat>& images, bool WRITE_TO_FILE)
{
        assert(images.size() > 0);

	for(int i = 0; i < (int)images.size(); ++i)
	{
		std::stringstream ss;
		ss<<"Image "<<i;
		Display(ss.str(), images[i], WRITE_TO_FILE);
	}
}

/*!
 * To check if a given three channel image is in black and white
 * Input should definitely be a three channel image
 * Storing grayscale images in three channels causes the value in all three channels to be equal
 */
bool GenericHelpers::isBlackWhite(cv::Mat & src)
{
        assert(src.channels() == 3);
        for(int i = 0; i < src.rows; ++i)
        {
                for(int j = 0; j < src.cols; ++j)
                {
                        cv::Vec3b rgb = src.at<cv::Vec3b>(i, j);
                        int b = rgb[0], g = rgb[1], r = rgb[2];
                        if(r != g || g != b || r != b) return false;
                }
        }
        return true;
}

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
void GenericHelpers::fill(cv::Mat & dest, cv::Rect & rect, uchar fillValue)
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
void GenericHelpers::performAND(cv::Mat & img1, cv::Mat & img2, cv::Mat & dest)
{
        assert(img1.size() == img2.size());
        assert(img1.depth() == img2.depth());

        dest = img1.clone();
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
void GenericHelpers::superImpose(std::vector<cv::Mat> & images, cv::Mat & dest)
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
void GenericHelpers::applyContrastStretch(std::vector<cv::Mat> & images, std::vector<cv::Mat> & dest)
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
void GenericHelpers::sobel(cv::Mat & src, cv::Mat & grad_x, cv::Mat & grad_y) 
{
        //depth of both gradient images are 16S to avoid overflow
        //Sobel(source, dest, depth, n_th-orderivative-x, n_th-orderivative-y, kernel_size, scale, delta, border_type)
        cv::Sobel(src, grad_x, CV_32FC1, 1, 0, 3);
        cv::Sobel(src, grad_y, CV_32FC1, 0, 1, 3);
}

void GenericHelpers::removeDotNoise(cv::Mat & src) 
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

bool GenericHelpers::isWidthBad(cv::Rect & rect) 
{
        if(rect.width < 5) return true;
        return false;
}
bool GenericHelpers::isHeightBad(cv::Rect & rect) 
{
        if(rect.width < 5) return true;
        return false;
}
bool GenericHelpers::isDotNoise(cv::Rect & rect) 
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
void GenericHelpers::findLines(cv::Mat & src, cv::Mat & dest, cv::Scalar bg, cv::Scalar fillValue, int thickness)
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
                
void GenericHelpers::removeLines(cv::Mat & orig, cv::Mat & filterImg, cv::Mat & dest)
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

void GenericHelpers::cleanDots(cv::Mat & src, cv::Mat & dest)
{
        dest = src.clone();
        for(int i = 0; i < src.rows; ++i)
        {
                for(int j = 0; j < src.cols; ++j)
                {
                        int row = i, col = j;
                        if(!(row - 1 > 0 && row + 1 < src.rows && col - 1 > 0 && col + 1 < src.cols))
                        {
                                continue;
                        }
                        if(src.at<uchar>(i, j) == 0) continue;
                        int count = 0;
                        if(src.at<uchar>(row - 1, col - 1) > 0) count++;
                        if(src.at<uchar>(row - 1, col) > 0) count++;
                        if(src.at<uchar>(row - 1, col + 1) > 0) count++;
                
                        if(src.at<uchar>(row, col - 1) > 0) count++;
                        if(src.at<uchar>(row, col) > 0) count++;
                        if(src.at<uchar>(row, col + 1) > 0) count++;
        
                        if(src.at<uchar>(row + 1, col - 1) > 0) count++;
                        if(src.at<uchar>(row + 1, col) > 0) count++;
                        if(src.at<uchar>(row + 1, col + 1) > 0) count++;
                        if(count < 2) dest.at<uchar>(i, j) = 0;
                }
        }       
        src = dest.clone();
}

void GenericHelpers::fixGaps(cv::Mat & src)
{
        cv::Mat dest = src.clone();
        for(int i = 0; i < src.rows; ++i)
        {
                for(int j = 0; j < src.cols; ++j)
                {
                        int row = i, col = j;
                        if(!(row - 1 > 0 && row + 1 < src.rows && col - 1 > 0 && col + 1 < src.cols))
                        {
                                continue;
                        }
                        if(src.at<uchar>(i, j) != 0) continue;
                        
                        int count = 0;
                        int arr[8];
                        memset(arr, 0, sizeof arr);
                        if((arr[0] = src.at<uchar>(row - 1, col - 1)) > 0) count++;
                        if((arr[1] = src.at<uchar>(row - 1, col)) > 0) count++;
                        if((arr[2] = src.at<uchar>(row - 1, col + 1)) > 0) count++;
                
                        if((arr[3] = src.at<uchar>(row, col + 1)) > 0) count++;
                        if((arr[4] = src.at<uchar>(row + 1, col + 1)) > 0) count++;
                        if((arr[5] = src.at<uchar>(row + 1, col)) > 0) count++;
        
                        if((arr[6] = src.at<uchar>(row + 1, col - 1)) > 0) count++;
                        if((arr[7] = src.at<uchar>(row, col - 1)) > 0) count++;

                        if(count >= 2)
                        {
                                if((arr[0] && arr[4]) || (arr[2] && arr[6]) || (arr[7] && arr[3]) || (arr[1] && arr[5]))
                                {
                                        dest.at<uchar>(i, j) = static_cast<uchar>(255);
                                }
                        }
                }
        }
        src = dest.clone();
}

void GenericHelpers::drawRectOnImg(cv::Mat & src, std::vector<cv::Rect> roi)
{
        cv::Size curSize = src.size();
        cv::Mat colorSrc(curSize, CV_8UC3, cv::Scalar(0, 0, 0));

        for(int i = 0; i < src.rows; ++i)
        {
                for(int j = 0; j < src.cols; ++j)
                {
                        cv::Vec3b color;
                        color[0] = color[1] = color[2] = src.at<uchar>(i, j);
                        colorSrc.at<cv::Vec3b>(i, j) = color;
                }
        }

        std::stringstream ss;
        for(int i = 0; i < roi.size(); ++i)
        {
                ss<<(i + 1);
               
                cv::Scalar color(3, 3, 255);
//                cv::putText(colorSrc, ss.str(), cv::Point(roi[i].x, roi[i].y), cv::FONT_HERSHEY_SIMPLEX, 0.4, color);
                cv::rectangle(colorSrc, roi[i], color);

                ss.str("");
                ss.clear();
        }
        src = colorSrc.clone();
}
