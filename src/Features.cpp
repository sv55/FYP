#include "Features.h"

using namespace std;
using namespace cv;

Features :: Features(Mat image, string oPat)
{
	image.copyTo(inputImage);
	cvtColor(image, sourceImage, COLOR_BGR2GRAY);
	profileImage = sourceImage > 128;
	sourceImage = sourceImage < 128;
	sourceImage.copyTo(dilatedImage);
	bottomLines.push_back(0);
	horizontalOutputImage = profileImage.clone();
	verticalOutputImage = profileImage.clone();
	perwittOutputImage = profileImage.clone();
	horPerwittOutputImage = profileImage.clone();
	oPath = oPat;
}

void Features :: dilateImage()
{
	for(int i = 1; i < sourceImage.rows - 1; ++i) {
		for(int j = 1; j < sourceImage.cols - 1; ++j) {
			if(sourceImage.at<uchar>(i, j - 2) == 255 || sourceImage.at<uchar>(i, j - 1) == 255 || sourceImage.at<uchar>(i, j) == 255 || sourceImage.at<uchar>(i, j + 1) == 255 || sourceImage.at<uchar>(i, j + 2) == 255)
			{
				dilatedImage.at<uchar>(i, j - 3) = 255;
				dilatedImage.at<uchar>(i, j - 2) = 255;
				dilatedImage.at<uchar>(i, j - 1) = 255;
				dilatedImage.at<uchar>(i, j) = 255;
				dilatedImage.at<uchar>(i, j + 1) = 255;
				dilatedImage.at<uchar>(i, j + 2) = 255;
				dilatedImage.at<uchar>(i, j + 3) = 255;
			}

		}
	}
}

void Features :: findBoundingBoxes()
{
	cout << "Dilating the image...\n";
	dilateImage();
	/*!
	 * Running Gabor Filter
	 */
	Mat srcImg = this->dilatedImage.clone();
	//PreProcessor::toGrayScale(srcImg);
	PreProcessor::contrastStretch(srcImg);
	GenericHelpers::Display(oPath + "ImgOriginal", srcImg, 1);

	GaborFilter *gf;
	cout << "Applying Gabor Filter...\n";
	gf = new GaborFilter();
	gf->setDefaultParameters(srcImg);
	gf->createFilterBank();

	std::vector<cv::Mat> gfoImg = gf->applyFilter(srcImg);
	std::vector<cv::Mat> tgfoImg;
	ProcessingHelper::applyContrastStretch(gfoImg, tgfoImg);
	cout << "Superimposing the Gabor Filter Results...\n";
	cv::Mat imgCC(srcImg.size(), CV_8UC1, cv::Scalar(255));
	ProcessingHelper::superImpose(tgfoImg, imgCC);
	cv::threshold(imgCC, imgCC, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	GenericHelpers::Display(oPath + "ImgSuperImpose", imgCC, 1);
	cout << "Finding the connected Components...\n";
	Processor::findAndFillContours(imgCC);
	GenericHelpers::Display(oPath + "ImgConnectedComponents", imgCC, 1);

	findContours(imgCC, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	cout << "Storing the bounding boxes...\n";
	reverse(contours.begin(), contours.end());
	for(int k = 0; k < (int)contours.size(); ++k)
	{
		Rect rect = boundingRect(contours[k]);
		if((rect.height * rect.width) > 250)
			boundingBoxes.push_back(rect);
	}

	/*findContours(dilatedImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	reverse(contours.begin(), contours.end());
	for(int k = 0; k < (int)contours.size(); ++k)
	{
		Rect rect = boundingRect(contours[k]);
		if((rect.height * rect.width) > 500)
			boundingBoxes.push_back(rect);
	}*/
}

void Features :: findCharacterBoundingBoxes()
{
	//cout << "Dilating the image...\n";
	//dilateImage();
	/*!
	 * Running Gabor Filter
	 */
	Mat srcImg = this->sourceImage.clone();
	//PreProcessor::toGrayScale(srcImg);
	PreProcessor::contrastStretch(srcImg);
	GenericHelpers::Display("/home/vignesh/second_review/Gabor/ImgOriginal", srcImg, 1);

	GaborFilter *gf;
	cout << "Applying Gabor Filter...\n";
	gf = new GaborFilter();
	gf->setDefaultParameters(srcImg);
	gf->createFilterBank();

	std::vector<cv::Mat> gfoImg = gf->applyFilter(srcImg);
	std::vector<cv::Mat> tgfoImg;
	ProcessingHelper::applyContrastStretch(gfoImg, tgfoImg);
	cout << "Superimposing the Gabor Filter Results...\n";
	cv::Mat imgCC(srcImg.size(), CV_8UC1, cv::Scalar(255));
	ProcessingHelper::superImpose(tgfoImg, imgCC);
	cv::threshold(imgCC, imgCC, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	GenericHelpers::Display("/home/vignesh/second_review/Gabor/ImgSuperImpose", imgCC, 1);
	cout << "Finding the connected Components...\n";
	Processor::findAndFillContours(imgCC);
	GenericHelpers::Display("/home/vignesh/second_review/Gabor/ImgConnectedComponents", imgCC, 1);

	findContours(imgCC, characterContours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	cout << "Storing the bounding boxes...\n";
	reverse(characterContours.begin(), characterContours.end());
	for(int k = 0; k < (int)characterContours.size(); ++k)
	{
		Rect rect = boundingRect(characterContours[k]);
		if((rect.height * rect.width) > 250)
			characterBoundingBoxes.push_back(rect);
	}

	/*findContours(dilatedImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	reverse(contours.begin(), contours.end());
	for(int k = 0; k < (int)contours.size(); ++k)
	{
		Rect rect = boundingRect(contours[k]);
		if((rect.height * rect.width) > 500)
			boundingBoxes.push_back(rect);
	}*/
}

void Features :: findDensity()
{
	vector<int> boundingBoxesArea;
	int blackPixels = 0;

	for(unsigned int k = 0; k < boundingBoxes.size(); ++k)
	{
		boundingBoxesArea.push_back(boundingBoxes[k].height * boundingBoxes[k].width);
		for(int i = boundingBoxes[k].y; i < boundingBoxes[k].y + boundingBoxes[k].height; ++i)
		{
			for(int j = boundingBoxes[k].x; j < boundingBoxes[k].x + boundingBoxes[k].width; ++j)
			{
				if(profileImage.at<uchar>(i, j) == 0)
					++blackPixels;
			}
		}
		boundingBoxesDensity.push_back((double)blackPixels / (double)boundingBoxesArea[k]);
		blackPixels = 0;
	}
}

/*void Features :: findBottomLines()
{
	vector<int> horizontalProfile;
	horizontalProfile.resize(sourceImage.rows);
	for(unsigned int i = 0; i < horizontalProfile.size(); ++i)
		horizontalProfile[i] = 0;
	for(int i = 0; i < sourceImage.rows; ++i)
	{
		for(int j = 0; j < sourceImage.cols; ++j)
		{
			if(profileImage.at<uchar>(i, j) == 0)
				++horizontalProfile[i];
		}
	}
	for(unsigned int i = 1; i < horizontalProfile.size(); ++i)
	{
		if(horizontalProfile[i] == 0)
		{
			if(horizontalProfile[i - 1] == 0)
			{
				continue;
			}
			else
				bottomLines.push_back(i);
		}
		else
			continue;
	}
}

Rect Features :: findLastWord(int& top, int& bottom)
{
	Rect lastWord;
	int lastWordX = 0;
	for(unsigned int i = 0; i < boundingBoxes.size(); ++i)
	{
		if(boundingBoxes[i].y > top && boundingBoxes[i].y < bottom)
		{
			if((boundingBoxes[i].x + boundingBoxes[i].width) > lastWordX)
			{
				if(boundingBoxes[i].width * boundingBoxes[i].height != 0)
				{
					lastWord = boundingBoxes[i];
					lastWordX = boundingBoxes[i].x + boundingBoxes[i].width;
				}
			}
		}
	}
	return lastWord;
}

void Features :: findLastWords()
{
	for(unsigned int i = 1; i < bottomLines.size(); ++i)
	{
		lastWords.push_back(findLastWord(bottomLines[i - 1], bottomLines[i]));
	}
}

void Features :: findLineLengthDeviation()
{
	findBottomLines();
	findLastWords();
	double totalLineLength = 0;
	double meanLineLength = 0;
	for(unsigned int i = 0; i < lastWords.size(); ++i)
	{
		totalLineLength += (lastWords[i].x + lastWords[i].width);
	}
	meanLineLength = totalLineLength / (double)lastWords.size();
	for(unsigned int i = 0; i < lastWords.size(); ++i)
	{
		if((lastWords[i].width * lastWords[i].height) > 0)
			lineLengthDeviation.push_back(abs(meanLineLength - (lastWords[i].x + lastWords[i].width)));
	}
}*/

void Features :: findBlackPixelDensityInBottomLine()
{
	/*for(unsigned int i = 1; i < bottomLines.size(); ++i)
	{
		int blackPixels = 0;
		for(int j = 0; j < profileImage.cols; ++j) {
			if(profileImage.at<uchar>(bottomLines[i] - 1, j) == 0) {
				++blackPixels;
			}
		}
		bottomLinesBlackPixelDensity.push_back(((double)blackPixels / (double)profileImage.cols) * 100);
	}*/
	for(unsigned int i = 0; i < boundingBoxes.size(); ++i)
	{
		int blackPixels = 0;
		int bottom = boundingBoxes[i].y + boundingBoxes[i].height - 2;
		unsigned int j;
		for(j = boundingBoxes[i].x; j < boundingBoxes[i].x + boundingBoxes[i].width; ++j)
		{
			if(profileImage.at<uchar>(bottom, j) == 0)
			{
				++blackPixels;
			}
		}
		bottomLinesBlackPixelDensity.push_back(((double)blackPixels / (double)j) * 100);
	}
}

void Features :: findBlackPixelDensityInTopLine()
{
	for(unsigned int i = 0; i < boundingBoxes.size(); ++i)
	{
		int blackPixels = 0;
		int top = boundingBoxes[i].y + 1;
		unsigned int j;
		for(j = boundingBoxes[i].x; j < boundingBoxes[i].x + boundingBoxes[i].width; ++j)
		{
			if(profileImage.at<uchar>(top, j) == 0)
			{
				++blackPixels;
			}
		}
		topLinesBlackPixelDensity.push_back(((double)blackPixels / (double)j) * 100);
	}
}

int Features :: findXGradient(int x, int y)
{
	return sourceImage.at<uchar>(y-1, x-1) +
			sourceImage.at<uchar>(y, x-1) +
			sourceImage.at<uchar>(y+1, x-1) -
			sourceImage.at<uchar>(y-1, x+1) -
			sourceImage.at<uchar>(y, x+1) -
			sourceImage.at<uchar>(y+1, x+1);
}

int Features :: findYGradient(int x, int y)
{
	return sourceImage.at<uchar>(y-1, x-1) +
			sourceImage.at<uchar>(y-1, x) +
			sourceImage.at<uchar>(y-1, x+1) -
			sourceImage.at<uchar>(y+1, x-1) -
			sourceImage.at<uchar>(y+1, x) -
			sourceImage.at<uchar>(y+1, x+1);
}

void Features :: findPerwittOutput()
{
	for(int y = 0; y < profileImage.rows; y++)
		for(int x = 0; x < profileImage.cols; x++)
			horizontalOutputImage.at<uchar>(y,x) = 0;
	for(int y = 1; y < profileImage.rows - 1; y++){
		for(int x = 1; x < profileImage.cols - 1; x++){
			int gy = findYGradient(x, y);
			int sum = abs(gy) + abs(gy);
			sum = sum > 255 ? 255:sum;
			sum = sum < 0 ? 0 : sum;
			horizontalOutputImage.at<uchar>(y,x) = sum;
		}
	}

	for(int y = 0; y < profileImage.rows; y++)
		for(int x = 0; x < profileImage.cols; x++)
			verticalOutputImage.at<uchar>(y,x) = 0;
	for(int y = 1; y < profileImage.rows - 1; y++){
		for(int x = 1; x < profileImage.cols - 1; x++){
			int gx = findXGradient(x, y);
			int sum = abs(gx) + abs(gx);
			sum = sum > 255 ? 255:sum;
			sum = sum < 0 ? 0 : sum;
			verticalOutputImage.at<uchar>(y,x) = sum;
		}
	}
	for(int y = 1; y < profileImage.rows - 1; y++){
		for(int x = 1; x < profileImage.cols - 1; x++){
			perwittOutputImage.at<uchar>(y, x) = 0;
		}
	}
	for(int i = 0; i < horizontalOutputImage.rows; ++i)
	{
		for(int j = 0; j < horizontalOutputImage.cols; ++j)
		{
			if(horizontalOutputImage.at<uchar>(i, j) == 255 && verticalOutputImage.at<uchar>(i, j) == 255)
				perwittOutputImage.at<uchar>(i, j) = 0;
			else if(horizontalOutputImage.at<uchar>(i, j) != 255 && verticalOutputImage.at<uchar>(i, j) == 255)
				perwittOutputImage.at<uchar>(i, j) = 255;
			else if(horizontalOutputImage.at<uchar>(i, j) == 255 && verticalOutputImage.at<uchar>(i, j) != 255)
				perwittOutputImage.at<uchar>(i, j) = 0;
		}
	}
	for(int i = 0; i < horizontalOutputImage.rows; ++i)
	{
		for(int j = 0; j < horizontalOutputImage.cols; ++j)
		{
			if(horizontalOutputImage.at<uchar>(i, j) == 255 && verticalOutputImage.at<uchar>(i, j) == 255)
				horPerwittOutputImage.at<uchar>(i, j) = 0;
			else if(horizontalOutputImage.at<uchar>(i, j) != 255 && verticalOutputImage.at<uchar>(i, j) == 255)
				horPerwittOutputImage.at<uchar>(i, j) = 0;
			else if(horizontalOutputImage.at<uchar>(i, j) == 255 && verticalOutputImage.at<uchar>(i, j) != 255)
				horPerwittOutputImage.at<uchar>(i, j) = 255;
		}
	}
}

void Features :: findVerticalEdgeContribution()
{
	findPerwittOutput();
	//imwrite("/home/vignesh/second_review/Vertical/FYP_PERWITT.jpg", perwittOutputImage);
	int whitePixels = 0;
	for(unsigned int k = 0; k < boundingBoxes.size(); ++k)
	{
		double area = boundingBoxes[k].height * boundingBoxes[k].width;
		for(int i = boundingBoxes[k].y; i < boundingBoxes[k].y + boundingBoxes[k].height; ++i)
		{
			for(int j = boundingBoxes[k].x; j < boundingBoxes[k].x + boundingBoxes[k].width; ++j)
			{
				if(perwittOutputImage.at<uchar>(i, j) == 255)
					++whitePixels;
			}
		}
		verticalEdgeContribution.push_back((double)whitePixels / area);
		whitePixels = 0;
	}
}

void Features :: findProjectionProfileForBoundingBoxes()
{
	for(unsigned int k = 0; k < boundingBoxes.size(); ++k)
	{
		Mat currentImage = profileImage(boundingBoxes[k]);
		vector<int> verticalProfile;
		verticalProfile.resize(currentImage.cols);
		for(unsigned int i = 0; i < verticalProfile.size(); ++i)
			verticalProfile[i] = 0;
		for(int i = 0; i < currentImage.rows; ++i)
		{
			for(int j = 0; j < currentImage.cols; ++j)
			{
				if(currentImage.at<uchar>(i, j) == 0)
					++verticalProfile[j];
			}
		}
		verticalProfileOutput.push_back(verticalProfile);
	}
}

int Features :: findCandidate(vector<int> word)
{
	int maj_index = 0, count = 1;
	for(int i = 1; i < word.size(); ++i)
	{
		if(word[maj_index] == word[i])
			count++;
		else
			count--;
		if(count == 0)
		{
			maj_index = i;
			count = 1;
		}
	}
	return word[maj_index];
}

bool Features :: isMajority(vector<int> word)
{
	int count = 0;
	int candidate = findCandidate(word);
	for (int i = 0; i < word.size(); ++i)
		if(word[i] == candidate)
			count++;
	if (count > word.size() / 2)
		return 1;
	else
		return 0;
}

void Features :: findPixelDistribution()
{
	for(int k = 0; k < boundingBoxes.size(); ++k)
	{
		int halfHeight = boundingBoxes[k].height / 2;
		int bound = halfHeight * 0.4;
		int blackPixels = 0;
		double topDensity;
		double bottomDensity;
		double area = boundingBoxes[k].width * halfHeight;
		for(int i = boundingBoxes[k].y + bound; i < boundingBoxes[k].y + halfHeight; ++i)
		{
			for(int j = boundingBoxes[k].x; j < boundingBoxes[k].x + boundingBoxes[k].width; ++j)
			{
				if(profileImage.at<uchar>(i, j) == 0)
					++blackPixels;
			}
		}
		topDensity = (double)blackPixels / area;
		blackPixels = 0;
		for(int i = halfHeight + 1; i < boundingBoxes[k].y + boundingBoxes[k].height - bound; ++i)
		{
			for(int j = boundingBoxes[k].x; j < boundingBoxes[k].x + boundingBoxes[k].width; ++j)
			{
				if(profileImage.at<uchar>(i, j) == 0)
					++blackPixels;
			}
		}
		bottomDensity = (double)blackPixels / area;
		pixelDistribution.push_back(abs(topDensity - bottomDensity));
	}
}

void Features :: findCharacterSpacing()
{
	findProjectionProfileForBoundingBoxes();
	for(unsigned int i = 0; i < verticalProfileOutput.size(); ++i)
	{
		//vector<int> charSpace;
		int count = 0;
		int j;
		for(j = 0; j < verticalProfileOutput[i].size(); ++j)
		{
			if(verticalProfileOutput[i][j] == 0)
				++count;
			/*else if(verticalProfileOutput[i][j] != 0 && j - 1 != -1 && verticalProfileOutput[i][j - 1] == 0)
			{
				charSpace.push_back(count);
				count = 0;
			}*/
		}
		characterSpacing.push_back((double)count / (double)j);
	}
}

bool Features :: isIntersecting(Rect a, Rect b)
{
	return !(a.x + a.width <= b.x || a.y + a.height <= b.y || a.x >= b.x + b.width || a.y >= b.y + b.height);
}

void Features :: findIntersection()
{
	for(int i = 0; i < boundingBoxes.size(); ++i)
	{
		bool flag = true;
		for(int j = i + 1; j < boundingBoxes.size(); ++j)
		{
			if(isIntersecting(boundingBoxes[i], boundingBoxes[j]))
			{
				hand.push_back(boundingBoxes[i]);
				flag = false;
				break;
			}
		}
		if(flag)
			machine.push_back(boundingBoxes[i]);
	}
}

void Features :: writeVector()
{
	//Values written to a local file
	ofstream outputFile;
	outputFile.open((oPath + "FYP_DENSITY").c_str());
	for(unsigned int i = 0; i < boundingBoxesDensity.size(); ++i)
		outputFile << boundingBoxesDensity[i] << endl;
	outputFile.close();
	outputFile.open((oPath + "FYP_TOP_LINE_BLACK_PIXEL_DENSITY").c_str());
	for(unsigned int i = 0; i < topLinesBlackPixelDensity.size(); ++i)
		outputFile << topLinesBlackPixelDensity[i] << endl;
	outputFile.close();
	outputFile.open((oPath + "FYP_BOTTOM_LINE_BLACK_PIXEL_DENSITY").c_str());
	for(unsigned int i = 0; i < bottomLinesBlackPixelDensity.size(); ++i)
		outputFile << bottomLinesBlackPixelDensity[i] << endl;
	outputFile.close();
	outputFile.open((oPath + "FYP_VERTICAL_EDGE_CONTRIBUTION").c_str());
	for(unsigned int i = 0; i < verticalEdgeContribution.size(); ++i)
		outputFile << verticalEdgeContribution[i] << endl;
	outputFile.close();
	outputFile.open((oPath + "FYP_PIXEL_DISTRIBUTION").c_str());
	for(unsigned int i = 0; i < pixelDistribution.size(); ++i)
		outputFile << pixelDistribution[i] << endl;
	outputFile.close();
	outputFile.open((oPath + "FYP_STEMS_HORS").c_str());
	for(unsigned int i = 0; i < stemsHors.size(); ++i)
	{
		if(stemsHors[i] >= 0 && stemsHors[i] <= 1)
			outputFile << stemsHors[i] << endl;
		else
			outputFile << 0 << endl;
	}
	outputFile.close();
	outputFile.open((oPath + "FYP_CHARACTER_SPACING").c_str());
	for(unsigned int i = 0; i < characterSpacing.size(); ++i)
	{
		//for(unsigned int j = 0; j < verticalProfileOutput[i].size(); ++j)
		//{
			outputFile << characterSpacing[i] << "\n";
		//}
		//outputFile << endl;
	}
	outputFile.close();
	outputFile.open((oPath + "FYP_STROKE_POSITIVE").c_str());
	for(unsigned int i = 0; i < strokeWidthPositive.size(); ++i)
		outputFile << strokeWidthPositive[i] << endl;
	outputFile.close();
	outputFile.open((oPath + "FYP_STROKE_NEGATIVE").c_str());
	for(unsigned int i = 0; i < strokeWidthNegative.size(); ++i)
		outputFile << strokeWidthNegative[i] << endl;
	outputFile.close();
}
