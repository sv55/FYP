#include "Features.h"
#include "RadonTransform.h"
#include "SWTMarshaller.h"
#include <sstream>
#include <fstream>

using namespace cv;
using namespace std;

int sBase[] = {0, 2, 0, 1, 0, 1, 0, 4, 2, 2, 2, 1, 1, 4, 0, 1, 1, 1, 0, 2, 1, 0, 0, 0, 0, 1, 1, 3, 2, 2, 0, 2, 1, 2, 1, 2, 3, 2, 0, 1, 1, 1, 2, 3, 0, 0, 0, 0};
int hBase[] = {3, 2, 1, 2, 2, 1, 1, 0, 0, 0, 2, 0, 4, 1, 2, 1, 1, 1, 2, 0, 0, 2, 1, 1, 4, 2, 2, 2, 2, 2, 3, 5, 1, 1, 2, 2, 2, 2, 1, 3, 2, 0, 2, 2, 2, 1, 3, 1};

void fillBoundingBoxes(Mat& img, Rect rect, bool flag)
{
	cv::Scalar color;
	if(flag)
	{
		color = cv::Scalar(0, 0, 255);
	}
	else
	{
		color = cv::Scalar(0, 255, 0);
	}
	cv::rectangle(img, rect, color, 3);
	/*for(int i = rect.y; i < rect.y + rect.height; ++i)
	{
		for(int j = rect.x; j < rect.x + rect.width; ++j)
		{
			if(flag)
				img.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			else
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
		}
	}*/
}

bool checkForMatch(int a, int b)
{
	for(int i = 0; i < 48; ++i)
	{
		if(sBase[i] == a && hBase[i] == b)
			return true;
		else
			continue;
	}
	return false;
}

void fillBoundingBox(Mat &img, Rect rect)
{
	for(int i = rect.y; i < rect.y + rect.height; ++i)
	{
		for(int j = rect.x; j < rect.x + rect.width; ++j)
		{
			img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		}
	}
}

void makeBlack(Mat &img)
{
	for(int i = 0; i < img.rows; ++i)
	{
		for(int j = 0; j < img.cols; ++j)
		{
			img.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
		}
	}
}

bool isInBoundingBox(Rect rect, int x, int y)
{
	if(rect.x <= x && rect.x + rect.width >= x && rect.y <= y && rect.y + rect.height >= y)
		return true;
	else
		return false;
}

std::vector<std::pair<double, double> > doStrokeWidth(vector<Rect> & boundingBoxes, Mat & src)
{
        Mat resizeImg = src.clone();
        int baseSize = 800;
        PreProcessor::resize(resizeImg, baseSize);

        std::vector< std::pair<double, double> > strokewidths;
        for(int i = 0; i < boundingBoxes.size(); ++i)
        {
                cv::Mat boxMat(src, boundingBoxes[i]);
                std::pair<double, double> ans = SWTMarshaller::performSWT(boxMat, boxMat);
                strokewidths.push_back(ans);
        }
        return strokewidths;
}

int main(int argc, char *argv[])
{
 	if(argc != 3)
        {
                cerr<<"Usage: ./hws [InputImage] [OutputImagePath]\n";
                return 1;
        }
	string path(argv[1]);
	string oPath(argv[2]);
	cout << "Loading image...\n";
	Mat image = imread(path, CV_LOAD_IMAGE_COLOR);
        Mat origImgClone = image.clone();

	imwrite(oPath + "input_image.jpg", image);
	Features *features;
	cout << "Making some initial adjustments on the image...\n";
	features = new Features(image, oPath);
	cout << "Finding the bounding boxes...\n";
	features->findBoundingBoxes();

        std::cout<<"Finding the stroke widths\n";
        vector<Rect> boundingBoxes = features->getBoundingBoxes();
        std::vector<std::pair<double, double> > strokewidths = doStrokeWidth(boundingBoxes, origImgClone);
        cout << "Storing the average stroke width for each bounding box\n";
	for(int i = 0; i < boundingBoxes.size(); ++i)
	{
		features->strokeWidthPositive.push_back(strokewidths[i].first);
		features->strokeWidthNegative.push_back(strokewidths[i].second);
	}

	cout << "Finding Density of bounding boxes...\n";
	features->findDensity();
	Mat densImage = image.clone();
	cout << "Performing classification using density...\n";
	for(int i = 0; i < features->boundingBoxesDensity.size(); ++i)
	{
		if(features->boundingBoxesDensity[i] < 0.2)
			fillBoundingBoxes(densImage, features->boundingBoxes[i], true);
		else
			fillBoundingBoxes(densImage,features->boundingBoxes[i], false);
	}
	imwrite(oPath + "FYP_CLASSIFICATION_DENSITY.jpg", densImage);
	cout << "Finding the Vertical Edge Contribution of the bounding boxes...\n";
	features->findVerticalEdgeContribution();
	Mat vertImage = image.clone();
	cout << "Performing Classification using vertical edge contribution...\n";
	for(int i = 0; i < features->verticalEdgeContribution.size(); ++i)
	{
		if(features->verticalEdgeContribution[i] < 0.05)
			fillBoundingBoxes(vertImage, features->boundingBoxes[i], true);
		else
			fillBoundingBoxes(vertImage,features->boundingBoxes[i], false);
	}
	imwrite(oPath + "FYP_CLASSIFICATION_VERTICAL_EDGES.jpg", vertImage);
	cout << "Finding black pixel density in bottom lines of bounding boxes...\n";
	features->findBlackPixelDensityInBottomLine();
	Mat botImage = image.clone();
	cout << "Performing Classification using the bottom lines pixel density...\n";
	for(int i = 0; i < features->bottomLinesBlackPixelDensity.size(); ++i)
	{
		if(features->bottomLinesBlackPixelDensity[i] < 0.6)
			fillBoundingBoxes(botImage, features->boundingBoxes[i], true);
		else
			fillBoundingBoxes(botImage,features->boundingBoxes[i], false);
	}
	imwrite(oPath + "FYP_CLASSIFICATION_BOTTOM_LINES.jpg", botImage);
	cout << "Finding black pixel density in top lines of bounding boxes...\n";
	features->findBlackPixelDensityInTopLine();
	Mat topImage = image.clone();
	cout << "Performing Classification using the top lines pixel density...\n";
	for(int i = 0; i < features->topLinesBlackPixelDensity.size(); ++i)
	{
		if(features->topLinesBlackPixelDensity[i] < 0.6)
			fillBoundingBoxes(topImage, features->boundingBoxes[i], true);
		else
			fillBoundingBoxes(topImage,features->boundingBoxes[i], false);
	}
	imwrite(oPath + "FYP_CLASSIFICATION_TOP_LINES.jpg", topImage);
	Mat charImage = image.clone();
	Mat bImage = image.clone();
	cout << "Finding the character spacing in bounding boxes...\n";
	features->findCharacterSpacing();
	for(int i = 0; i < features->characterSpacing.size(); ++i)
	{
		if(features->characterSpacing[i] > 0.15)
			fillBoundingBoxes(charImage, features->boundingBoxes[i], false);
		else
			fillBoundingBoxes(charImage, features->boundingBoxes[i], true);
		fillBoundingBoxes(bImage, features->boundingBoxes[i], true);
	}
	imwrite(oPath + "FYP_CLASSIFICATION_CHARACTER_SPACING.jpg", charImage);
	imwrite(oPath + "FYP_BOUNDING_BOXES.jpg", bImage);
	cout << "Finding the pixel distribution in bounding boxes...\n";
	features->findPixelDistribution();
	Mat pixImage = image.clone();
	cout << "Performing Classification using the pixel distribution...\n";
	for(int i = 0; i < features->pixelDistribution.size(); ++i)
	{
		if(features->pixelDistribution[i] > 2.0)
			fillBoundingBoxes(pixImage, features->boundingBoxes[i], true);
		else
			fillBoundingBoxes(pixImage,features->boundingBoxes[i], false);
	}
	imwrite(oPath + "FYP_CLASSIFICATION_PIXEL_DISTRIBUTION.jpg", pixImage);
	cout << "Finding the stems and horizontal lines...\n";
	Mat rot(Size(image.rows, image.cols), CV_8U);
	Mat org(Size(image.cols, image.rows), CV_8U);
	vector< vector<Rect> > groupBB;
	groupBB.resize(features->boundingBoxes.size());
	for(int i = 0; i < features->profileImage.rows; ++i)
	{
		for(int j = 0; j < features->profileImage.cols; ++j)
		{
			rot.at<uchar>(j, i) = features->profileImage.at<uchar>(i, j);
		}
	}
	cvtColor(rot, rot, COLOR_GRAY2BGR);
	Features *f = new Features(rot, oPath);
	f->findVerticalEdgeContribution();
	for(int i = 0; i < f->perwittOutputImage.rows; ++i)
	{
		for(int j = 0; j < f->perwittOutputImage.cols; ++j)
		{
			org.at<uchar>(j, i) = f->perwittOutputImage.at<uchar>(i, j);
		}
	}
	features->findCharacterBoundingBoxes();
	Mat a = features->perwittOutputImage.clone();
	vector< vector<Point> > contours;
	findContours(a, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	vector<Rect> bb;
	reverse(contours.begin(), contours.end());
	Mat img = image.clone();
	makeBlack(img);
	for(int k = 0; k < (int)contours.size(); ++k)
	{
		Rect rect = boundingRect(contours[k]);
		if((rect.height * rect.width) > 20)
		{
			bb.push_back(rect);
		}
	}
	vector< vector<Point> > contours2;
	findContours(org, contours2, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	vector<Rect> aa;
	reverse(contours2.begin(), contours2.end());
	Mat im = image.clone();
	makeBlack(im);
	for(int k = 0; k < (int)contours2.size(); ++k)
	{
		Rect rect = boundingRect(contours2[k]);
		if((rect.height * rect.width) > 20)
		{
			aa.push_back(rect);
		}
	}
	vector<int> count;
	count.resize(features->characterBoundingBoxes.size());
	for(int j = 0; j < features->characterBoundingBoxes.size(); ++j)
	{
		count.push_back(0);
	}
	for(int v = 0; v < bb.size(); ++v)
	{
		for(int w = 0; w < features->characterBoundingBoxes.size(); ++w)
		{
			if(isInBoundingBox(features->characterBoundingBoxes[w], bb[v].x, bb[v].y))
			{
				++count[w];
				break;
			}
		}
	}
	ofstream outputFile;
	vector<int> cont;
	for(int j = 0; j < features->characterBoundingBoxes.size(); ++j)
	{
		cont.push_back(0);
	}
	for(int v = 0; v < aa.size(); ++v)
	{
		for(int w = 0; w < features->characterBoundingBoxes.size(); ++w)
		{
			if(isInBoundingBox(features->characterBoundingBoxes[w], aa[v].x, aa[v].y))
			{
				++cont[w];
				break;
			}
		}
	}
	for(int v = 0; v < features->characterBoundingBoxes.size(); ++v)
	{
		for(int w = 0; w < features->boundingBoxes.size(); ++w)
		{
			if(isInBoundingBox(features->boundingBoxes[w], features->characterBoundingBoxes[v].x, features->characterBoundingBoxes[v].y))
			{
				groupBB[w].push_back(features->characterBoundingBoxes[v]);
				break;
			}
		}
	}
	for(int i = 0; i < bb.size(); ++i)
		fillBoundingBox(img, bb[i]);
	for(int i = 0; i < aa.size(); ++i)
		fillBoundingBox(im, aa[i]);
	imwrite(oPath + "FYP_STEMS.jpg", img);
	imwrite(oPath + "FYP_HORS.jpg", im);
	int j = 0;
	cout << "Performing classification using stems and horizontal lines...\n";
	for(int z = 0; z < groupBB.size(); ++z)
	{
		int t = 0;
		int f = 0;
		for(int x = 0; x < groupBB[z].size(); ++x)
		{
			if(checkForMatch(count[j], cont[j++]))
				++t;
			else
				++f;
		}
		double p = (double)t / (double)(t + f);
		features->stemsHors.push_back(p);
		if(p > 0.5)
		{
			fillBoundingBoxes(image, features->boundingBoxes[z], false);
		}
		else
		{
			fillBoundingBoxes(image, features->boundingBoxes[z], true);
		}
	}
	features->writeVector();
	outputFile.open((oPath + "features.xml").c_str());
	outputFile << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>" << endl;
	outputFile << "<?xml-stylesheet type=\"text/xsl\" href=\"features.xsl\"?>" << endl;
	outputFile << "<features>\n";
	for(int u = 0; u < features->boundingBoxes.size(); ++u)
	{
		outputFile << "\t<word>\n";
		outputFile << "\t\t<density>" << features->boundingBoxesDensity[u] << "</density>\n";
		outputFile << "\t\t<vertical>" << features->verticalEdgeContribution[u] << "</vertical>\n";
		outputFile << "\t\t<top>" << features->topLinesBlackPixelDensity[u] << "</top>\n";
		outputFile << "\t\t<bottom>" << features->bottomLinesBlackPixelDensity[u] << "</bottom>\n";
		outputFile << "\t\t<stemshors>" << features->stemsHors[u] << "</stemshors>\n";
		outputFile << "\t\t<pixel>" << features->pixelDistribution[u] << "</pixel>\n";
		outputFile << "\t\t<char>" << features->characterSpacing[u] << "</char>\n";
		outputFile << "\t\t<spos>" << features->strokeWidthPositive[u] << "</spos>\n";
		outputFile << "\t\t<sneg>" << features->strokeWidthNegative[u] << "</sneg>\n";
		outputFile << "\t</word>\n";
	}
	outputFile << "</features>";
	outputFile.close();
	imwrite(oPath + "FYP_CLASSIFICATION_STHORS.jpg", image);
	system("gnuplot /home/vignesh/density.gnu");
	system("gnuplot /home/vignesh/top.gnu");
	system("gnuplot /home/vignesh/bottom.gnu");
	system("gnuplot /home/vignesh/vertical.gnu");
	system("gnuplot /home/vignesh/pixel.gnu");
	system("gnuplot /home/vignesh/char.gnu");
	system("gnuplot /home/vignesh/stem.gnu");
	cout << "Program terminated successfully...\n";
	return 0;
}
