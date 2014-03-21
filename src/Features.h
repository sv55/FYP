#ifndef FEATURES_H_
#define FEATURES_H_

#include<iostream>
#include<stdio.h>
#include<vector>
#include<fstream>
#include<algorithm>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "PreProcessing.h"
#include "GaborFilter.h"
#include "GenericHelpers.h"
#include "Processing.h"

using namespace cv;
using namespace std;

class Features
{
private:
	Mat inputImage, dilatedImage;
	vector<double> lineLengthDeviation;
	vector<Rect> lastWords;
	vector<int> bottomLines;
	vector< vector<Point> > contours;
	vector< vector<int> > verticalProfileOutput;
	void dilateImage();
	Rect findLastWord(int& top, int& bottom);
	void findLastWords();
	int findXGradient(int x, int y);
	int findYGradient(int x, int y);
	void findPerwittOutput();
	string oPath;

public:
	vector<Rect> boundingBoxes;
        vector<Rect> getBoundingBoxes() const { return boundingBoxes; }
	vector<Rect> characterBoundingBoxes;
	vector<Rect> hand;
	vector<Rect> machine;
	Mat perwittOutputImage, horizontalOutputImage, verticalOutputImage, horPerwittOutputImage, profileImage, sourceImage;
	vector<double> boundingBoxesDensity;
	vector<double> boundingBoxesHeightDeviation;
	vector<double> verticalEdgeContribution;
	vector<double> bottomLinesBlackPixelDensity;
	vector<double> topLinesBlackPixelDensity;
	vector<double> pixelDistribution;
	vector<double> stemsHors;
	vector<double> characterSpacing;
	vector<double> strokeWidthPositive;
	vector<double> strokeWidthNegative;
	vector< vector<Point> > characterContours;
	Features(Mat image, string oPath);
	void findBoundingBoxes();
	void findCharacterBoundingBoxes();
	void findDensity();
	void findBottomLines();
	void writeVector();
	void findBlackPixelDensityInBottomLine();
	void findBlackPixelDensityInTopLine();
	void findVerticalEdgeContribution();
	void findProjectionProfileForBoundingBoxes();
	void findCharacterSpacing();
	void findPixelDistribution();
	int findCandidate(vector<int> word);
	bool isMajority(vector<int> word);
	bool isIntersecting(Rect a, Rect b);
	void findIntersection();
};
#endif
