#ifndef RADONTRANSFORM_H_
#define RADONTRANSFORM_H_

#include<iostream>
#include<stdio.h>
#include<vector>
#include<fstream>
#include<algorithm>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>

#define PI 3.14159265358979323846

using namespace std;
using namespace cv;

class RadonTransform
{
private:
	float cos45, sin45, cos22, sin22, sec22, cos67, sin67, sqrt2, halfsqrt2, halfsqrt2divsin22, doublesqrt2;
	static const int halfBeams = 25, beams = 2*halfBeams + 1;
	int numPixels, subBeams, totalSubBeams, halfSubBeams, centroid[2];
	float radius, beamWidth;
	float sbTotals[8][100000];
	float bTotals[8][beams];
	int count;
	void prepare(Mat& image);
	void calculateCentroid(Mat& image);
	void calculateRadius(Mat& image);
	void project0degrees(int x, int y);
	void project22degrees(int x, int y);
	void project45degrees(int x, int y);
	void project67degrees(int x, int y);
	void project90degrees(int x, int y);
	void project112degrees(int x, int y);
	void project135degrees(int x, int y);
	void project157degrees(int x, int y);
	void formatsbTotals();

public:
	RadonTransform(int c);
	void transform(Mat& image);
};

#endif
