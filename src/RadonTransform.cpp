#include "RadonTransform.h"

RadonTransform :: RadonTransform(int c)
{
	sqrt2 = sqrt(2.0);
	halfsqrt2 = 0.5 * sqrt2;
	halfsqrt2divsin22 = 0.5 * sqrt2 / sinf(PI/8);
	doublesqrt2 = 2 * sqrt2;
	cos45 = sqrt2/2;
	sin45 = cos45;
	cos22 = cosf(PI/8);
	sin22 = sinf(PI/8);
	sec22 = 1/cos22;
	cos67 = cosf(3*PI/8);
	sin67 = sinf(3*PI/8);
	count = c;
}

void RadonTransform :: transform(Mat& image)
{
	prepare(image);
	for ( int i = 0; i < image.rows; ++i)
	{
		for(int j = 0; j < image.cols; ++j)
		{
			if(image.at<uchar>(i, j) == 0)
			{
				project0degrees(i, j);
				project22degrees(i, j);
				project45degrees(i, j);
				project67degrees(i, j);
				project90degrees(i, j);
				project112degrees(i, j);
				project135degrees(i, j);
				project157degrees(i, j);
			}
		}
	}
	formatsbTotals();
}

void RadonTransform :: prepare(Mat& image)
{
	numPixels = image.rows * image.cols;
	calculateCentroid(image);
	calculateRadius(image);
	subBeams = 2 * ceil( (10 * radius + halfBeams + 1) / beams ) + 1;
	totalSubBeams = beams * subBeams;
	halfSubBeams = (totalSubBeams - 1) / 2;
	for (int i=0; i<8; i++)
	{
		for ( int j=0; j < totalSubBeams; j++ ) sbTotals[i][j] = 0 ;
		for ( int j=0; j < beams; j++ ) bTotals[i][j] = 0 ;
	}

	beamWidth = radius / halfSubBeams;
}

void RadonTransform :: calculateCentroid(Mat& image)
{
	long long sumX = 0, sumY = 0;

	for (int i = 0; i < image.rows; ++i)
	{
		for(int j = 0; j < image.cols; ++j)
		{
			sumX += i;
			sumY += j;
		}
	}
	centroid[0] = sumX / numPixels;
	centroid[1] = sumY / numPixels;
}

void RadonTransform :: calculateRadius(Mat& image)
{
	radius = 0;

	for(int i = 0; i < image.rows; ++i)
	{
		for(int j = 0; j < image.cols; ++j)
		{
			float len = pow((double)(i - centroid[0]), 2) + pow((double)(j - centroid[1]), 2);
			radius = max(radius, len);
		}
	}
	radius = sqrt((float)radius) + halfsqrt2;
}

void RadonTransform :: project0degrees(int x, int y)
{
	float c = (x - centroid[0]) / beamWidth;

	int l = ceil( c - 0.5/beamWidth ) + halfSubBeams;
	int r = floor( c + 0.5/beamWidth ) + halfSubBeams;

	for (int i = l; i <= r; ++i)
	{
		sbTotals[0][i] += 1;
	}
}

void RadonTransform :: project22degrees(int x, int y)
{
	float cl = ((y - centroid[1] - 0.5) * cos22 + (x - centroid[0] - 0.5 ) * sin22) / beamWidth;
	float cr = ((y - centroid[1] + 0.5) * cos22 + (x - centroid[0] + 0.5 ) * sin22) / beamWidth;

	float c = (cl + cr) / 2;

	int l = ceil( cl ) + halfSubBeams;
	int r = floor( cr ) + halfSubBeams;

	float incl = floor(0.293 * (cr - cl + 1) );

	for (int i = l; i <= l+incl-1; ++i)
	{
		sbTotals[1][i] += halfsqrt2divsin22 - doublesqrt2*abs( i - halfSubBeams - c) * beamWidth;
	}

	for (int i = l+incl; i <= r-incl; ++i)
	{
		sbTotals[1][i] += sec22;
	}

	for (int i = r-incl+1; i <= r; ++i)
	{
		sbTotals[1][i] += halfsqrt2divsin22 - doublesqrt2*abs( i - halfSubBeams - c) * beamWidth;
	}
}

void RadonTransform :: project45degrees(int x, int y)
{
	float cl = ((y - centroid[1] - 0.5) * cos45 + (x - centroid[0] - 0.5 ) * sin45 ) / beamWidth;
	float cr = ((y - centroid[1] + 0.5) * cos45 + (x - centroid[0] + 0.5 ) * sin45 ) / beamWidth;

	float c = (cl + cr) / 2;

	int l = ceil(cl) + halfSubBeams;
	int r = floor(cr) + halfSubBeams;

	for (int i = l; i <= r; ++i)
	{
		sbTotals[2][i] += sqrt2 - 2*abs( i - halfSubBeams - c) * beamWidth;
	}
}

void RadonTransform :: project67degrees(int x, int y)
{
	float cl = ((y - centroid[1] - 0.5) * cos67 + (x - centroid[0] - 0.5 ) * sin67 ) / beamWidth;
	float cr = ((y - centroid[1] + 0.5) * cos67 + (x - centroid[0] + 0.5 ) * sin67 ) / beamWidth;

	float c = (cl + cr) / 2;

	int l = ceil(cl) + halfSubBeams;
	int r = floor(cr) + halfSubBeams;

	float incl = floor(0.293 * (cr - cl + 1) );

	for(int i = l; i <= l+incl-1; ++i)
	{
		sbTotals[3][i] += halfsqrt2divsin22 - doublesqrt2*abs( i - halfSubBeams - c) * beamWidth;
	}

	for(int i = l+incl; i <= r-incl; ++i)
	{
		sbTotals[3][i] += sec22;
	}

	for(int i = r-incl+1; i <= r; ++i)
	{
		sbTotals[3][i] += halfsqrt2divsin22 - doublesqrt2*abs( i - halfSubBeams - c) * beamWidth;
	}
}

void RadonTransform :: project90degrees(int x, int y)
{
	float c = (y - centroid[1]) / beamWidth;

	int l = ceil( c - 0.5/beamWidth ) + halfSubBeams;
	int r = floor( c + 0.5/beamWidth ) + halfSubBeams;

	for (int i = l; i <= r; ++i)
	{
		sbTotals[4][i] += 1;
	}
}

void RadonTransform :: project112degrees(int x, int y)
{
	float cl = ( -(y - centroid[1] + 0.5) * cos22 + (x - centroid[0] - 0.5 ) * sin22 ) / beamWidth;
	float cr = ( -(y - centroid[1] - 0.5) * cos22 + (x - centroid[0] + 0.5 ) * sin22 ) / beamWidth;

	float c = (cl + cr) / 2;

	int l = ceil( cl ) + halfSubBeams;
	int r = floor( cr ) + halfSubBeams;

	float incl = floor(0.293 * (cr - cl + 1) );

	for ( int i=l; i<=l+incl-1; i++ )
	{
		sbTotals[5][i] += halfsqrt2divsin22 - doublesqrt2*abs( i - halfSubBeams - c) * beamWidth;
	}

	for ( int i=l+incl; i<=r-incl; i++ )
	{
		sbTotals[5][i] += sec22;
	}

	for ( int i=r-incl+1; i<=r; i++ )
	{
		sbTotals[5][i] += halfsqrt2divsin22 - doublesqrt2*abs( i - halfSubBeams - c) * beamWidth;
	}
}

void RadonTransform :: project135degrees(int x, int y)
{
	float cl = ( -(y - centroid[1] + 0.5) * cos45 + (x - centroid[0] - 0.5 ) * sin45 ) / beamWidth;
	float cr = ( -(y - centroid[1] - 0.5) * cos45 + (x - centroid[0] + 0.5 ) * sin45 ) / beamWidth;

	float c = (cl + cr) / 2;

	int l = ceil( cl ) + halfSubBeams;
	int r = floor( cr ) + halfSubBeams;

	for ( int i=l; i<=r; i++ )
	{
		sbTotals[6][i] += sqrt2 - 2*abs( i - halfSubBeams - c) * beamWidth;
	}
}

void RadonTransform :: project157degrees(int x, int y)
{
	//projected left & right corners of pixel
	float cl = ( -(y - centroid[1] + 0.5) * cos67 + (x - centroid[0] - 0.5 ) * sin67 ) / beamWidth;
	float cr = ( -(y - centroid[1] - 0.5) * cos67 + (x - centroid[0] + 0.5 ) * sin67 ) / beamWidth;

	//average of cl and cr (center point of pixel)
	float c = (cl + cr) / 2;

	//rounded values - left and right sub-beams affected
	int l = ceil( cl ) + halfSubBeams;
	int r = floor( cr ) + halfSubBeams;

	//length of incline
	float incl = floor(0.293 * (cr - cl + 1) );

	//add contributions to sub-beams
	for ( int i=l; i<=l+incl-1; i++ )
	{
		sbTotals[7][i] += halfsqrt2divsin22 - doublesqrt2*abs( i - halfSubBeams - c) * beamWidth;
	}

	for ( int i=l+incl; i<=r-incl; i++ )
	{
		sbTotals[7][i] += sec22;
	}

	for ( int i=r-incl+1; i<=r; i++ )
	{
		sbTotals[7][i] += halfsqrt2divsin22 - doublesqrt2*abs( i - halfSubBeams - c) * beamWidth;
	}
}

void RadonTransform :: formatsbTotals()
{
	float scalingFactor = beamWidth / numPixels;

	//for each projection
	for ( int h=0; h < 8; h++ )
	{
		//beam counter
		int beam = 0;

		//for each beam grouping
		for ( int i=0; i < totalSubBeams; i += subBeams )
		{
			float sum = 0;

			//add all the beams together in grouping
			for ( int j=i; j < i + subBeams; j++ ) sum += sbTotals[h][j];

			//store in final result with scaling
			bTotals[h][beam] = sum * scalingFactor;
			beam++;
		}
	}
	ofstream outputFile;
	outputFile.open("/home/vignesh/second_review/Radon/FYP_RADON", std::ios_base::app);
	//debug: print sum of beams
	outputFile << "----------" << count << "----------\n";
	for ( int h=0; h < 8; h++ )
	{
		float sum = 0;

		for (int i=0; i<beams; i++)
		{
			sum += bTotals[h][i];
		}

		outputFile << sum << endl;
	}
	outputFile << "----------------------------------------------------------------------------\n";
	outputFile.close();
}
