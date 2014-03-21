#include "GaborFilter.h"
#include "string"
#include "math.h"
#include "iostream"
#include "cassert"

/*!
 * Set default kernel parameters 
 */
void GaborFilter::setDefaultParameters(cv::Mat & src)
{
	//GABOR FILTERING PARAMETERS
	/*
	 * wavelength - lambda
	 * orientation - theta
	 * bandwidth
	 * phase offsets - psi
	 * aspect ratio - gamma
	 */
	psi = 90.0;
	gamma = 0.0;
	bandwidth = 1.0;
	double psi = 90.0;
	for(int i = 0; i < 2; ++i)
	{
		this->addTheta(i * 90);
	}
	this->addKernelSize(5);
	this->addSigma(0.5);
	this->addLambda(0.3);
	this->setPSI(psi);
	this->setGamma(gamma);
	this->setBandwidth(bandwidth);

}

/*!
 * Function to calculate a Gabor Kernel based on the parameters
 * theta - orientation
 * kSize - size of the kernel
 * bandwidth - bandwidth = sigma / lambda
 * sigma - standard deviation of normal distribution curve
 * kSize - kernel size - odd number
 * lambda - wavelength determining factor of the cosine wave
 * psi - phase offset
 * gamma - aspect ratio of the Gabor curve

 * Actual Formula
 *
 * g(x, y) = exp(-0.5 * pow(xx, 2) + pow(gamma, 2) * pow(yy, 2) * 1/pow(sigma, 2)) * cos(2 * CV_PI * pow(xx, 2) * 1 / lambda + psi)
 *
 */
cv::Mat GaborFilter::getGaborKernel(double theta, int kSize, double sigma, double lambda, double psi, double gamma, double bandwidth)
{

	 int xmax = kSize / 2;
	 int ymax = kSize / 2;

	 double xx, yy;

	 //cos and sin functions take only radian input. Convert from degrees to radians
	 theta = theta * CV_PI/180;
	 psi = psi * CV_PI / 180;

	 double del = 2.0 / (kSize - 1);
	 double gamma2 = gamma * gamma;

	 cv::Mat kernel(kSize, kSize, CV_32F);
	 sigma /= kSize;

	 double psum = 0.0, nsum = 0.0;
	 for(int y = -ymax; y <= ymax; ++y)
	 {
		 for(int x = -xmax; x <= xmax; ++x)
		 {
			 xx = (double)x * del * cos(theta) + (double)y * del * sin(theta);
			 yy = -(double)x * del * sin(theta) + (double)y * del * cos(theta);
			 double temp;

			 temp = kernel.at<float>(ymax + y, xmax + x) = (float)exp(-0.5 * ((xx * xx) + (gamma2 + yy * yy)) / (sigma * sigma)) * cos(2 * CV_PI * xx / lambda + psi);
			 if(temp >= 0) psum += temp;
			 else nsum += -temp;
		 }
	 }

         //Normalize the kernel values
	 //Possible reason for this to stay: We are using a Gaussian function, so the kernel should sum to 1.0(normally distributed)
	 double msum = (psum + nsum) / 2.0;
	 if(msum >= 0.0)
	 {
		 psum /= msum;
		 nsum /= msum;
	 }
	 for(int y = -ymax; y <= ymax; ++y)
	 {
		 for(int x = -xmax; x <= xmax; ++x)
		 {
			 float & temp = kernel.at<float>(ymax + y, xmax + x);
			 if(temp >= 0) temp *= nsum;
			 else temp *= psum;
		 }
	 }
	 return kernel;
}

/*!
 * For each variation in the ks, theta, sigma and lambda values a filter kernel is created
 */
void GaborFilter::createFilterBank()
{
	assert(ks.size() > 0 && theta.size() > 0 && sigma.size() > 0 && lambda.size() > 0);

	for(int i = 0; i < (int)ks.size(); ++i)
	{
		for(int j = 0; j < (int)theta.size(); ++j)
		{
			for(int k = 0; k < (int)sigma.size(); ++k)
			{
				for(int l = 0; l < (int)lambda.size(); ++l)
				{
					filterBank.push_back(getGaborKernel(theta[j], ks[i], sigma[k], lambda[l], psi, gamma, bandwidth));
				}
			}
		}
	}
}

/*!
 * For each filter in filterBank, apply the filter, store the results in a vector and
 * return a vector of images
 */
std::vector<cv::Mat> GaborFilter::applyFilter(cv::Mat & image)
{
	std::vector<cv::Mat> result;
	result.resize( this->getFilterBankSize() );
	for(int i = 0; i < this->getFilterBankSize(); ++i)
	{
		cv::filter2D(image, result[i], CV_32F, filterBank[i]);
	}
	return result;
}
