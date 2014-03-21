#ifndef GABOR_FILTER_H_
#define GABOR_FILTER_H_

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "string"

/*! The class supports a single filter or a bank of filters
 * 
 * TO CREATE A BANK OF FILTERS:
 * 4 variable parameters are supported
 * ks - Kernel Size
 * theta - Orientation of the sinusoid with respect to the normal distribution function
 * sigma - Standard deviation of the Gaussian function
 * lambda - Wavelength of the sinusoid
 */
class GaborFilter
{

private:
	std::vector<int> ks;
	std::vector<double> theta;
	std::vector<double> sigma;
	std::vector<double> lambda;
	double psi;
	double gamma;
	double bandwidth;

	std::vector<cv::Mat> filterBank;

public:
        void setDefaultParameters(cv::Mat & src);
	int getFilterBankSize() const
	{
		return (int)filterBank.size();
	}

        /*!
         * Function to get a single Gabor Kernel based on the parameters
         * Useful when you don't want a filter bank
         */
	cv::Mat getGaborKernel(double theta, int kernel_size, double sigma, double lambda, double psi, double gamma, double bandwidth);

        /*! 
         * Function to get a bank of Gabor Kernels based on the values of ks, theta, sigma, lambda 
         */
	void createFilterBank();

        /*!
         * Functions to populate the various kernel values
         * Use only when you want a filter bank
         */
	void addTheta(double val)
	{
		theta.push_back(val);
	}
	void addKernelSize(int val)
	{
		ks.push_back(val);
	}
	void addSigma(double val)
	{
		sigma.push_back(val);
	}
	void addLambda(double val)
	{
		lambda.push_back(val);
	}
	void setPSI(double val)
	{
		psi = val;
	}
	void setGamma(double val)
	{
		gamma = val;
	}
	void setBandwidth(double val)
	{
		bandwidth = val;
	}

        /*!
         * Function to apply a single filter or a bank of filters
         * Uses the class member filterBank as the kernel
         */
	std::vector<cv::Mat> applyFilter(cv::Mat & image);

};

#endif /* GABOR_FILTER_H_ */
