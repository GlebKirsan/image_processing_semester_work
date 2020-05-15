#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include "ini.h"

using namespace std;
using namespace cv;
using cv::utils::fs::glob;
using cv::utils::fs::createDirectory;

vector<string> GetImagePaths(const string& directory) {
	vector<string> cv_paths;
	bool recursive = true;
	glob(directory, "*.png", cv_paths, recursive);
	return cv_paths;
}

pair<double, double> GetMseAndPsnr(const Mat& I1, const Mat& I2) {
    Mat s1;
    absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    Scalar s = sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];
    double mse  = sse / (double)(I1.channels() * I1.total());
    if(sse <= 1e-10) {
        return {mse, 0};
    } else {
        double psnr = 10.0 * log10((255 * 255) / mse);
        return {mse, psnr};
    }
}

Scalar GetSSIM(const Mat& i1, const Mat& i2) {
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);
    Mat I1_2   = I1.mul(I1);
    Mat I1_I2  = I1.mul(I2);
    
    Mat mu1, mu2;
    Size kernel(9, 9);
    GaussianBlur(I1, mu1, kernel, 1.5);
    GaussianBlur(I2, mu2, kernel, 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, kernel, 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, kernel, 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, kernel, 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);
    
    Mat ssim_map;
    divide(t3, t1, ssim_map);
    return mean(ssim_map);
}

#define as(x) #x << " = " << x
int main() {
	vector<string> sources = GetImagePaths("../images");
	vector<string> noised = GetImagePaths("../noised");

	for (int i = 0; i < sources.size(); ++i) {
		Mat source = imread(sources[i]);
		Mat noise = imread(noised[i]);
		auto [mse, psnr] = GetMseAndPsnr(source, noise);
		Scalar ssim = GetSSIM(source, noise);
		cout << as(mse) << ", " << as(psnr) << ", " << as(ssim);
		cout << endl;
	}
}
