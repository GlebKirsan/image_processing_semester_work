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

void CheckPaths(const vector<string>& paths) {
	cerr << "Image paths:\n";
	for (const string& path : paths) {
		cerr << path << '\n';
	}
}

unordered_map<string, float> GetFloatValues(const Ini::Document& config,
		                            const string& section) {
	unordered_map<string, float> result;
	for (const auto& [k, v] : config.GetSection(section)) {
		result[k] = stof(v);
	}
	return result;
}

unordered_map<string, int> GetCirclePosition(const Ini::Document& config) {
	unordered_map<string, int> result;
	for (const auto& [k, v] : config.GetSection("CIRCLE POSITION")) {
		result[k] = stoi(v);
	}
	return result;
}

Mat AddNoise(const Mat& image) {
	cerr << "Adding gaussian noise" << endl;
	static RNG random;
	fstream file("../config.ini");
	const int image_width = image.size().width;
	const int image_height = image.size().height;
	const float min_dimension = min(image_width, image_height);
	const Ini::Document config = Ini::Load(file);
	unordered_map<string, float> radii_multiplier = 
		GetFloatValues(config, "RADII");
	const float ORADIUS = random.uniform(
			min_dimension * radii_multiplier["OUTER MINIMUM"], 
			min_dimension * radii_multiplier["OUTER MAXIMUM"]);
	const float IRADIUS = random.uniform(
			radii_multiplier["INNER MINIMUM"], 
			ORADIUS * radii_multiplier["INNER MAXIMUM"]);
	const float RADIUS_DIFF = ORADIUS - IRADIUS;
	Mat glare(image.size(), CV_32F, Scalar(0));
	unordered_map<string, int> circle_position = GetCirclePosition(config);
	const int x_circle = random.uniform(
			circle_position["X MINIMUM"], 
			image_width * circle_position["X MAXIMUM"]);
	const int y_circle = random.uniform(
			circle_position["Y MINIMUM"], 
			image_height * circle_position["Y MAXIMUM"]);
	
   	for(size_t r = 0; r < glare.rows; r++){
      		for(size_t c = 0; c < glare.cols; c++) {
			const int x_dist = c - x_circle;
			const int y_dist = r - y_circle;
			const float radius = hypot(x_dist, y_dist);
			float& pixel = glare.at<float>(r, c);
			if (radius > ORADIUS) {
				pixel = 0;
			} else if(radius < IRADIUS) {
				pixel = 1.0; 
			} else {
				float num = radius - IRADIUS;
		        	pixel = 1 - num / RADIUS_DIFF;
			}
      		}
   	}
	glare *= 255;
	glare.convertTo(glare, CV_8U);
	cvtColor(glare, glare, COLOR_GRAY2BGR);
	
	Mat result;
	unordered_map<string, float> transparency_multiplier = 
		GetFloatValues(config, "TRANSPARENCY");
	float transparency = random.uniform(transparency_multiplier["MINIMUM"],
			                    transparency_multiplier["MAXIMUM"]);
	addWeighted(image, 1, glare, transparency, 0.0, result);
	return result;
}

Mat StackHorizontal(const Mat& image1, const Mat& image2) {
	Mat result;
	cerr << "Concatenation" << endl;
	hconcat(image1, image2, result);
	return result;
}

void SaveImage(const string& directory, 
	       const string& filename, 
	       const Mat& image) {
	cerr << "Creating directory " << directory << endl;
	createDirectory(directory);
	const string path = directory + '/' + filename;
	cerr << "Saving to " << path << endl;
	imwrite(path, image);
}

int main() {
	const vector<string> image_paths = GetImagePaths("../images");
	CheckPaths(image_paths);
	int i = 0;
	for (const string& path : image_paths) {
		cerr << "Reading source image " << path << endl;
		Mat source = imread(path);
		Mat noised = AddNoise(source);
		string filename = to_string(i) + ".png";
		SaveImage("../noised", filename, noised);
		Mat concat = StackHorizontal(source, noised);
		SaveImage("../concat", filename, concat);
		++i;
	}
	return 0;
}
