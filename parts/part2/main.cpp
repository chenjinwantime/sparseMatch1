#include <stdio.h>
#include <iostream>
#include<opencv2/opencv.hpp>  
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <vector> 
#include <string>
#include "fast_grid.h"
#include "grid_matcher.h"
//#include "fundamental_matcher.h"
#include "gms_matcher.h"
#include "func.h"
#include "homo_matcher.h"
#include <future>
//#define DEBUG_WRITE
#define DEBUG_EVALUTION

using namespace cv;
using namespace std;
const float HARRIS_K = 0.04f;

string path = "E:\\algorithm\\parts\\part2\\trees\\";
string data_path = path + "data.txt";
string image_path1 = path + "img1.ppm";
string image_path2 = path + "img5.ppm";
//string image_path1 = path + "01.png";
//string image_path2 = path + "02.png";


void is_prime(Mat &img1, Mat &img2, vector<KeyPoint> &keypoint_result1, vector<KeyPoint> &keypoint_result2,
	vector<DMatch> &matches)
{
	Mat descriptor1, descriptor2;
	OrbDescriptorExtractor desc;
	//BriefDescriptorExtractor desc;
	//Mat descriptor1, descriptor2;
	desc.compute(img1, keypoint_result1, descriptor1);
	desc.compute(img2, keypoint_result2, descriptor2);
	//vector<DMatch> matches;

	BFMatcher matcher;
	matcher.match(descriptor1, descriptor2, matches);
	//cout << "1" << endl;

}

void is_prime1(Mat &img1, Mat &img2, vector<KeyPoint> &keypoint_result1, vector<KeyPoint> &keypoint_result2,
	vector<DMatch> &matches_surf)
{
	Mat descriptor1, descriptor2;
	BriefDescriptorExtractor desc;
	//BriefDescriptorExtractor desc;
	//Mat descriptor1, descriptor2;
	desc.compute(img1, keypoint_result1, descriptor1);
	desc.compute(img2, keypoint_result2, descriptor2);
	//vector<DMatch> matches_surf;
	BFMatcher matcher_surf;
	matcher_surf.match(descriptor1, descriptor2, matches_surf);
}

//单应矩阵
void main(int argc, char **argv)
{

	Mat img1, img2;

#ifdef  DEBUG_EVALUTION
	ofstream f1(data_path, ios::app);
	if (!f1)return;
	f1 << image_path1<<"##########"<< image_path2 << endl;
	f1.close();
#endif //  DEBUG_TIME

	img1 = imread(image_path1, CV_LOAD_IMAGE_GRAYSCALE);
	img2 = imread(image_path2, CV_LOAD_IMAGE_GRAYSCALE);
	//imresize(img1, 480);
	//imresize(img2, 480);
	
	std::vector<KeyPoint> keyPoints1, keyPoints2;
	int constract = 80;
	FastFeatureDetector fast(constract);   // 检测的阈值为80  
	fast.detect(img1, keyPoints1);
	//fast.detect(img2, keyPoints2);
	int keypointNum = 1000;
	while (keyPoints1.size() <keypointNum)
	{
		int distance = keypointNum - keyPoints1.size();
		keyPoints1.clear();
		if (distance>1000)
			constract -= (0.05*distance);
		else if (distance >= 1000)
			constract -= (0.02*distance);
		else if (distance >= 500)
			constract -= (0.01*distance);
		else
			constract = constract - 3;

		FastFeatureDetector fast1(constract);   // 检测的阈值为80  

		fast1.detect(img1, keyPoints1);
		
	}
	FastFeatureDetector fast2(constract);
	fast2.detect(img2, keyPoints2);
	
	HarrisResponses(img1, keyPoints1, 5, HARRIS_K);
	HarrisResponses(img2, keyPoints2, 5, HARRIS_K);
	
	fast_grid fastgrid1(keyPoints1, img1.cols, img1.rows, 10, 10);
	fast_grid fastgrid2(keyPoints2, img2.cols, img1.rows, 10, 10);
	KeyPointsFilter::runByImageBorder(fastgrid1.keypoint_result, img1.size(), 31);
	//KeyPointsFilter::runByKeypointSize(fastgrid1.keypoint_result, std::numeric_limits<float>::epsilon());

	KeyPointsFilter::runByImageBorder(fastgrid2.keypoint_result, img2.size(), 31);
	
	vector<DMatch> identical_matches;
	
	vector<DMatch> matches_surf;
	vector<DMatch> matches;
	std::thread t(is_prime, img1, img2, fastgrid1.keypoint_result, fastgrid2.keypoint_result,
	std::ref(matches));
	std::thread t1(is_prime1, img1, img2, fastgrid1.keypoint_result, fastgrid2.keypoint_result,
	std::ref(matches_surf));

	t.join();
	t1.join();
	for (int i = 0; i < matches.size(); ++i)
	{
	if (matches[i].trainIdx == matches_surf[i].trainIdx)
	identical_matches.push_back(matches[i]);
	}
#ifdef  DEBUG_WRITE
	ofstream f2("graf\\data_point_match.txt", ios::app);
	if (!f2)return;
	for (int i = 0; i < identical_matches.size(); i++)
	{
		f2 << identical_matches.size() << endl;
		//f2 <<"queryIdx: "<<identical_matches[i].queryIdx<<"   " <<fastgrid1.keypoint[identical_matches[i].queryIdx].pt.x << "  " << fastgrid1.keypoint[identical_matches[i].queryIdx].pt.y <<
		//	"  trainIdx:  "<<identical_matches[i].trainIdx <<"   "<< fastgrid2.keypoint[identical_matches[i].trainIdx].pt.x << "  " << fastgrid2.keypoint[identical_matches[i].trainIdx].pt.y<<endl;
		f2 << fastgrid1.keypoint_result[identical_matches[i].queryIdx].pt.x << "  " << fastgrid1.keypoint_result[identical_matches[i].queryIdx].pt.y <<
			"   " << fastgrid2.keypoint_result[identical_matches[i].trainIdx].pt.x << "  " << fastgrid2.keypoint_result[identical_matches[i].trainIdx].pt.y << endl;
	}
	//f1 << "img1.ppm  img5.ppm" << endl;
	f2.close();
#endif //  DEBUG_TIME

	Mat identicalImg1;
	drawMatches(img1, fastgrid1.keypoint_result, img2, fastgrid2.keypoint_result, identical_matches, identicalImg1);
	imwrite("trevi02\\identicalImg1.jpg", identicalImg1);
	Mat m_hommography;
	calcHomo(fastgrid1.keypoint_result, fastgrid2.keypoint_result, identical_matches, m_hommography);
	cout << "calcHomo" << m_hommography << endl;
	

	//grid_match_Homo(img1, img2, m_hommography);
	grid_match_Homo(img1, img2, keyPoints1, keyPoints2, m_hommography);//最快
	//cout << "H" << m_hommography << endl;


}


