#include <stdio.h>
#include <iostream>
#include<opencv2/opencv.hpp>  
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <vector> 
#include "_modelest_jw.h"
#include "calib3d_jw.hpp"
using namespace cv;
using namespace std;
void ransac(Mat &img01, Mat &img02, vector<KeyPoint> &keypoint01, vector<KeyPoint> &keypoint02, vector<DMatch> &matches, vector<DMatch> &RR_matches,int flag=0)
{

	vector<KeyPoint> R_keypoint01, R_keypoint02;
	for (size_t i = 0; i<matches.size(); i++)
	{
		R_keypoint01.push_back(keypoint01[matches[i].queryIdx]);
		R_keypoint02.push_back(keypoint02[matches[i].trainIdx]);
		//这两句话的理解：R_keypoint1是要存储img01中能与img02匹配的特征点，
		//matches中存储了这些匹配点对的img01和img02的索引值
	}

	//坐标转换
	vector<Point2f>p01, p02;
	for (size_t i = 0; i<matches.size(); i++)
	{
		p01.push_back(R_keypoint01[i].pt);
		p02.push_back(R_keypoint02[i].pt);
	}


	double Time = (double)cvGetTickCount();
	Mat m_hommography;
	if (flag == 0)
	{
		m_hommography = cv::findHomography(p01, p02, CV_RANSAC);
		Time = (double)cvGetTickCount() - Time;
		printf("ransac time = %gms\n", Time / (cvGetTickFrequency() * 1000));//毫秒
		printf("ransac time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//毫秒
		cout << "homography:" << m_hommography << endl;
	}
	if (flag == 1)
	{
		m_hommography = jw::findHomography(p01, p02, CV_RANSAC);
		Time = (double)cvGetTickCount() - Time;
		printf("proposed ransac time = %gms\n", Time / (cvGetTickFrequency() * 1000));//毫秒
		printf("proposed ransac time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//毫秒
		cout << "proposed homography:" << m_hommography << endl;
	}
	vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);
	cout << Fundamental << endl;
	vector<KeyPoint> RR_keypoint01, RR_keypoint02;
	//vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
	int index = 0;
	for (size_t i = 0; i<matches.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			RR_keypoint01.push_back(R_keypoint01[i]);
			RR_keypoint02.push_back(R_keypoint02[i]);
			matches[i].queryIdx = index;
			matches[i].trainIdx = index;
			RR_matches.push_back(matches[i]);
			index++;
		}
	}
	cout << "Ransac Matches size:" << RR_matches.size() << endl;

	Mat img_RR_matches;
	drawMatches(img01, RR_keypoint01, img02, RR_keypoint02, RR_matches, img_RR_matches);
	//imshow("消除误匹配点后", img_RR_matches);
	imwrite("leuven\\img_RR_matches.jpg", img_RR_matches);

}



void main()
{
	Mat img1, img2;
	//img1 = imread("4\\1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("4\\2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	
	cout << "leuven" << endl;
	img1 = imread("leuven\\img1.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	img2 = imread("leuven\\img2.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("bank\\01.png", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("bank\\02.png", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("adam\\adamA.png", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("adam\\adamB.png", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("4\\1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("4\\2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("5\\1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("5\\2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//double Time = (double)cvGetTickCount();
	vector<KeyPoint> keypoints1, keypoints2;
	//SiftFeatureDetector detector(200);
	OrbFeatureDetector detector(3000);
	detector.detect(img1, keypoints1);
	detector.detect(img2, keypoints2);
	Mat descriptor1, descriptor2;
	OrbDescriptorExtractor desc;
	//BriefDescriptorExtractor desc;
	//Mat descriptor1, descriptor2;
	desc.compute(img1, keypoints1, descriptor1);
	desc.compute(img2, keypoints2, descriptor2);
	//vector<DMatch> matches;
	vector<DMatch> matches;
	BFMatcher matcher;
	matcher.match(descriptor1, descriptor2, matches);
	vector<DMatch> RR_matches;
	vector<DMatch> RR_matches_pro;
	ransac(img1, img2, keypoints1, keypoints2, matches, RR_matches,0);
	vector<DMatch> matches_pro;
	BFMatcher matcher_pro;
	matcher_pro.match(descriptor1, descriptor2, matches_pro);
	ransac(img1, img2, keypoints1, keypoints2, matches_pro, RR_matches_pro,1);

}