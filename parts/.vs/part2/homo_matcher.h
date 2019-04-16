#pragma once

#include <stdio.h>
#include <iostream>
#include<opencv2/opencv.hpp>  
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <vector> 
#include "func.h"
#include"evalution.h"
//#include "_modelest_jw.h"
//#include "calib3d_jw.hpp"
#include "grid_matcher.h"
#include "fast_grid.h"
using namespace cv;
using namespace std;

#define DEBUG_TIME
#define DEBUG_EVALUTION
//#define DEBUG_WRITE
extern string data_path;
extern string path;





void calcHomo(Mat& img01,  Mat& img02,vector<KeyPoint> &keypoint01, vector<KeyPoint> &keypoint02, vector<DMatch> &matches, Mat &m_hommography,int para=0)
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

	m_hommography = cv::findHomography(p01, p02, CV_RANSAC);
	if (para == 1)
	{
		#ifdef DEBUG_EVALUTION
				vector<DMatch> ransac_matches;
				vector<uchar> RansacStatus;
				Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);
				vector<KeyPoint> RR_keypoint01, RR_keypoint02;
				vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
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
						//cout << "distance" << i << R_keypoint01[i].response - R_keypoint02[i].response << endl;
					}
				}
				cout << "Ransac_matches size:" << RR_matches.size() << endl;
				Mat img_RR_matches;
				drawMatches(img01, RR_keypoint01, img02, RR_keypoint02, RR_matches, img_RR_matches);
				//imshow("消除误匹配点后", img_RR_matches);
				imwrite("img_RR_matches.jpg", img_RR_matches);
		#endif

	}


	cout << "单应矩阵：" << endl;
	cout<<m_hommography << endl;
	

}

void calcHomo(vector<KeyPoint> &keypoint01, vector<KeyPoint> &keypoint02, vector<DMatch> &matches, Mat &m_hommography, int para = 0)
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
	//KeyPoint::convert(R_keypoint01, p01);
	//KeyPoint::convert(R_keypoint02, p02);
	for (size_t i = 0; i<matches.size(); i++)
	{
		p01.push_back(R_keypoint01[i].pt);
		p02.push_back(R_keypoint02[i].pt);
	}
	
	m_hommography = cv::findHomography(p01, p02, CV_RANSAC);
	
}


void ransac(Mat &img01, Mat &img02, vector<KeyPoint> &keypoint01, vector<KeyPoint> &keypoint02, vector<DMatch> &matches, vector<DMatch> &RR_matches)
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
	Mat m_hommography = cv::findHomography(p01, p02, CV_RANSAC);


#ifdef DEBUG_EVALUTION
	vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);
	//cout << "Fundamental:" << Fundamental << endl;
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

#endif // DEBUG_EVALUTION


#ifdef DEBUG_EVALUTION
	//Mat m_hommography;
	int correspondences = 0;
	Mat homo;
	Mat img1_transform;
	get_homo(img01, img1_transform, homo);
	calculateCorrespondencesCount(img01, img02, homo, keypoint01, keypoint02, correspondences);
	int correctMatch = 0;
	int falseMatch = 0;
	evaluateMatcher(img01, img02, homo, keypoint01, keypoint02, RR_matches, correctMatch, falseMatch);
#endif

#ifdef DEBUG_WRITE
	Mat img_RR_matches;
	drawMatches(img01, RR_keypoint01, img02, RR_keypoint02, RR_matches, img_RR_matches);
	//imshow("消除误匹配点后", img_RR_matches);
	imwrite("img_RR_matches.jpg", img_RR_matches);

#endif // 


}



void grid_match_Homo(Mat& img01, Mat& img02, vector<KeyPoint> &keypoint01 , vector<KeyPoint> &keypoint02_origin, Mat& homography)
{
	
	vector<KeyPoint> keypoint02;

	int GRIDSIZE = 60;

	vector<DMatch> matches_back_big;
	vector<vector<KeyPoint>> grid_vkp1(GRIDSIZE*GRIDSIZE), grid_vkp2(GRIDSIZE*GRIDSIZE);

	OrbDescriptorExtractor extractor;
	Mat descriptor01, descriptor02;
	extractor.compute(img01, keypoint01, descriptor01);
	extractor.compute(img02, keypoint02_origin, descriptor02);
	vector<Mat> grid_descriptor02(3600);
	
	GridMatcher gmatcher(keypoint01, Size(img01.cols, img01.rows), keypoint02_origin, Size(img02.cols, img02.rows), grid_vkp1, grid_vkp2, descriptor02, grid_descriptor02,img02);
	gmatcher.Matcher(img01, img02, keypoint01, descriptor01, keypoint02_origin, keypoint02, grid_vkp2, grid_descriptor02, homography, matches_back_big);
	//cout << "fastMatch size:" << matches_back_big.size() << endl;
	vector<DMatch> RR_matches;

	//ransac(img01, img02, keypoint01, keypoint02, matches_back_big, RR_matches);

#ifdef DEBUG_EVALUTION
	ofstream f1(data_path, ios::app);
	if (!f1)return;
	f1 << "fastMatch size:" << matches_back_big.size() << endl;
	f1.close();
#endif // DEBUG_WRITE
	//评价
#ifdef DEBUG_EVALUTION
	//Mat m_hommography;
	int correspondences = 0;
	Mat homo;
	Mat img1_transform;
	get_homo(img01, img1_transform, homo);
	calculateCorrespondencesCount(img01, img02, homo, keypoint01, keypoint02_origin, correspondences);
	int correctMatch = 0;
	int falseMatch = 0;
	evaluateMatcher(img01, img02, homo, keypoint01, keypoint02, matches_back_big, correctMatch, falseMatch);
	
	
#endif
#ifdef DEBUG_TIME
	Mat outimg1;
	drawMatches(img01, keypoint01, img02, keypoint02, matches_back_big, outimg1);
	//imshow("FAST feature", outimg1);
	//imwrite("F://dataset//herzjesu_images//herzjesu//h00000_0007.jpg", outimg1);
	//imwrite("11//39.jpg", outimg1);
	imwrite("trevi02//finally.jpg", outimg1);
#endif
}



