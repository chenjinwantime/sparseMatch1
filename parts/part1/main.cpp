#include <stdio.h>
#include <iostream>
#include<opencv2/opencv.hpp>  
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <vector> 
#include "fast_grid.h"
//#include "fundamental_matcher.h"
#include "func.h"
#include <future>
#define DEBUG_WRITE
#define DEBUG_EVALUTION

using namespace cv;
using namespace std;
const float HARRIS_K = 0.04f;


//µ•”¶æÿ’Û
void main(int argc, char **argv)
{

	Mat img1, img2;

	img1 = imread("ubc\\img1.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	img2 = imread("ubc\\img2.ppm", CV_LOAD_IMAGE_GRAYSCALE);

	imresize(img1, 480);
	imresize(img2, 480);
	
	double Time = (double)cvGetTickCount();


	std::vector<KeyPoint> keyPoints1, keyPoints2;
	// construction of the fast feature detector object  

	//Time = (double)cvGetTickCount();
	int constract = 80;
	FastFeatureDetector fast(constract);   // ºÏ≤‚µƒ„–÷µŒ™80  
										   // feature point detection  
										   //OrbFeatureDetector fast(1500);

	fast.detect(img1, keyPoints1);
	//fast.detect(img2, keyPoints2);
	int keypointNum = 1000;
	while (keyPoints1.size() < keypointNum)
	{
		int distance = keypointNum - keyPoints1.size();
		keyPoints1.clear();
		//keyPoints2.clear();
		//constract -= 20;
		//constract -= (5 * 2 / 3.14*atan(distance/1000) + 2);
		if (distance > 1500)
			constract -= (0.05*distance);
		else if (distance >= 1000)
			constract -= (0.02*distance);
		else if (distance >= 500)
			constract -= (0.01*distance);
		else
			constract = constract - 3;

		FastFeatureDetector fast1(constract);   // ºÏ≤‚µƒ„–÷µŒ™80  

		fast1.detect(img1, keyPoints1);
		//fast1.detect(img2, keyPoints2);
		//cout << "constract" << constract << endl;
		//cout << "keyPoints1 size:   " << keyPoints1.size() << endl;
	}
	FastFeatureDetector fast2(constract);
	fast2.detect(img2, keyPoints2);

	//cout << "keyPoints2 size:   " << keyPoints2.size() << endl;
	//HarrisResponses(img1, keyPoints1,5, HARRIS_K);
	//HarrisResponses(img2, keyPoints2, 5, HARRIS_K);

	//Time = (double)cvGetTickCount() - Time;
	//printf("FastFeatureDetector time = %gms\n", Time / (cvGetTickFrequency() * 1000));//∫¡√Î
	//printf("FastFeatureDetector time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//∫¡√Î
	HarrisResponses(img1, keyPoints1, 5, HARRIS_K);
	HarrisResponses(img2, keyPoints2, 5, HARRIS_K);
	//Time = (double)cvGetTickCount();



	fast_grid fastgrid1(keyPoints1, img1.cols, img1.rows, 10, 10);
	fast_grid fastgrid2(keyPoints2, img2.cols, img1.rows, 10, 10);



	KeyPointsFilter::runByImageBorder(fastgrid1.keypoint_result, img1.size(), 31);
	//KeyPointsFilter::runByKeypointSize(fastgrid1.keypoint_result, std::numeric_limits<float>::epsilon());

	KeyPointsFilter::runByImageBorder(fastgrid2.keypoint_result, img2.size(), 31);
	


	vector<DMatch> identical_matches;
	
	Time = (double)cvGetTickCount();
	Mat descriptor1, descriptor2;
	OrbDescriptorExtractor desc;
	desc.compute(img1, fastgrid1.keypoint_result, descriptor1);
	desc.compute(img2, fastgrid2.keypoint_result, descriptor2);

	Time = (double)cvGetTickCount() - Time;
	printf("orb√Ë ˆ time = %gms\n", Time / (cvGetTickFrequency() * 1000));//∫¡√Î
	printf("orb√Ë ˆ  time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//∫¡√Î
	//imwrite("descriptor1.jpg", descriptor1);
	//imwrite("descriptor2.jpg", descriptor2);
	Time = (double)cvGetTickCount();
	vector<DMatch> matches;
	BFMatcher matcher;
	matcher.match(descriptor1, descriptor2, matches);
	Time = (double)cvGetTickCount() - Time;
	printf("orb∆•≈‰ time = %gms\n", Time / (cvGetTickFrequency() * 1000));//∫¡√Î
	printf("orb∆•≈‰  time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//∫¡√Î
	



	Time = (double)cvGetTickCount();
	Mat descriptor1_surf, descriptor2_surf;
	BriefDescriptorExtractor desc_surf;
	desc_surf.compute(img1, fastgrid1.keypoint_result, descriptor1_surf);
	desc_surf.compute(img2, fastgrid2.keypoint_result, descriptor2_surf);
	//strongMatches(descriptor1, descriptor2, descriptor1_surf, descriptor2_surf, identical_matches,1);

	Time = (double)cvGetTickCount() - Time;
	printf("brief√Ë ˆ time = %gms\n", Time / (cvGetTickFrequency() * 1000));//∫¡√Î
	printf("brief√Ë ˆ  time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//∫¡√Î
																		   //imwrite("descriptor1.jpg", descriptor1);
																		   //imwrite("descriptor2.jpg", descriptor2);
	Time = (double)cvGetTickCount();


	vector<DMatch> matches_surf;
	BFMatcher matcher_surf;
	matcher_surf.match(descriptor1_surf, descriptor2_surf, matches_surf);
	Time = (double)cvGetTickCount() - Time;
	printf("brief∆•≈‰ time = %gms\n", Time / (cvGetTickFrequency() * 1000));//∫¡√Î
	printf("brief∆•≈‰ time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//∫¡√Î




	Time = (double)cvGetTickCount();
	for (int i = 0; i < matches.size(); ++i)
	{
		if (matches[i].trainIdx == matches_surf[i].trainIdx)
			identical_matches.push_back(matches[i]);
	}

	Time = (double)cvGetTickCount() - Time;
	printf("Õ∂∆±…∏—° time = %gms\n", Time / (cvGetTickFrequency() * 1000));//∫¡√Î
	printf("Õ∂∆±…∏—° time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//∫¡√Î

	Mat briefImg1;
	drawMatches(img1, fastgrid1.keypoint_result, img2, fastgrid2.keypoint_result, matches, briefImg1);
	imwrite("ubc\\brief.jpg", briefImg1);

	Mat orbImg1;
	drawMatches(img1, fastgrid1.keypoint_result, img2, fastgrid2.keypoint_result, matches_surf, orbImg1);
	imwrite("ubc\\orb.jpg", orbImg1);



	Mat identicalImg1;
	drawMatches(img1, fastgrid1.keypoint_result, img2, fastgrid2.keypoint_result, identical_matches, identicalImg1);
	imwrite("ubc\\identicalImg1.jpg", identicalImg1);

	//º∆À„µ•”¶æÿ’Û
	Mat brief_homo;
	Mat orb_homo;
	Mat select_homo;
	printf("orb:");//∫¡√Î
	calcHomo(fastgrid1.keypoint_result, fastgrid2.keypoint_result, matches, brief_homo);
	printf("brief:");
	calcHomo(fastgrid1.keypoint_result, fastgrid2.keypoint_result, matches_surf, orb_homo);
	printf("…∏—°:");
	calcHomo(fastgrid1.keypoint_result, fastgrid2.keypoint_result, identical_matches, select_homo);

}