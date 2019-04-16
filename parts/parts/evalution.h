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

#include<fstream>
using namespace cv;
using namespace std;


static void filterKeyPointsByImageSize(vector<KeyPoint>& keypoints, const Size& imgSize)
{
	if (!keypoints.empty())
	{
		vector<KeyPoint> filtered;
		filtered.reserve(keypoints.size());
		vector<KeyPoint>::const_iterator it = keypoints.begin();
		for (int i = 0; it != keypoints.end(); ++it, i++)
		{
			if (it->pt.x  < imgSize.width &&
				it->pt.y < imgSize.height)
				filtered.push_back(*it);
		}
		keypoints.assign(filtered.begin(), filtered.end());
	}
}


static void computeOneToOneMatchedCorrespondencesCount(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2t,int &correspondencesCount)
{
	correspondencesCount = 0;
	for (size_t i1 = 0; i1 < keypoints1.size(); i1++)
	{
		KeyPoint kp1 = keypoints1[i1];
		
		for (size_t i2 = 0; i2 < keypoints2t.size(); i2++)
		{
			KeyPoint kp2 = keypoints2t[i2];
			Point2f diff = kp2.pt - kp1.pt;
			int maxDist = 5;
			if (norm(diff) < maxDist)
			{
				correspondencesCount++;
				break;
			}
		}
	}

	
}

void KeyPointsToPoints(vector<KeyPoint> kpts, vector<Point2f> &pts)
{
	for (int i = 0; i < kpts.size(); i++) {
		pts.push_back(kpts[i].pt);
	}
}

void PointsToKeyPoints(vector<Point2f>&pts, vector<KeyPoint>&kpts)
{
	for (size_t i = 0; i < pts.size(); i++) {
		kpts.push_back(KeyPoint(pts[i], 1.f));
	}
}


static void calculateCorrespondencesCount(const Mat& img1, const Mat& img2, const Mat& H1to2,
	const vector<KeyPoint>& _keypoints1, const vector<KeyPoint>& _keypoints2,
	int& correspondencesCount)
{

	vector<KeyPoint> keypoints1t, keypoints2t;
	vector<Point2f> _points1, _points2, points1t, points2t;
	KeyPointsToPoints(_keypoints1, _points1);
	KeyPointsToPoints(_keypoints2, _points2);
	perspectiveTransform(_points1, points2t, H1to2);
	Mat H2to1; invert(H1to2, H2to1);
	perspectiveTransform(_points2, points1t, H2to1);
	PointsToKeyPoints(points2t, keypoints2t);
	PointsToKeyPoints(points1t, keypoints1t);

	Size sz1 = img1.size(), sz2 = img2.size();
	filterKeyPointsByImageSize(keypoints1t, sz1);
	filterKeyPointsByImageSize(keypoints2t, sz2);
	size_t size1 = _keypoints1.size(), size2 = keypoints2t.size();
	size_t minCount = MIN(size1, size2);
	cout << "minCount" << minCount << endl;
	computeOneToOneMatchedCorrespondencesCount(_keypoints2, keypoints2t, correspondencesCount);
	cout << "correspondencesCount" << correspondencesCount << endl;


	ofstream f1("E:\\algorithm\\fastMatchEvalution\\fastMatchEvalution\\leuven\\data.txt", ios::app);
	if (!f1)return;
	f1 <<  "minCount£º" << minCount << endl;
	f1 <<  "correspondencesCount£º" << correspondencesCount << endl;
	f1.close();







	return;

}


void evaluateMatcher(const Mat& img1, const Mat& img2, const Mat& H1to2,
	vector<KeyPoint>& _keypoints1, vector<KeyPoint>& _keypoints2,
	vector<DMatch> _matches1to2,int &correctMatch,int &falseMatch)
{
	vector<KeyPoint>  keypoints2t;
	vector<Point2f> _points1, points2t;
	KeyPointsToPoints(_keypoints1, _points1);
	perspectiveTransform(_points1, points2t, H1to2);
	PointsToKeyPoints(points2t, keypoints2t);

	for (int i = 0; i < _matches1to2.size(); i++)
	{
		KeyPoint kp1 = keypoints2t[_matches1to2[i].queryIdx];
		KeyPoint kp2 = _keypoints2[_matches1to2[i].trainIdx];
		Point2f diff = kp2.pt - kp1.pt;
		int maxDist = 5;
		if (norm(diff) < maxDist)
		{
			correctMatch++;
			continue;
		}
		falseMatch++;

	}
	cout << "correctMatch" << correctMatch << endl;
	cout << "falseMatch" << falseMatch << endl;
	ofstream f1("E:\\algorithm\\fastMatchEvalution\\fastMatchEvalution\\leuven\\data.txt", ios::app);
	if (!f1)return;
	f1 << "correctMatch" << correctMatch << endl;
	f1 << "falseMatch" << falseMatch << endl;
	f1.close();
	return;




}
