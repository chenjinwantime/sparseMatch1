#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <windows.h>
#include <ppl.h>
#include <mutex>
using namespace std;
using namespace cv;
using namespace concurrency;
#define GRID_SIZE 40
mutex mut;
class GridMatcher
{
public:
	vector<Point2f> mvP1, mvP2;
	//Grid Size
	Size mGridSizeLeft, mGridSizeRight;
	int mGridNumberLeft, mGridNumberRight;
	int subwidth1, subheight1;
	int subwidth2, subheight2;
	//vector<vector<KeyPoint>> grid_vkp1, grid_vkp2;
	GridMatcher(const vector<KeyPoint> &vkp1, const Size size1, 
		const vector<KeyPoint> &vkp2, const Size size2, 
		vector<vector<KeyPoint>> &grid_vkp1, vector<vector<KeyPoint>> &grid_vkp2, Mat decsriptor02,vector<Mat> &grid_descriptor02,Mat &img02)
	{
		subwidth1 = ceil(size1.width / GRID_SIZE);
		subheight1 = ceil(size1.height / GRID_SIZE);
		//std::cout << "subwidth1:  " << subwidth1 << endl;
		//std::cout << "subheight1:  " << subheight1 << endl;
		subwidth2 = ceil(size2.width / GRID_SIZE);
		subheight2 = ceil(size2.height / GRID_SIZE);
		initright(vkp2, grid_vkp2,  decsriptor02,grid_descriptor02,img02);
	}

	void initleft(const vector<KeyPoint> &vkp1, vector<vector<KeyPoint>> &grid_vkp1,Mat decsriptor02,vector<Mat> &grid_decsriptor02)
	{
		//for (int i = 0; i < vkp1.size(); ++i)
		Concurrency::parallel_for(size_t(0), vkp1.size(), [&](int i)
		{
			int index= (int)vkp1[i].pt.x / subwidth1 + (int)(vkp1[i].pt.y / subheight1) * GRID_SIZE;
			grid_vkp1[index].push_back(vkp1[i]);
		});
	}
	void initright(const vector<KeyPoint> &vkp2,
		vector<vector<KeyPoint>> &grid_vkp2, Mat descriptor02,  vector<Mat> &grid_decsriptor02, Mat &img02)
	{

		for (int i = 0; i < vkp2.size(); ++i)
		//Concurrency::parallel_for(size_t(0), vkp2.size(), [&](int i)
		{
			int index = (int)vkp2[i].pt.x / subwidth2 + (int)(vkp2[i].pt.y / subheight2) *GRID_SIZE;
			//mut.lock();
			grid_vkp2[index].push_back(vkp2[i]);
			Mat temp = descriptor02.rowRange(i, i + 1).clone();
			grid_decsriptor02[index].push_back(temp);
			//mut.unlock();
			}
		//);
		
		
	}
	void Matcher(const Mat &img01, const Mat &img02, vector<KeyPoint> &vkp1,Mat &decsriptor01, vector<KeyPoint> &vkp2,vector<KeyPoint> &vkp2_match,
		vector<vector<KeyPoint>> &grid_vkp2, vector<Mat> &grid_decsriptor02, Mat &homo, vector<DMatch> &big_matches)
	{
		//vector<Point2f> point01_left;
		vector<Point2f> point01_left(vkp1.size());
		vector<Point2f> point02_right(vkp1.size());
		double Time = (double)cvGetTickCount();

		for (int i = 0; i < vkp1.size(); ++i)
		{
			point01_left[i]=vkp1[i].pt;
		}
	
		perspectiveTransform(point01_left, point02_right, homo);

		//OrbDescriptorExtractor extractor;
		//Mat decsriptor01;
		//extractor.compute(img01, vkp1, decsriptor01);

		//Mat decsriptor02;
		//extractor.compute(img02, vkp2, decsriptor02);
		Mat subdescriptor02;
		int count = 0;
		//vector<DMatch> big_matches;
		BFMatcher matcher;
		//int Time = (double)cvGetTickCount();

		for (int i = 0; i < vkp1.size(); ++i)
		{
			if (point02_right[i].x < 0 || point02_right[i].y < 0||point02_right[i].x>img01.cols||point02_right[i].y>img01.rows) continue;
			int index = (int)point02_right[i].x / subwidth1 + (int)(point02_right[i].y / subheight2 )  *GRID_SIZE;
			if (index >= GRID_SIZE*GRID_SIZE) continue;
			if (grid_vkp2[index].empty()) continue;
			//if (grid_vkp2[index].size() == 1)
			//{

			//	vkp2_match.push_back(grid_vkp2[index][0]);
			//	DMatch matches_temp;
			//	matches_temp.queryIdx = i;

			//	matches_temp.trainIdx = count;
			//	//matches_temp.distance = matches[0].distance;
			//	big_matches.push_back(matches_temp);
			//	count++;
			//	continue;

			//}
			Mat subdescriptor01(1, 32, CV_32FC1);
			decsriptor01.row(i).copyTo(subdescriptor01.row(0));
			
			vector<DMatch> matches;
			Mat img_matches;
			grid_decsriptor02[index].convertTo(grid_decsriptor02[index], CV_32FC1);
			matcher.match(subdescriptor01, grid_decsriptor02[index], matches);
			//Ìí¼Óµã
			vkp2_match.push_back(grid_vkp2[index][matches[0].trainIdx]);
			DMatch matches_temp;
			matches_temp.queryIdx = i;

			matches_temp.trainIdx = count;
			matches_temp.distance = matches[0].distance;
			big_matches.push_back(matches_temp);
			count++;
		}
		
	}

	~GridMatcher() {}
};

