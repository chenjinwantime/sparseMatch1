#pragma once
#include <vector>
#include "opencv2/features2d/features2d.hpp"
#include <algorithm> 
using namespace cv;
using namespace std;
class gridCell {
public:
	int cellIndex;
	KeyPoint coordinate;
	gridCell(int _cellIndex, KeyPoint _coordinate) :cellIndex(_cellIndex),coordinate(_coordinate){}

	~gridCell() {}
};
class fast_grid
{
public:
	fast_grid(vector<KeyPoint> _keypoint,int _weight,int _height,int _cellX, int cellY);
	fast_grid(vector<KeyPoint> _keypoint, int _weight, int _height, int GRID_SIZE);
	~fast_grid();
	int cellX;
	int cellY;
	int weight;
	int height;
	int cellNumX;
	int cellNumY;
	vector<KeyPoint> keypoint;
	void keypointClassify();	
	void keypointSelect();
	vector<vector<gridCell>> point_grid;
	vector<KeyPoint> keypoint_result;
	vector<KeyPoint> keypoint_remain;

	vector<vector<KeyPoint>> point;

};

