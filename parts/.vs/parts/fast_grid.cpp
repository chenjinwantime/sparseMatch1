#include "fast_grid.h"
#include <algorithm>
#include <iostream>

bool response_sort(gridCell &kp1, gridCell &kp2)
{
	return kp1.coordinate.response < kp2.coordinate.response;
}


bool response(KeyPoint &kp1, KeyPoint &kp2)
{
	return kp1.response < kp2.response;
}

fast_grid::fast_grid(vector<KeyPoint> _keypoint,int _weight,int _height,int _cellX,int _cellY):
	keypoint(_keypoint),weight(_weight),height(_height),cellX(_cellX),cellY(_cellY)
{

	cellNumX = weight / cellX;
	cellNumY = height / cellY;
	point_grid.resize(cellNumX*cellNumY);
	keypointClassify();
	keypointSelect();

}
fast_grid::fast_grid(vector<KeyPoint> _keypoint, int _weight, int _height, int GRID_SIZE) :
	keypoint(_keypoint), weight(_weight), height(_height)
{
	int subwidth1 = ceil(weight / GRID_SIZE);
	int subheight1 = ceil(height / GRID_SIZE);
	//std::cout << "subwidth1:  " << subwidth1 << endl;
	//std::cout << "subheight1:  " << subheight1 << endl;

	point.resize(GRID_SIZE*GRID_SIZE);
	for (int i = 0; i < keypoint.size(); i++)
	{
		int index = (int)keypoint[i].pt.x / subwidth1 + (int)(keypoint[i].pt.y / subheight1) * GRID_SIZE;
		point[index].push_back(keypoint[i]);
	}
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; ++i)
	{
		if (!point[i].empty())
			sort(point[i].begin(), point[i].end(), response);
	}


	for (int i = 0; i < point.size(); ++i)
	{
		//if (point_grid[i].size() == 1)
		if (!point[i].empty())
		{
			if (point[i].size() == 1)
				keypoint_result.push_back(point[i][0]);
			else if (point[i].size() >= 2)
			{
				//for (int j = 0; j < point_grid[i].size() *2/ 3; ++j)
				for (int j = 0; j < std::min((int)point[i].size() / 2, 5); ++j)
				{
					keypoint_result.push_back(point[i][j]);
				}

				for (int j = std::min((int)point[i].size() / 2, 5); j < point[i].size(); ++j)
				{
					keypoint_remain.push_back(point[i][j]);
				}

			}
		}

	}
}

void fast_grid::keypointClassify()
{
	
	for (int i = 0; i < keypoint.size(); ++i)
	{
		int x = keypoint[i].pt.x;
		int y = keypoint[i].pt.y;
		int index = x / cellX + (int)y / cellY*cellNumX;
		point_grid[index].push_back(gridCell(index, keypoint[i]));
	}

	for (int i = 0; i < cellNumX*cellNumY; ++i)
	{
		if (!point_grid[i].empty())
			sort(point_grid[i].begin(), point_grid[i].end(), response_sort);
	}


}


void fast_grid::keypointSelect()
{
	for (int i = 0; i < point_grid.size(); ++i)
	{
		//if (point_grid[i].size() == 1)
		if (!point_grid[i].empty())
		{
			if (point_grid[i].size() == 1)
				keypoint_result.push_back(point_grid[i][0].coordinate);
			else if (point_grid[i].size() >= 2)
			{
				//for (int j = 0; j < point_grid[i].size() *2/ 3; ++j)
				for (int j = 0; j < std::min((int)point_grid[i].size() / 2,1); ++j)
				{
					keypoint_result.push_back(point_grid[i][j].coordinate);
				}

				for (int j = std::min((int)point_grid[i].size() / 2, 1); j < point_grid[i].size(); ++j)
				{
					keypoint_remain.push_back(point_grid[i][j].coordinate);
				}

			}
		}
	}
	
}
fast_grid::~fast_grid()
{
	
}