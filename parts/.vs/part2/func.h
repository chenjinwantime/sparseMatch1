#pragma once
#include <stdio.h>
#include <iostream>
#include<opencv2/opencv.hpp>  
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <vector> 
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::detail;



static void
HarrisResponses(const Mat& img, vector<KeyPoint>& pts, int blockSize, float harris_k)
{
	CV_Assert(img.type() == CV_8UC1 && blockSize*blockSize <= 2048);

	size_t ptidx, ptsize = pts.size();

	const uchar* ptr00 = img.ptr<uchar>();
	int step = (int)(img.step / img.elemSize1());
	int r = blockSize / 2;

	float scale = (1 << 2) * blockSize * 255.0f;
	scale = 1.0f / scale;
	float scale_sq_sq = scale * scale * scale * scale;

	AutoBuffer<int> ofsbuf(blockSize*blockSize);
	int* ofs = ofsbuf;
	for (int i = 0; i < blockSize; i++)
		for (int j = 0; j < blockSize; j++)
			ofs[i*blockSize + j] = (int)(i*step + j);

	for (ptidx = 0; ptidx < ptsize; ptidx++)
	{
		int x0 = cvRound(pts[ptidx].pt.x - r);
		int y0 = cvRound(pts[ptidx].pt.y - r);

		const uchar* ptr0 = ptr00 + y0*step + x0;
		int a = 0, b = 0, c = 0;

		for (int k = 0; k < blockSize*blockSize; k++)
		{
			const uchar* ptr = ptr0 + ofs[k];
			int temp1 = (ptr[1] - ptr[-1]) * 2;
			int temp2 = (ptr[-step + 1] - ptr[-step - 1]);
			int temp3 = (ptr[step + 1] - ptr[step - 1]);
			int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]);



			int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]);
			a += Ix*Ix;
			b += Iy*Iy;
			c += Ix*Iy;
		}
		pts[ptidx].response = ((float)a * b - (float)c * c -
			harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
	}
}

void readFile() {
	FILE *fp;
	fp = fopen("E:\\algorithm\\fastMatchEvalution\\fastMatchEvalution\\graf\\H1to3p", "r");
	if (fp == NULL) {
		printf("Open File failed.\n");
		exit(0);
	}
	double a;
	double b;
	double c;
	
	//char str[100];
	while (!feof(fp)) {
		fscanf(fp, "%lf%lf%lf", &a, &b, &c);
		printf("Line:%lf %lf %lf\n", a, b, c);
	}
	fclose(fp);
}





