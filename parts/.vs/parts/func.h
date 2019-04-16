#pragma once
#include <stdio.h>
#include <iostream>
#include<opencv2/opencv.hpp>  
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/stitching/detail/seam_finders.hpp>
#include "opencv2/stitching/detail/blenders.hpp"
#include <vector> 
#include <fstream>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace cv::detail;

void MyGammaCorrection(Mat& src, Mat& dst, float fGamma)
{
	CV_Assert(src.data);

	// accept only char type matrices
	CV_Assert(src.depth() != sizeof(uchar));

	// build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{

		MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			//*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
			*it = lut[(*it)];

		break;
	}
	case 3:
	{

		MatIterator_<Vec3b> it, end;
		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
		{
			//(*it)[0] = pow((float)(((*it)[0])/255.0), fGamma) * 255.0;
			//(*it)[1] = pow((float)(((*it)[1])/255.0), fGamma) * 255.0;
			//(*it)[2] = pow((float)(((*it)[2])/255.0), fGamma) * 255.0;
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}
}

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





static void get_homo(const Mat& src, Mat& dst, Mat& H)
{
	H.create(3, 3, CV_32FC1);
	//readFile();

	ofstream f1("E:\\algorithm\\fastMatchEvalution\\fastMatchEvalution\\leuven\\data.txt", ios::app);
	if (!f1)return;
	f1 << "H1to5p" << endl;
	f1.close();

	FILE *fp;
	fp = fopen("E:\\algorithm\\fastMatchEvalution\\fastMatchEvalution\\leuven\\H1to2p", "r");
	if (fp == NULL) {
		printf(" get_homo Open File failed.\n");
		exit(0);
	}
	double H00;
	double H01;
	double H02;
	double H10;
	double H11;
	double H12;
	double H20;
	double H21;
	double H22;
	fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &H00, &H01, &H02, &H10, &H11, &H12, &H20, &H21, &H22);
	printf(  "%lf  %lf  %lf  %lf  %lf  %lf  %lf  %lf  lf", H00, H01, H02, H10, H11, H12, H20, H21,H22);
	//char str[100];
	//while (!feof(fp)) {
		//fscanf(fp, "%lf%lf%lf", &a, &b, &c,);
		//printf("Line:%lf %lf %lf\n", a, b, c);
	//}
	fclose(fp);
	H.at<float>(0, 0) = H00;
	H.at<float>(0, 1) = H01;
	H.at<float>(0, 2) = H02;
	H.at<float>(1, 0) = H10;
	H.at<float>(1, 1) = H11;
	H.at<float>(1, 2) = H12;
	H.at<float>(2, 0) = H20;
	H.at<float>(2, 1) = H21;
	H.at<float>(2, 2) = H22;

	/*
	H.at<float>(0, 0) = 8.7976964e-01;
	H.at<float>(0, 1) = 3.1245438e-01;
	H.at<float>(0, 2) = -3.9430589e+01;
	H.at<float>(1, 0) = -1.8389418e-01;
	H.at<float>(1, 1) = 9.3847198e-01;
	H.at<float>(1, 2) = 1.5315784e+02;
	H.at<float>(2, 0) = 1.9641425e-04;
	H.at<float>(2, 1) = -1.6015275e-05;
	H.at<float>(2, 2) = 1.0000000e+00;
	*/
	/*H.at<float>(0, 0) = 1.0069126947300848;
	H.at<float>(0, 1) = -0.059399055250437316;
	H.at<float>(0, 2) = -7.189614652014083;
	H.at<float>(1, 0) = -0.0027756672772947626;
	H.at<float>(1, 1) = 0.9589243921374578;
	H.at<float>(1, 2) = 39.20575988656465;
	H.at<float>(2, 0) = 2.8434591017074435E-5;
	H.at<float>(2, 1) = -1.1030292818482223E-4;
	H.at<float>(2, 2) = 1.0000000e+00;*/
	
	
	/*H.at<float>(0, 0) = 7.6285898e-01;
	H.at<float>(0, 1) = -2.9922929e-01;
	H.at<float>(0, 2) = 2.2567123e+02;
	H.at<float>(1, 0) = 3.3443473e-01;
	H.at<float>(1, 1) = 1.0143901e+00;
	H.at<float>(1, 2) = -7.6999973e+01;
	H.at<float>(2, 0) = 3.4663091e-04;
	H.at<float>(2, 1) = -1.4364524e-05;
	H.at<float>(2, 2) = 1.0000000e+00;*/

	warpPerspective(src, dst, H, src.size());

}

 void strongMatches(Mat &descriptor1, Mat &descriptor2, Mat &descriptor1_surf, Mat &descriptor2_surf, vector<DMatch> &identical_matches, int para = 0)
{
	if (para == 1)
	{
		vector<DMatch> matches;
		//BFMatcher matcher(NORM_L2, true);
		BFMatcher matcher;
		matcher.match(descriptor1, descriptor2, matches);



		//drawKeypoints(img1, fastgrid1.keypoint_result, img1, Scalar::all(255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
		double Time = (double)cvGetTickCount();
		vector<DMatch> matches_surf;
		//BFMatcher matcher(NORM_L2, true);
		BFMatcher matcher_surf;
		matcher_surf.match(descriptor1_surf, descriptor2_surf, matches_surf);
		//drawKeypoints(img1, fastgrid1.keypoint_result, img1, Scalar::all(255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
		Time = (double)cvGetTickCount() - Time;
		printf("ransac time = %gms\n", Time / (cvGetTickFrequency() * 1000));//毫秒
		printf("ransac time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//毫秒
		for (int i = 0; i < matches.size(); ++i)
		{
			if (matches[i].trainIdx == matches_surf[i].trainIdx)
				identical_matches.push_back(matches[i]);
		}
		return;
	}
	if (para == 0)
	{
		Mat descriptor1_merge, descriptor2_merge;

		//hconcat(descriptor1, descriptor1_surf, descriptor1_merge);
		//hconcat(descriptor2, descriptor2_surf, descriptor2_merge);

		//vector<DMatch> identical_matches;
		//BFMatcher matcher(NORM_L2, true);
		BFMatcher matcher;
		matcher.match(descriptor1, descriptor2, identical_matches);
		return;
	}
	

}


Mat mirror(Mat src)
{
	Mat result;
	result.create(src.size(), src.type());
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			//result.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, src.cols - 1 - j)[0];
			//result.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, src.cols - 1 - j)[1];
			//result.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, src.cols - 1 - j)[2];
			result.at<uchar>(i, j) = src.at<uchar>(i, src.cols - 1 - j);
		}
	}
	return result;
}



void seamcut(Mat &canvas1, Mat &canvas2)
{
	Mat image1 = canvas1(Range::all(), Range(0, canvas1.cols / 2));
	Mat image2 = canvas2(Range::all(), Range(canvas2.cols / 4, canvas2.cols * 3 / 4));//假设大概1/2重复区域
																	  //image2.convertTo(image2, CV_32FC3);
	image1.convertTo(image1, CV_32FC3);
	image2.convertTo(image2, CV_32FC3);
	image1 /= 255.0;
	image2 /= 255.0;

	//在找拼缝的操作中，为了减少计算量，用image_small
	Mat image1_small;
	Mat image2_small;
	Size small_size1 = Size(image1.cols / 2, image1.rows / 2);
	Size small_size2 = Size(image2.cols / 2, image2.rows / 2);
	resize(image1, image1_small, small_size1);
	resize(image2, image2_small, small_size2);

	// 左图的左上角坐标
	cv::Point corner1;
	corner1.x = 0;
	corner1.y = 0;

	//右图的左上角坐标
	cv::Point corner2;
	corner2.x = image2_small.cols / 2;
	corner2.y = 0;

	std::vector<cv::Point> corners;

	corners.push_back(corner1);
	corners.push_back(corner2);

	std::vector<cv::Mat> masks;
	Mat imageMask1(small_size1, CV_8U);
	Mat imageMask2(small_size2, CV_8U);
	imageMask1 = Scalar::all(255);
	imageMask2 = Scalar::all(255);

	masks.push_back(imageMask1);
	masks.push_back(imageMask2);

	std::vector<cv::Mat> sources;

	sources.push_back(image1_small);
	sources.push_back(image2_small);

	Ptr<SeamFinder> seam_finder = new cv::detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	seam_finder->find(sources, corners, masks);

	//将mask恢复放大
	resize(masks[0], imageMask1, image1.size());
	resize(masks[1], imageMask2, image2.size());

	Mat canvas(image1.rows, image1.cols * 3 / 2, CV_32FC3);
	image1.copyTo(canvas(Range::all(), Range(0, canvas.cols * 2 / 3)), imageMask1);
	image2.copyTo(canvas(Range::all(), Range(canvas.cols / 3, canvas.cols)), imageMask2);
	//canvas *= 255;
	//canvas.convertTo(canvas, CV_8UC3);
	

	//imshow("Mask1", masks[0]);
	//imshow("Mask2", masks[1]);

	//imshow("src1", sources[0]);
	//imshow("src2", sources[1]);
	//canvas.convertTo(canvas, CV_8UC1, 255, 0);
	//imwrite("waterCubic_inside7\\canvas.jpg", canvas);
	//imshow("canvas", canvas);
	//waitKey(0);


}

//typedef struct
//{
//	Point2f left_top;
//	Point2f left_bottom;
//	Point2f right_top;
//	Point2f right_bottom;
//}four_corners_t;
//
//four_corners_t corners;
//
//
//void CalcCorners(const Mat& H, const Mat& src)
//{
//	double v2[] = { 0, 0, 1 };//左上角
//	double v1[3];//变换后的坐标值
//	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//
//	V1 = H * V2;
//	//左上角(0,0,1)
//	cout << "V2: " << V2 << endl;
//	cout << "V1: " << V1 << endl;
//	corners.left_top.x = v1[0] / v1[2];
//	corners.left_top.y = v1[1] / v1[2];
//
//	//左下角(0,src.rows,1)
//	v2[0] = 0;
//	v2[1] = src.rows;
//	v2[2] = 1;
//	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//	V1 = H * V2;
//	corners.left_bottom.x = v1[0] / v1[2];
//	corners.left_bottom.y = v1[1] / v1[2];
//
//	//右上角(src.cols,0,1)
//	v2[0] = src.cols;
//	v2[1] = 0;
//	v2[2] = 1;
//	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//	V1 = H * V2;
//	corners.right_top.x = v1[0] / v1[2];
//	corners.right_top.y = v1[1] / v1[2];
//
//	//右下角(src.cols,src.rows,1)
//	v2[0] = src.cols;
//	v2[1] = src.rows;
//	v2[2] = 1;
//	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//	V1 = H * V2;
//	corners.right_bottom.x = v1[0] / v1[2];
//	corners.right_bottom.y = v1[1] / v1[2];
//
//}
//
//
//
//
//
//void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
//{
//	int start = min(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界  
//
//	double processWidth = img1.cols - start;//重叠区域的宽度  
//	int rows = dst.rows;
//	int cols = img1.cols; //注意，是列数*通道数
//	double alpha = 1;//img1中像素的权重  
//	for (int i = 0; i < rows; i++)
//	{
//		uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
//		uchar* t = trans.ptr<uchar>(i);
//		uchar* d = dst.ptr<uchar>(i);
//		for (int j = start; j < cols; j++)
//		{
//			//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
//			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
//			{
//				alpha = 1;
//			}
//			else
//			{
//				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
//				alpha = (processWidth - (j - start)) / processWidth;
//			}
//
//			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
//			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
//			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
//
//		}
//	}
//
//}