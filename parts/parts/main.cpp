#include <stdio.h>
#include <iostream>
#include<opencv2/opencv.hpp>  
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <vector> 
#include "fast_grid.h"
#include "grid_matcher.h"
//#include "fundamental_matcher.h"
#include "gms_matcher.h"
#include "func.h"
#include "homo_matcher.h"
#include <future>
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include <string>
#define DEBUG_WRITEblend
//#define DEBUG_WRITE
//#define DEBUG_EVALUTION

using namespace cv;
using namespace std;
const float HARRIS_K = 0.04f;

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

four_corners_t fourcorners;
string path = "playground2\\";
string data_path = path + "data.txt";


void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//���Ͻ�
	double v1[3];//�任�������ֵ
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //������
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //������

	V1 = H * V2;
	//���Ͻ�(0,0,1)
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	fourcorners.left_top.x = v1[0] / v1[2];
	fourcorners.left_top.y = v1[1] / v1[2];

	//���½�(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	fourcorners.left_bottom.x = v1[0] / v1[2];
	fourcorners.left_bottom.y = v1[1] / v1[2];

	//���Ͻ�(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	fourcorners.right_top.x = v1[0] / v1[2];
	fourcorners.right_top.y = v1[1] / v1[2];

	//���½�(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	fourcorners.right_bottom.x = v1[0] / v1[2];
	fourcorners.right_bottom.y = v1[1] / v1[2];

}


void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(fourcorners.left_top.x, fourcorners.left_bottom.x);//��ʼλ�ã����ص��������߽�  

	double processWidth = img1.cols - start;//�ص�����Ŀ��  
	int rows = dst.rows;
	int cols = img1.cols; //ע�⣬������*ͨ����
	double alpha = 1;//img1�����ص�Ȩ��  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//�������ͼ��trans�������صĺڵ㣬����ȫ����img1�е�����
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1�����ص�Ȩ�أ��뵱ǰ�������ص�������߽�ľ�������ȣ�ʵ��֤�������ַ���ȷʵ��  
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}


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

//��Ӧ����
void main(int argc, char **argv)
{
	
	Mat img1, img2;

	//img1 = imread("waterCubic_inside2//2.JPG", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("waterCubic_inside2//1.JPG", CV_LOAD_IMAGE_GRAYSCALE);
	img1 = imread(path+"1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	img2 = imread(path+"2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("input_0.png", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("input_1.png", CV_LOAD_IMAGE_GRAYSCALE);

	//img1 = imread("15\\01.png", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("15\\02.png", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("3\\2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("3\\1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("1111\\2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("1111\\1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("leuven\\img1.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("leuven\\img2.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("leuven\\img1.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("leuven\\img5.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("F://dataset//herzjesu_images//herzjesu//0002.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("F://dataset//herzjesu_images//herzjesu//0007.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("adam1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("adam2.png", CV_LOAD_IMAGE_GRAYSCALE);
	//img1 = imread("temple1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("temple2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	
	imresize(img1, 480);
	imresize(img2, 480);
	
	double Time = (double)cvGetTickCount();
	
	std::vector<KeyPoint> keyPoints1, keyPoints2;
	// construction of the fast feature detector object  
	
	//Time = (double)cvGetTickCount();
	int constract=80;
	FastFeatureDetector fast(constract);   // ������ֵΪ80  
									// feature point detection  
	//OrbFeatureDetector fast(1500);

	fast.detect(img1, keyPoints1);
	//fast.detect(img2, keyPoints2);
	int keypointNum = 1000;
	while(keyPoints1.size() <keypointNum)
	{
		int distance = keypointNum - keyPoints1.size();
		keyPoints1.clear();
		//keyPoints2.clear();
		//constract -= 20;
		//constract -= (5 * 2 / 3.14*atan(distance/1000) + 2);
		if(distance>1500)
			constract -= (0.05*distance);
		else if(distance>=1000)
			constract -= (0.02*distance );
		else if (distance>=500)
			constract -= (0.01*distance );
		else
			constract= constract-3;
		
		FastFeatureDetector fast1(constract);   // ������ֵΪ80  
		
		fast1.detect(img1, keyPoints1);
		//fast1.detect(img2, keyPoints2);
		//cout << "constract" << constract << endl;
		//cout << "keyPoints1 size:   " << keyPoints1.size() << endl;
	}
	FastFeatureDetector fast2(constract);
	fast2.detect(img2, keyPoints2);
	
	
	HarrisResponses(img1, keyPoints1, 5, HARRIS_K);
	HarrisResponses(img2, keyPoints2, 5, HARRIS_K);

	fast_grid fastgrid1(keyPoints1, img1.cols, img1.rows, 10,10);
	fast_grid fastgrid2(keyPoints2, img2.cols, img1.rows,10, 10);
	
	KeyPointsFilter::runByImageBorder(fastgrid1.keypoint_result, img1.size(),31);
	//KeyPointsFilter::runByKeypointSize(fastgrid1.keypoint_result, std::numeric_limits<float>::epsilon());
	KeyPointsFilter::runByImageBorder(fastgrid2.keypoint_result, img2.size(),31);
	
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
	
	Mat m_hommography;
	vector<DMatch> matches_back_big;
	//calcHomo(fastgrid1.keypoint_result, fastgrid2.keypoint_result, good_matches, m_hommography);
	calcHomo(fastgrid1.keypoint_result, fastgrid2.keypoint_result, identical_matches, m_hommography);
	Mat h_ransac=grid_match_Homo(img1, img2,keyPoints1,keyPoints2, m_hommography);//���
	Time = (double)cvGetTickCount() - Time;
	printf("������ time = %gms\n", Time / (cvGetTickFrequency() * 1000));//����
	printf("������ time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//����
	ofstream f1(data_path, ios::app);
	if (!f1)return ;
	f1 << "---------------------------���Ƿָ���-----------------------" << endl;
	f1 << "ƥ�� time:" << Time / (cvGetTickFrequency() * 1000) << "ms" << endl;
	f1 << "ƥ�� time:" << Time / (cvGetTickFrequency() * 1000000) << "s" << endl;

	/*������ѷ����*/
	/*
	Time = (double)cvGetTickCount();
	//Mat h_ransac=ransac(img1, img2, keyPoints1, keyPoints2, matches_back_big);

	//�Ҷ�ͼ
	Mat shftMat = (Mat_<double>(3, 3) << 1.0, 0, img1.cols, 0, 1.0, 0, 0, 0, 1.0);

	//ƴ��ͼ��

	Mat tiledImg;

	warpPerspective(img1, tiledImg, shftMat*h_ransac, Size(img1.cols + img2.cols, img2.rows));
	//imwrite("tiled_warp.jpg", tiledImg);

	Mat right = mirror(tiledImg);
	Size size = Size(1460, right.rows);
	resize(right, right, size);
	//imwrite("right.jpg", right);
	img2.copyTo(Mat(tiledImg, Rect(img1.cols, 0, img2.cols, img2.rows)));
	//����ͼ��

	//imwrite("tiled.jpg", tiledImg);
	//��ʾƴ�ӵ�ͼ��


	//imshow("tiled image", tiledImg);

	Mat left = mirror(tiledImg);

	resize(left, left, size);
	//imwrite("left.jpg", left);
	
	Mat img1_seam;
	Mat img2_seam;
	cv::cvtColor(left, img1_seam, CV_GRAY2RGB);
	cv::cvtColor(right, img2_seam, CV_GRAY2RGB);
	//Mat img1_seam = imread("left.jpg");
	//Mat img2_seam = imread("right.jpg");
	//Time = (double)cvGetTickCount();
	//seamcut(left, right);
	seamcut(img1_seam, img2_seam);
	Time = (double)cvGetTickCount() - Time;
	printf("pinjie time = %gms\n", Time / (cvGetTickFrequency() * 1000));//����
	printf("pinjie time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//����
	//waitKey(0);
	*/
  //�ನ���ں�

	Time = (double)cvGetTickCount();
	CalcCorners(m_hommography, img1);
	cout << "left_top:" << fourcorners.left_top << endl;
	cout << "left_bottom:" << fourcorners.left_bottom << endl;
	cout << "right_top:" << fourcorners.right_top << endl;
	cout << "right_bottom:" << fourcorners.right_bottom << endl;


	cv::cvtColor(img1, img1, CV_GRAY2RGB);
	cv::cvtColor(img2, img2, CV_GRAY2RGB);
	Mat imageTransform1;
	warpPerspective(img1, imageTransform1, m_hommography, Size(MAX(fourcorners.right_top.x, fourcorners.right_bottom.x), img2.rows));
	////warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
	////imshow("ֱ�Ӿ���͸�Ӿ���任", imageTransform1);
	//imwrite("trans1.jpg", imageTransform1);

	Mat imageTransform2(imageTransform1.rows, imageTransform1.cols, CV_8UC3);
	imageTransform2(Rect(0, 0, imageTransform2.cols, imageTransform2.rows)).setTo(0);
	img2.copyTo(imageTransform2(Range::all(), Range(0, img2.cols)));


	cv::Point corner1;
	corner1.x = 0;
	corner1.y = 0;

	//��ͼ�����Ͻ�����
	cv::Point corner2;
	corner2.x = fourcorners.left_top.x;
	corner2.y = 0;

	std::vector<cv::Point> corners;

	corners.push_back(corner1);
	corners.push_back(corner2);
	std::vector<cv::Point> corners_another;
	corners_another.push_back(corner1);
	corners_another.push_back(corner1);
	vector<Mat> masks_warped;
	vector<Mat> images_warped;

	images_warped.push_back(imageTransform2);
	images_warped.push_back(imageTransform1);


	std::vector<cv::Mat> masks;
	Mat mask1, mask2;
	mask1.create(imageTransform1.size(), CV_8U);
	mask1.setTo(Scalar::all(0));
	mask1(Rect(0, 0, img2.cols, img2.rows)).setTo(255);
	mask2.create(imageTransform1.size(), CV_8U);
	mask2.setTo(Scalar::all(255));
	mask2(Rect(0, 0, fourcorners.left_top.x, img2.rows)).setTo(0);
	masks.push_back(mask1);
	masks.push_back(mask2);
	//imwrite("masks1.jpg", masks[0]);
	//imwrite("masks2.jpg", masks[1]);

	masks_warped.push_back(mask1);
	masks_warped.push_back(mask2);
	Ptr<ExposureCompensator> compensator =
		ExposureCompensator::createDefault(ExposureCompensator::GAIN);
	vector<Mat> images_warped_f(2);
	for (int i = 0; i < 2; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32FC3);
	//sources.push_back(imageTransform2);
	//sources.push_back(imageTransform1);
	Ptr<SeamFinder> seam_finder = new cv::detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	seam_finder->find(images_warped_f, corners, masks_warped);
	//imwrite("masks11.jpg", masks_warped[0]);
	//imwrite("masks22.jpg", masks_warped[1]);
	bitwise_not(masks_warped[1], masks_warped[0]);

	//imwrite("masks14.jpg", masks_warped[0]);
	//imwrite("masks24.jpg", masks_warped[1]);
	Mat canvas(imageTransform1.rows, imageTransform1.cols, CV_8UC3);
	imageTransform2.copyTo(canvas, masks_warped[0]);
	imageTransform1.copyTo(canvas, masks_warped[1]);

#ifdef DEBUG_WRITEblend
	imwrite(path+"proposed_canvas.jpg", canvas);
	//ֻ�ڷ���߸������ں�
#endif
	//imageTransform2.convertTo(imageTransform2, CV_16S);
	//imageTransform1.convertTo(imageTransform1, CV_16S);
	images_warped[0].convertTo(images_warped[0], CV_16S);
	images_warped[1].convertTo(images_warped[1], CV_16S);
	Mat multibandResult, maskFinal;
	Rect dst(0, 0, imageTransform1.cols, imageTransform1.rows);
	detail::MultiBandBlender blender;
	//detail::FeatherBlender blender;
	//detail::Blender blender;
	blender.prepare(dst);
	Mat dilated_mask, seam_mask, mask, mask_warped;
	Mat element = getStructuringElement(MORPH_RECT, Size(50, 50));

	for (int i = 0; i < 2; i++)
	{
		dilate(masks_warped[i], dilated_mask, element);
		resize(dilated_mask, seam_mask, masks_warped[i].size());
		mask_warped = seam_mask & masks_warped[i];
		//imwrite(path + "mask_warped.jpg", mask_warped);
		//imwrite(path + "images_warped.jpg", images_warped[i]);
		blender.feed(images_warped[i], mask_warped, corners[0]);
		
	}
	blender.blend(multibandResult, maskFinal);



	//otherwise, result is a gray image
	multibandResult.convertTo(multibandResult, (multibandResult.type() / 8) * 8);
#ifdef DEBUG_WRITEblend
	imwrite(path+"proposed_blend.jpg", multibandResult);
#endif

	Time = (double)cvGetTickCount() -Time;
	printf("ƴ�� time = %gms\n", Time / (cvGetTickFrequency() * 1000));//����
	printf("ƴ�� time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//����

	f1 << "ƴ�� time:" << Time / (cvGetTickFrequency() * 1000) << "ms" << endl;
	f1 << "ƴ�� time:" << Time / (cvGetTickFrequency() * 1000000) << "s" << endl;
	f1.close();
}


