/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "_modelest_jw.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <iostream>
#include <ppl.h>
#include <windows.h>
#include <unordered_map>
using namespace std;
using namespace cv;
namespace jw{
CvModelEstimator2::CvModelEstimator2(int _modelPoints, CvSize _modelSize, int _maxBasicSolutions)
{
    modelPoints = _modelPoints;
    modelSize = _modelSize;
    maxBasicSolutions = _maxBasicSolutions;
    checkPartialSubsets = true;
    rng = cvRNG(-1);
}

CvModelEstimator2::~CvModelEstimator2()
{
}

void CvModelEstimator2::setSeed( int64 seed )
{
    rng = cvRNG(seed);
}


int CvModelEstimator2::findInliers( const CvMat* m1, const CvMat* m2,
                                    const CvMat* model, CvMat* _err,
                                    CvMat* _mask, double threshold )
{
    int i, count = _err->rows*_err->cols, goodCount = 0;
    const float* err = _err->data.fl;
    uchar* mask = _mask->data.ptr;
	
    computeReprojError( m1, m2, model, _err );
	
    threshold *= threshold;
    for( i = 0; i < count; i++ )
        goodCount += mask[i] = err[i] <= threshold;
    return goodCount;
}


 int
cvRANSACUpdateNumIters( double p, double ep,
                        int model_points, int max_iters )
{
    if( model_points <= 0 )
        CV_Error( CV_StsOutOfRange, "the number of model points should be positive" );

    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    double num = MAX(1. - p, DBL_MIN);
    double denom = 1. - pow(1. - ep,model_points);
    if( denom < DBL_MIN )
        return 0;

    num = log(num);
    denom = log(denom);

    return denom >= 0 || -num >= max_iters*(-denom) ?
        max_iters : cvRound(num/denom);
}

double calcArea(vector<Point2f> &m_corners)
{
	double d01 = sqrt((m_corners[0].x - m_corners[1].x)*(m_corners[0].x - m_corners[1].x) +
		(m_corners[0].y - m_corners[1].y)*(m_corners[0].y - m_corners[1].y));
	double	d12 = sqrt((m_corners[2].x - m_corners[1].x)*(m_corners[2].x - m_corners[1].x) +
		(m_corners[2].y - m_corners[1].y)*(m_corners[2].y - m_corners[1].y));
	double	d23 = sqrt((m_corners[2].x - m_corners[3].x)*(m_corners[2].x - m_corners[3].x) +
		(m_corners[2].y - m_corners[3].y)*(m_corners[2].y - m_corners[3].y));
	double	d03 = sqrt((m_corners[0].x - m_corners[3].x)*(m_corners[0].x - m_corners[3].x) +
		(m_corners[0].y - m_corners[3].y)*(m_corners[0].y - m_corners[3].y));
	double	d13 = sqrt((m_corners[1].x - m_corners[3].x)*(m_corners[1].x - m_corners[3].x) +
		(m_corners[1].y - m_corners[3].y)*(m_corners[1].y - m_corners[3].y));
	double	k1 = (d01 + d03 + d13) / 2;
	double	k2 = (d12 + d23 + d13) / 2;
	double	s1 = (k1*(k1 - d01)*(k1 - d03)*(k1 - d13));
	double	s2 = (k2*(k2 - d12)*(k2 - d23)*(k2 - d13));
	double	s = sqrt(s1) + sqrt(s2);
	return s;
}

double calcCircumference(vector<Point2f> &m_corners)
{
	double d01 = sqrt((m_corners[0].x - m_corners[1].x)*(m_corners[0].x - m_corners[1].x) +
		(m_corners[0].y - m_corners[1].y)*(m_corners[0].y - m_corners[1].y));
	double	d12 = sqrt((m_corners[2].x - m_corners[1].x)*(m_corners[2].x - m_corners[1].x) +
		(m_corners[2].y - m_corners[1].y)*(m_corners[2].y - m_corners[1].y));
	double	d23 = sqrt((m_corners[2].x - m_corners[3].x)*(m_corners[2].x - m_corners[3].x) +
		(m_corners[2].y - m_corners[3].y)*(m_corners[2].y - m_corners[3].y));
	double	d03 = sqrt((m_corners[0].x - m_corners[3].x)*(m_corners[0].x - m_corners[3].x) +
		(m_corners[0].y - m_corners[3].y)*(m_corners[0].y - m_corners[3].y));
	double c=d01+d12+d23+d03;
	return c;
}

void calcBianchang(vector<Point2f> &m_corners, vector<int> &m_cornersBianchang)
{
	double d01 = sqrt((m_corners[0].x - m_corners[1].x)*(m_corners[0].x - m_corners[1].x) +
		(m_corners[0].y - m_corners[1].y)*(m_corners[0].y - m_corners[1].y));
	double	d12 = sqrt((m_corners[2].x - m_corners[1].x)*(m_corners[2].x - m_corners[1].x) +
		(m_corners[2].y - m_corners[1].y)*(m_corners[2].y - m_corners[1].y));
	double	d23 = sqrt((m_corners[2].x - m_corners[3].x)*(m_corners[2].x - m_corners[3].x) +
		(m_corners[2].y - m_corners[3].y)*(m_corners[2].y - m_corners[3].y));
	double	d03 = sqrt((m_corners[0].x - m_corners[3].x)*(m_corners[0].x - m_corners[3].x) +
		(m_corners[0].y - m_corners[3].y)*(m_corners[0].y - m_corners[3].y));
	m_cornersBianchang[0] = d01;
	m_cornersBianchang[1] = d12;
	m_cornersBianchang[2] = d23;
	m_cornersBianchang[3] = d03;

	

}

 bool checkHash(CvMat* model, vector<vector<Point2f>> &mv_cornersRight,vector<Point2f> &m_corners)
{
	//cout << "checkHash**********************" << endl;
	//vector<Point2f> m_corners;
	//vector<vector<int>> m_homos;
	//vector<vector<Point2f>> mv_cornersRight;
	vector<Point2f> m_cornersRight(4);
	
	Mat homo(model);
	perspectiveTransform(m_corners, m_cornersRight, homo);
	//double recArea= calcArea(m_cornersRight);
	//double reccircumference = calcCircumference(m_cornersRight);
	//vector<int> m_cornersBianchang(4);
	//calcBianchang(m_cornersRight, m_cornersBianchang);
	//cout << "rec :   " << recArea << endl;
	//cout << "reccircumference :   " << reccircumference << endl;
	//concurrency::parallel_for(size_t(0), mv_cornersRight.size(), [&](size_t j)
	for (int j = 0; j < mv_cornersRight.size(); ++j)
	{
		//int sum = 0;
		//for (int k = 0; k < 4; k++)
		//{
		//	sum += abs(mv_cornersRight[j][k].x - m_cornersRight[k].x) + abs(mv_cornersRight[j][k].y - m_cornersRight[k].y);
		//	if (sum > 30)
		//	{
		//		//mv_cornersRight.push_back(m_cornersRight);
		//		break;
		//	}

		//}


		int sum, sum0, sum1, sum2, sum3;
		//int sum = 0,sum0=0, sum1 = 0, sum2 = 0, sum3 = 0;
		sum0= abs(mv_cornersRight[j][0].x - m_cornersRight[0].x) + abs(mv_cornersRight[j][0].y - m_cornersRight[0].y);
		sum1 = abs(mv_cornersRight[j][1].x - m_cornersRight[1].x) + abs(mv_cornersRight[j][1].y - m_cornersRight[1].y);
		sum2 = abs(mv_cornersRight[j][2].x - m_cornersRight[2].x) + abs(mv_cornersRight[j][2].y - m_cornersRight[2].y);
		sum3 = abs(mv_cornersRight[j][3].x - m_cornersRight[3].x) + abs(mv_cornersRight[j][3].y - m_cornersRight[3].y);
		sum = sum0 + sum1 + sum2 + sum3;
		if (sum <= 30)
		{
			//cout << "当前 " << "与" << j << "  冲突" << endl;
			//cout << homo << endl;
			//cout << " succeed rec :   " << recArea << endl;
			//cout << " succeed reccircumference :   " << reccircumference << endl;
			//cout << "rec :   " << homo << endl;
			mv_cornersRight.insert(mv_cornersRight.begin(),m_cornersRight);
			return false;
		}

	}
	mv_cornersRight.push_back(m_cornersRight);
	return true;

}

 inline void  perspectiveTransform_jw(CvMat* model, vector<Point2f> &m_corners, vector<Point> &m_cornersRight)
 {
	 const double* F = model->data.db;
	 double a, b, c;
	 for (int i = 0; i < 4; ++i)
	 {
		 a = F[0] * m_corners[i].x + F[1] * m_corners[i].y + F[2];
		 b = F[3] * m_corners[i].x + F[4] * m_corners[i].y + F[5];
		 c = F[6] * m_corners[i].x + F[7] * m_corners[i].y + F[8];

		 m_cornersRight[i].x = a / c;
		 m_cornersRight[i].y = b / c;

	 }
	
	
 }

 bool checkHash5(CvMat* model, vector<vector<Point>> &mv_cornersRight, vector<vector<Point>> &mv_cornershash, vector<Point2f> &m_corners)
 {
	
	 vector<Point> m_cornersRight(4);
	
	 //perspectiveTransform(m_corners, m_cornersRight, homo);
	 perspectiveTransform_jw(model, m_corners, m_cornersRight);
	 double Time = (double)cvGetTickCount();
	 for (int j = 0; j < mv_cornershash.size(); ++j)
	 {
		 int sum = 0;
		 for (int k = 0; k < 4; ++k)
		 {
		 	sum += abs(mv_cornershash[j][k].x - m_cornersRight[k].x) + abs(mv_cornershash[j][k].y - m_cornersRight[k].y);
		 	if (sum > 30)
		 	{
		 		//mv_cornersRight.push_back(m_cornersRight);
		 		break;
		 	}
		 }

		 if (sum <= 30)
		 {
			 mv_cornershash.push_back(m_cornersRight);
			 return false;
		 }

	 }
	 Time = (double)cvGetTickCount() - Time;
	 if (Time > 0.001)
	 {
		 mv_cornersRight.push_back(m_cornersRight);
		 return false;
	 }
		 
	 for (int j = 0; j < mv_cornersRight.size(); ++j)
	 {
		 int sum = 0;
		 for (int k = 0; k < 4; ++k)
		 {
		 	sum += abs(mv_cornersRight[j][k].x - m_cornersRight[k].x) + abs(mv_cornersRight[j][k].y - m_cornersRight[k].y);
		 	if (sum > 30)
		 	{
		 		//mv_cornersRight.push_back(m_cornersRight);
		 		break;
		 	}

		 }
		 if (sum <= 30)
		 {
			 
			 mv_cornershash.push_back(m_cornersRight);
			 return false;
		 }

		 
	 }
	 mv_cornersRight.push_back(m_cornersRight);
	 return true;

 }


 bool checkHash6(CvMat* model, multimap<int, vector<Point>> &mapCornersRight, vector<Point2f> &m_corners)
 {

	 vector<Point> m_cornersRight(4);

	 //perspectiveTransform(m_corners, m_cornersRight, homo);
	 perspectiveTransform_jw(model, m_corners, m_cornersRight);

	 //multimap<int, vector<Point>> mapCornersRight;
	 //for (int j = 0; j < mv_cornershash.size(); ++j)
	 //multimap<int, vector<Point>>::iterator it = mapCornersRight.begin();
	 //multimap<int, vector<Point>>::iterator beg, end;
	 //unordered_multimap<int, vector<Point>>::iterator beg, end;
	 auto beg = mapCornersRight.lower_bound(m_cornersRight[0].x);
	 auto end = mapCornersRight.upper_bound(m_cornersRight[0].x);
	 for (auto m = beg; m != end; m++)
	 //auto it = mapCornersRight.find(m_cornersRight[0].x);
	 //for (int i = 0, len = mapCornersRight.count(m_cornersRight[0].x); i < len; ++i, ++it)
		        // cout << m->first << "--" << m->second << endl
	 {
		 int sum = 0;
		 for (int k = 0; k < 4; ++k)
		 {
			 sum += abs((*m).second[k].x - m_cornersRight[k].x) + abs((*m).second[k].y - m_cornersRight[k].y);
			 if (sum > 30)
			 {
				 //mv_cornersRight.push_back(m_cornersRight);
				 break;
			 }
		 }

		 if (sum <= 30)
		 {
			 //mv_cornershash.push_back(m_cornersRight);
			 mapCornersRight.insert(make_pair(m_cornersRight[0].x, m_cornersRight));
			 return false;
		 }
		 //return false;

	 }
	// mv_cornersRight.push_back(m_cornersRight);
	 mapCornersRight.insert(make_pair(m_cornersRight[0].x, m_cornersRight));
	 return true;

 }




 bool checkHash4(CvMat* model, vector<vector<Point2f>> &mv_cornersRight, vector<Point2f> &m_corners)
 {
	 return false;
 
 }

bool checkHash1(CvMat* model, vector<vector<Point2f>> &mv_cornersRight,vector<Point2f> &m_corners)
{
	//cout << "checkHash**********************" << endl;
	//vector<Point2f> m_corners;
	//vector<vector<int>> m_homos;
	//vector<vector<Point2f>> mv_cornersRight;
	vector<Point2f> m_cornersRight(4);
	/*int height = 480;
	int weight = 640;
	m_corners.push_back(Point2f(0.2*height, 0.2*weight));
	m_corners.push_back(Point2f(0.8*height, 0.2*weight));
	m_corners.push_back(Point2f(0.2*height, 0.8*weight));
	m_corners.push_back(Point2f(0.8*height, 0.8*weight));*/
	Mat homo(model);
	perspectiveTransform(m_corners, m_cornersRight, homo);
	if (m_cornersRight[0].x + 1500 > 2994|| m_cornersRight[0].x + 1500 < 6) return false;
	//for (int j = -6; j < 6; j++)
	//{
	//	if (mv_cornersRight[j + m_cornersRight[0].x + 1500].size() == 0)
	//		continue;
	//	int sum = 0;
	//	for (int k = 0; k < 4; k++)
	//	{
	//		sum += abs(mv_cornersRight[j+ m_cornersRight[0].x+1500][k].x - m_cornersRight[k].x) + abs(mv_cornersRight[j + m_cornersRight[0].x+1500][k].y - m_cornersRight[k].y);
	//		if (sum > 30)
	//		{
	//			//mv_cornersRight.push_back(m_cornersRight);
	//			break;
	//		}

	//	}
	//	if (sum < 30)
	//	{
	//		//cout << "当前 " << "与" << j << "  冲突" << endl;
	//		//cout << homo << endl;

	//		//mv_cornersRight.push_back(m_cornersRight);
	//		return true;
	//	}

	//}
	//mv_cornersRight.push_back(m_cornersRight);
	if(mv_cornersRight[m_cornersRight[0].x + 1500].size()==0)
		mv_cornersRight[m_cornersRight[0].x + 1500] = m_cornersRight;

	return mv_cornersRight[m_cornersRight[0].x + 1500 - 6].size() + mv_cornersRight[m_cornersRight[0].x + 1500 - 5].size() +
		mv_cornersRight[m_cornersRight[0].x + 1500 - 4].size() + mv_cornersRight[m_cornersRight[0].x + 1500 - 3].size()
		+ mv_cornersRight[m_cornersRight[0].x + 1500 - 2].size() + mv_cornersRight[m_cornersRight[0].x + 1500 - 1].size()
		+ mv_cornersRight[m_cornersRight[0].x + 1500].size() + mv_cornersRight[m_cornersRight[0].x + 1500 + 1].size()
		+ mv_cornersRight[m_cornersRight[0].x + 1500 + 2].size() + mv_cornersRight[m_cornersRight[0].x + 1500 + 3].size()
		+ mv_cornersRight[m_cornersRight[0].x + 1500 + 3].size() + mv_cornersRight[m_cornersRight[0].x + 1500 + 4].size();
	//return false;

}

bool checkHash2(CvMat* model, vector<vector<int>> &mv_cornersRight, vector<Point2f> &m_corners)
{
	//cout << "checkHash**********************" << endl;
	//vector<Point2f> m_corners;
	//vector<vector<int>> m_homos;
	//vector<vector<Point2f>> mv_cornersRight;
	vector<Point2f> m_cornersRight(4);
	/*int height = 480;
	int weight = 640;
	m_corners.push_back(Point2f(0.2*height, 0.2*weight));
	m_corners.push_back(Point2f(0.8*height, 0.2*weight));
	m_corners.push_back(Point2f(0.2*height, 0.8*weight));
	m_corners.push_back(Point2f(0.8*height, 0.8*weight));*/
	Mat homo(model);
	perspectiveTransform(m_corners, m_cornersRight, homo);
	double recArea = calcArea(m_cornersRight);
	double reccircumference = calcCircumference(m_cornersRight);
	vector<int> m_cornersBianchang(4);
	calcBianchang(m_cornersRight, m_cornersBianchang);
	//cout << "rec :   " << recArea << endl;
	//cout << "reccircumference :   " << reccircumference << endl;
	for (int j = 0; j < mv_cornersRight.size(); j++)
	{
		int sum = 0;
		for (int k = 0; k < 4; k++)
		{
			sum += abs(mv_cornersRight[j][k] - m_cornersBianchang[k]) ;
			if (sum > 50)
			{
				//mv_cornersRight.push_back(m_cornersRight);
				break;
			}

		}
		if (sum <= 50)
		{
			//cout << "当前 " << "与" << j << "  冲突" << endl;
			//cout << homo << endl;
			//cout << " succeed rec :   " << recArea << endl;
			//cout << " succeed reccircumference :   " << reccircumference << endl;
			//cout << "rec :   " << homo << endl;
			//cout << " succeed sum:   " << sum << endl;
			//cout << " succeed bianchang :   " << m_cornersBianchang[0]<<"   "<< m_cornersBianchang[1] << "   " << m_cornersBianchang[2] << "   " << m_cornersBianchang[3] << endl;
			mv_cornersRight.insert(mv_cornersRight.begin(), m_cornersBianchang);
			return true;
		}

	}
	mv_cornersRight.push_back(m_cornersBianchang);
	return false;

}





//bool CvModelEstimator2::runRANSAC( const CvMat* m1, const CvMat* m2, CvMat* model,
//                                    CvMat* mask0, double reprojThreshold,
//                                    double confidence, int maxIters )
//{
//    bool result = false;
//    cv::Ptr<CvMat> mask = cvCloneMat(mask0);
//    cv::Ptr<CvMat> models, err, tmask;
//    cv::Ptr<CvMat> ms1, ms2;
//
//    int iter, niters = maxIters;
//    int count = m1->rows*m1->cols, maxGoodCount = 0;
//    CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );
//
//    if( count < modelPoints )
//        return false;
//
//    models = cvCreateMat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
//    err = cvCreateMat( 1, count, CV_32FC1 );
//    tmask = cvCreateMat( 1, count, CV_8UC1 );
//
//    if( count > modelPoints )
//    {
//        ms1 = cvCreateMat( 1, modelPoints, m1->type );
//        ms2 = cvCreateMat( 1, modelPoints, m2->type );
//    }
//    else
//    {
//        niters = 1;
//        ms1 = cvCloneMat(m1);
//        ms2 = cvCloneMat(m2);
//    }
//	vector<vector<Point>> mv_cornersRight;
//	vector<vector<Point>> mv_cornershash;
//	vector<vector<Point2f>> mv_cornersRight(3000);
//	vector<vector<int>> mv_cornersRight;
//	vector<vector<int>> vecmodels(8);
//
//	vector<Point2f> m_corners;
//	vector<Point> m_cornersRight(4);
//	int height = 480;
//	int weight = 640;
//	m_corners.push_back(Point2f(0.2*height, 0.2*weight));
//	m_corners.push_back(Point2f(0.8*height, 0.2*weight));
//	m_corners.push_back(Point2f(0.8*height, 0.8*weight));
//	m_corners.push_back(Point2f(0.2*height, 0.8*weight));
//	
//    for( iter = 0; iter < niters; iter++ )
//    {
//        int i, goodCount, nmodels;
//        if( count > modelPoints )
//        {
//            bool found = getSubset( m1, m2, ms1, ms2, 300 );
//            if( !found )
//            {
//                if( iter == 0 )
//                    return false;
//                break;
//            }
//        }
//
//		
//		
//        nmodels = runKernel( ms1, ms2, models );
//		perspectiveTransform_jw(models, m_corners, m_cornersRight);
//		for (int k = 0; k < 4; k++)
//	    {
//			vecmodels[2*k].push_back(m_cornersRight[k].x);
//			vecmodels[2*k +1].push_back(m_cornersRight[k].y);	
//		}
//
//		
//
//        if( nmodels <= 0 )
//            continue;
//		double Time = (double)cvGetTickCount();
//		if (checkHash5(models, mv_cornersRight, mv_cornershash, m_corners))
//		{
//			cout << "checkHash failed -------------------" << endl;
//			continue;
//		}
//		for (int k = 0; k < 4; k++)
//		{
//			vecmodels[2 * k].push_back(m_cornersRight[k].x);
//			vecmodels[2 * k + 1].push_back(m_cornersRight[k].y);
//		}
//
//		Time = (double)cvGetTickCount() - Time;
//		printf("checkHash time = %gms\n", Time / (cvGetTickFrequency() * 1000));//毫秒
//		printf("checkHash time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//毫秒
//
//		Time = (double)cvGetTickCount();
//        for( i = 0; i < nmodels; i++ )
//        {
//            CvMat model_i;
//            cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );
//            goodCount = findInliers( m1, m2, &model_i, err, tmask, reprojThreshold );
//
//            if( goodCount > MAX(maxGoodCount, modelPoints-1) )
//            {
//                std::swap(tmask, mask);
//                cvCopy( &model_i, model );
//                maxGoodCount = goodCount;
//                niters = cvRANSACUpdateNumIters( confidence,
//                    (double)(count - goodCount)/count, modelPoints, niters )/3;
//				cout << "niters"<<niters << endl;
//            }
//        }
//		if (mv_cornershash.size() >300)
//			break;
//
//		Time = (double)cvGetTickCount() - Time;
//		printf("nmodels time = %gms\n", Time / (cvGetTickFrequency() * 1000));//毫秒
//		printf("nmodels time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//毫秒
//    }
//
//
//	
//
//
//	for (int k = 0; k < 8; k++)
//	{
//		sort(vecmodels[k].begin(), vecmodels[k].end());
//	}
//
//	vector<Point2f> selectcorners(4);
//	int datasaize = vecmodels[0].size();
//	for (int i = 0; i < 4; i++)
//	{
//		selectcorners[i].x = vecmodels[2 * i][datasaize / 2];
//		selectcorners[i].y = vecmodels[2 * i + 1][vecmodels[2 * i + 1].size() / 2];
//	}
//	Mat affine = findHomography(m_corners, selectcorners);
//	cout << affine << endl;
//    if( maxGoodCount > 0 )
//    {
//        if( mask != mask0 )
//            cvCopy( mask, mask0 );
//        result = true;
//    }
//	result = true;
//    return result;
//}

void uniform(vector<Point> m_cornersRight)
{
	

}
int hash64byMod(unsigned long long key, unsigned char num_bits)
{
	// these are large (the largest) primes with 0:31 bits
	//选为质数，小于或者等于哈希表表长
	unsigned int largePrimes[32] = { 0, 0, 3, 7, 13, 31, 61, 127, 251, 509, 1021, 2039, 4093, 8191, 16381, 32749, 65521, 131071, 262139, 524287, 1048573,
		2097143, 4194301, 8388593, 16777213, 33554393, 67108859, 134217689, 268435399, 536870909, 1073741789, 2147483647 };

	return (unsigned int)(key % largePrimes[num_bits]);

}

void gouzaohash(vector<Point> m_cornersRight)
{
	int hash_64 = 0;//8个字节
	int* ptr8 = &hash_64;//指向一个字节
    *ptr8 = m_cornersRight[0].x;//每次赋值为一个字节
	unsigned int hash_val = hash64byMod(hash_64, 12);

}

bool CvModelEstimator2::runRANSAC(const CvMat* m1, const CvMat* m2, CvMat* model,
	CvMat* mask0, double reprojThreshold,
	double confidence, int maxIters)
{

	double Time = (double)cvGetTickCount();
	bool result = false;
	cv::Ptr<CvMat> mask = cvCloneMat(mask0);
	cv::Ptr<CvMat> models, err, tmask;
	cv::Ptr<CvMat> ms1, ms2;

	int iter, niters = maxIters;
	int count = m1->rows*m1->cols, maxGoodCount = 0;
	CV_Assert(CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask));

	if (count < modelPoints)
		return false;

	models = cvCreateMat(modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1);
	err = cvCreateMat(1, count, CV_32FC1);
	tmask = cvCreateMat(1, count, CV_8UC1);

	if (count > modelPoints)
	{
		ms1 = cvCreateMat(1, modelPoints, m1->type);
		ms2 = cvCreateMat(1, modelPoints, m2->type);
	}
	else
	{
		niters = 1;
		ms1 = cvCloneMat(m1);
		ms2 = cvCloneMat(m2);
	}
	vector<vector<Point>> mv_cornersRight;
	vector<vector<Point>> mv_cornershash;
	multimap<int, vector<Point>> mapCornersRight;
	mv_cornersRight.reserve(200);
	mv_cornershash.reserve(200);

	//vector<vector<Point2f>> mv_cornersRight(3000);
	//vector<vector<int>> mv_cornersRight;
	//vector<vector<int>> vecmodels(8);

	vector<Point2f> m_corners;
	vector<Point> m_cornersRight(4);
	int height = 480;
	int weight = 640;
	m_corners.push_back(Point2f(0.2*height, 0.2*weight));
	m_corners.push_back(Point2f(0.8*height, 0.2*weight));
	m_corners.push_back(Point2f(0.8*height, 0.8*weight));
	m_corners.push_back(Point2f(0.2*height, 0.8*weight));


	int hashtable[] = { 500,500,300,200,50,20,10,5,3,2,1,0 };

	for (iter = 0; iter < niters; iter++)
	{
		int i, goodCount, nmodels;
		if (count > modelPoints)
		{
			bool found = getSubset(m1, m2, ms1, ms2, 300);
			if (!found)
			{
				if (iter == 0)
					return false;
				break;
			}
		}

		nmodels = runKernel(ms1, ms2, models);
		perspectiveTransform_jw(models, m_corners, m_cornersRight);
		
		if (nmodels <= 0)
			continue;

		//Time = (double)cvGetTickCount();
		bool flag = checkHash5(models, mv_cornersRight, mv_cornershash, m_corners);
		//
		//Time = (double)cvGetTickCount() - Time;
		//printf("checkHash time = %gms\n", Time / (cvGetTickFrequency() * 1000));//毫秒
		//printf("checkHash time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//毫秒
		//cout << "flag " << flag << endl;
		////if (checkHash5(models, mv_cornersRight, mv_cornershash, m_corners))
		if (flag)
			continue;
		//if(checkHash1(models, mv_cornersRight, m_corners))
		//if (checkHash6(models, mapCornersRight, m_corners))
		
			//cout << "checkHash failed -------------------" << endl;
			
		
		

		Time = (double)cvGetTickCount();
		for (i = 0; i < nmodels; i++)
		{
			CvMat model_i;
			cvGetRows(models, &model_i, i*modelSize.height, (i + 1)*modelSize.height);
			//double Time = (double)cvGetTickCount();
			goodCount = findInliers(m1, m2, &model_i, err, tmask, reprojThreshold);


			
			
			//Time = (double)cvGetTickCount() - Time;
			//printf("findInliers time = %gms\n", Time / (cvGetTickFrequency() * 1000));//毫秒
			//printf("findInliers time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//毫秒
			if (goodCount > MAX(maxGoodCount, modelPoints - 1))
			{
				std::swap(tmask, mask);
				cvCopy(&model_i, model);
				maxGoodCount = goodCount;

				//niters = cvRANSACUpdateNumIters(confidence,
					//(double)(count - goodCount) / count, modelPoints, niters)+hashtable[goodCount/count*10];
				niters = cvRANSACUpdateNumIters(confidence,
					(double)(count - goodCount) / count, modelPoints, niters) ;
				//cout << "niters" << niters << endl;
				
			}

		}
		//Time = (double)cvGetTickCount() - Time;
		//printf("findinliner time = %gms\n", Time / (cvGetTickFrequency() * 1000));//毫秒
		//printf("findinliner time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//毫秒
		
	}





	
	if (maxGoodCount > 0)
	{
		if (mask != mask0)
			cvCopy(mask, mask0);
		result = true;
	}
	
	
	return result;
}



//
//bool CvModelEstimator2::runRANSAC(const CvMat* m1, const CvMat* m2, CvMat* model,
//	CvMat* mask0, double reprojThreshold,
//	double confidence, int maxIters)
//{
//
//	//double Time = (double)cvGetTickCount();
//	bool result = false;
//	cv::Ptr<CvMat> mask = cvCloneMat(mask0);
//	cv::Ptr<CvMat> models, err, tmask;
//	cv::Ptr<CvMat> ms1, ms2;
//
//	int iter, niters = maxIters;
//	int count = m1->rows*m1->cols, maxGoodCount = 0;
//	CV_Assert(CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask));
//
//	if (count < modelPoints)
//		return false;
//
//	models = cvCreateMat(modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1);
//	err = cvCreateMat(1, count, CV_32FC1);
//	tmask = cvCreateMat(1, count, CV_8UC1);
//
//	if (count > modelPoints)
//	{
//		ms1 = cvCreateMat(1, modelPoints, m1->type);
//		ms2 = cvCreateMat(1, modelPoints, m2->type);
//	}
//	else
//	{
//		niters = 1;
//		ms1 = cvCloneMat(m1);
//		ms2 = cvCloneMat(m2);
//	}
//	vector<vector<Point>> mv_cornersRight;
//	vector<vector<Point>> mv_cornershash;
//	//vector<vector<Point2f>> mv_cornersRight(3000);
//	//vector<vector<int>> mv_cornersRight;
//	vector<vector<int>> vecmodels(8);
//
//	vector<Point2f> m_corners;
//	vector<Point> m_cornersRight(4);
//	int height = 480;
//	int weight = 640;
//	m_corners.push_back(Point2f(0.2*height, 0.2*weight));
//	m_corners.push_back(Point2f(0.8*height, 0.2*weight));
//	m_corners.push_back(Point2f(0.8*height, 0.8*weight));
//	m_corners.push_back(Point2f(0.2*height, 0.8*weight));
//
//
//	int hashtable[] = { 500,500,300,200,50,20,10,5,3,2,1,0 };
//
//	for (iter = 0; iter < niters; iter++)
//	{
//		int i, goodCount, nmodels;
//		if (count > modelPoints)
//		{
//			bool found = getSubset(m1, m2, ms1, ms2, 300);
//			if (!found)
//			{
//				if (iter == 0)
//					return false;
//				break;
//			}
//		}
//
//
//
//		nmodels = runKernel(ms1, ms2, models);
//		perspectiveTransform_jw(models, m_corners, m_cornersRight);
//		for (int k = 0; k < 4; k++)
//		{
//			vecmodels[2 * k].push_back(m_cornersRight[k].x);
//			vecmodels[2 * k + 1].push_back(m_cornersRight[k].y);
//		}
//
//
//
//		if (nmodels <= 0)
//			continue;
//		//double Time = (double)cvGetTickCount();
//		if (checkHash5(models, mv_cornersRight, mv_cornershash, m_corners))
//			//if(checkHash1(models, mv_cornersRight, m_corners))
//		{
//			//cout << "checkHash failed -------------------" << endl;
//			continue;
//		}
//		/*for (int k = 0; k < 4; k++)
//		{
//		vecmodels[2 * k].push_back(m_cornersRight[k].x);
//		vecmodels[2 * k + 1].push_back(m_cornersRight[k].y);
//		}*/
//
//		//Time = (double)cvGetTickCount() - Time;
//		//printf("checkHash time = %gms\n", Time / (cvGetTickFrequency() * 1000));//毫秒
//		//printf("checkHash time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//毫秒
//
//		//Time = (double)cvGetTickCount();
//		for (i = 0; i < nmodels; i++)
//		{
//			CvMat model_i;
//			cvGetRows(models, &model_i, i*modelSize.height, (i + 1)*modelSize.height);
//			goodCount = findInliers(m1, m2, &model_i, err, tmask, reprojThreshold);
//
//			if (goodCount > MAX(maxGoodCount, modelPoints - 1))
//			{
//				std::swap(tmask, mask);
//				cvCopy(&model_i, model);
//				maxGoodCount = goodCount;
//
//				niters = cvRANSACUpdateNumIters(confidence,
//					(double)(count - goodCount) / count, modelPoints, niters) + hashtable[goodCount / count * 10];
//				//niters = cvRANSACUpdateNumIters(confidence,
//				//(double)(count - goodCount) / count, modelPoints, niters) ;
//				//cout << "niters" << niters << endl;
//				//cout << "hashtable[goodCount/count*10]" << hashtable[goodCount / count * 10] << endl;
//			}
//		}
//		/*if (mv_cornershash.size() >300)
//		break;*/
//
//		//Time = (double)cvGetTickCount() - Time;
//		//printf("nmodels time = %gms\n", Time / (cvGetTickFrequency() * 1000));//毫秒
//		//printf("nmodels time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//毫秒
//	}
//
//
//
//
//
//	/*for (int k = 0; k < 8; k++)
//	{
//	sort(vecmodels[k].begin(), vecmodels[k].end());
//	}
//
//	vector<Point2f> selectcorners(4);
//	int datasaize = vecmodels[0].size();
//	for (int i = 0; i < 4; i++)
//	{
//	selectcorners[i].x = vecmodels[2 * i][datasaize / 2];
//	selectcorners[i].y = vecmodels[2 * i + 1][vecmodels[2 * i + 1].size() / 2];
//	}
//	Mat affine = findHomography(m_corners, selectcorners);
//	cout << affine << endl;*/
//	if (maxGoodCount > 0)
//	{
//		if (mask != mask0)
//			cvCopy(mask, mask0);
//		result = true;
//	}
//	//result = true;
//	//Time = (double)cvGetTickCount() - Time;
//	//printf("checkHash time = %gms\n", Time / (cvGetTickFrequency() * 1000));//毫秒
//	//printf("checkHash time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//毫秒
//	return result;
//}





bool CvModelEstimator2::getSubset( const CvMat* m1, const CvMat* m2,
                                   CvMat* ms1, CvMat* ms2, int maxAttempts )
{
    cv::AutoBuffer<int> _idx(modelPoints);
    int* idx = _idx;
    int i = 0, j, k, idx_i, iters = 0;
    int type = CV_MAT_TYPE(m1->type), elemSize = CV_ELEM_SIZE(type);
    const int *m1ptr = m1->data.i, *m2ptr = m2->data.i;
    int *ms1ptr = ms1->data.i, *ms2ptr = ms2->data.i;
    int count = m1->cols*m1->rows;

    assert( CV_IS_MAT_CONT(m1->type & m2->type) && (elemSize % sizeof(int) == 0) );
    elemSize /= sizeof(int);

    for(; iters < maxAttempts; iters++)
    {
        for( i = 0; i < modelPoints && iters < maxAttempts; )
        {
            idx[i] = idx_i = cvRandInt(&rng) % count;
            for( j = 0; j < i; j++ )
                if( idx_i == idx[j] )
                    break;
            if( j < i )
                continue;
            for( k = 0; k < elemSize; k++ )
            {
                ms1ptr[i*elemSize + k] = m1ptr[idx_i*elemSize + k];
                ms2ptr[i*elemSize + k] = m2ptr[idx_i*elemSize + k];
            }
            if( checkPartialSubsets && (!checkSubset( ms1, i+1 ) || !checkSubset( ms2, i+1 )))
            {
                iters++;
                continue;
            }
            i++;
        }
        if( !checkPartialSubsets && i == modelPoints &&
            (!checkSubset( ms1, i ) || !checkSubset( ms2, i )))
            continue;
        break;
    }

    return i == modelPoints && iters < maxAttempts;
}


bool CvModelEstimator2::checkSubset( const CvMat* m, int count )
{
    if( count <= 2 )
        return true;

    int j, k, i, i0, i1;
    CvPoint2D64f* ptr = (CvPoint2D64f*)m->data.ptr;

    assert( CV_MAT_TYPE(m->type) == CV_64FC2 );

    if( checkPartialSubsets )
        i0 = i1 = count - 1;
    else
        i0 = 0, i1 = count - 1;

    for( i = i0; i <= i1; i++ )
    {
        // check that the i-th selected point does not belong
        // to a line connecting some previously selected points
        for( j = 0; j < i; j++ )
        {
            double dx1 = ptr[j].x - ptr[i].x;
            double dy1 = ptr[j].y - ptr[i].y;
            for( k = 0; k < j; k++ )
            {
                double dx2 = ptr[k].x - ptr[i].x;
                double dy2 = ptr[k].y - ptr[i].y;
                if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                    break;
            }
            if( k < j )
                break;
        }
        if( j < i )
            break;
    }

    return i > i1;
}

}//jw