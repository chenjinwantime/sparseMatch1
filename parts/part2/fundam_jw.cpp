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

//#include "precomp.hpp"
#include "_modelest_jw.h"

using namespace cv;
namespace jw{
template<typename T> int icvCompressPoints( T* ptr, const uchar* mask, int mstep, int count )
{
    int i, j;
    for( i = j = 0; i < count; i++ )
        if( mask[i*mstep] )
        {
            if( i > j )
                ptr[j] = ptr[i];
            j++;
        }
    return j;
}

class CvHomographyEstimator : public CvModelEstimator2
{
public:
    CvHomographyEstimator( int modelPoints );

    virtual int runKernel( const CvMat* m1, const CvMat* m2, CvMat* model );
    virtual bool refine( const CvMat* m1, const CvMat* m2,
                         CvMat* model, int maxIters );
protected:
    virtual void computeReprojError( const CvMat* m1, const CvMat* m2,
                                     const CvMat* model, CvMat* error );
};


CvHomographyEstimator::CvHomographyEstimator(int _modelPoints)
    : CvModelEstimator2(_modelPoints, cvSize(3,3), 1)
{
    assert( _modelPoints == 4 || _modelPoints == 5 );
    checkPartialSubsets = false;
}

int CvHomographyEstimator::runKernel( const CvMat* m1, const CvMat* m2, CvMat* H )
{
    int i, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;

    double LtL[9][9], W[9][1], V[9][9];
    CvMat _LtL = cvMat( 9, 9, CV_64F, LtL );
    CvMat matW = cvMat( 9, 1, CV_64F, W );
    CvMat matV = cvMat( 9, 9, CV_64F, V );
    CvMat _H0 = cvMat( 3, 3, CV_64F, V[8] );
    CvMat _Htemp = cvMat( 3, 3, CV_64F, V[7] );
    CvPoint2D64f cM={0,0}, cm={0,0}, sM={0,0}, sm={0,0};

    for( i = 0; i < count; i++ )
    {
        cm.x += m[i].x; cm.y += m[i].y;
        cM.x += M[i].x; cM.y += M[i].y;
    }

    cm.x /= count; cm.y /= count;
    cM.x /= count; cM.y /= count;

    for( i = 0; i < count; i++ )
    {
        sm.x += fabs(m[i].x - cm.x);
        sm.y += fabs(m[i].y - cm.y);
        sM.x += fabs(M[i].x - cM.x);
        sM.y += fabs(M[i].y - cM.y);
    }

    if( fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
        fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON )
        return 0;
    sm.x = count/sm.x; sm.y = count/sm.y;
    sM.x = count/sM.x; sM.y = count/sM.y;

    double invHnorm[9] = { 1./sm.x, 0, cm.x, 0, 1./sm.y, cm.y, 0, 0, 1 };
    double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };
    CvMat _invHnorm = cvMat( 3, 3, CV_64FC1, invHnorm );
    CvMat _Hnorm2 = cvMat( 3, 3, CV_64FC1, Hnorm2 );

    cvZero( &_LtL );
    for( i = 0; i < count; i++ )
    {
        double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
        double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
        double Lx[] = { X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x };
        double Ly[] = { 0, 0, 0, X, Y, 1, -y*X, -y*Y, -y };
        int j, k;
        for( j = 0; j < 9; j++ )
            for( k = j; k < 9; k++ )
                LtL[j][k] += Lx[j]*Lx[k] + Ly[j]*Ly[k];
    }
    cvCompleteSymm( &_LtL );

    //cvSVD( &_LtL, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T );
    cvEigenVV( &_LtL, &matV, &matW );
    cvMatMul( &_invHnorm, &_H0, &_Htemp );
    cvMatMul( &_Htemp, &_Hnorm2, &_H0 );
    cvConvertScale( &_H0, H, 1./_H0.data.db[8] );

    return 1;
}


void CvHomographyEstimator::computeReprojError( const CvMat* m1, const CvMat* m2,
                                                const CvMat* model, CvMat* _err )
{
    int i, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
    const double* H = model->data.db;
    float* err = _err->data.fl;

    for( i = 0; i < count; i++ )
    {
        double ww = 1./(H[6]*M[i].x + H[7]*M[i].y + 1.);
        double dx = (H[0]*M[i].x + H[1]*M[i].y + H[2])*ww - m[i].x;
        double dy = (H[3]*M[i].x + H[4]*M[i].y + H[5])*ww - m[i].y;
        err[i] = (float)(dx*dx + dy*dy);
    }
}

bool CvHomographyEstimator::refine( const CvMat* m1, const CvMat* m2, CvMat* model, int maxIters )
{
    CvLevMarq solver(8, 0, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, maxIters, DBL_EPSILON));
    int i, j, k, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
    CvMat modelPart = cvMat( solver.param->rows, solver.param->cols, model->type, model->data.ptr );
    cvCopy( &modelPart, solver.param );

    for(;;)
    {
        const CvMat* _param = 0;
        CvMat *_JtJ = 0, *_JtErr = 0;
        double* _errNorm = 0;

        if( !solver.updateAlt( _param, _JtJ, _JtErr, _errNorm ))
            break;

        for( i = 0; i < count; i++ )
        {
            const double* h = _param->data.db;
            double Mx = M[i].x, My = M[i].y;
            double ww = h[6]*Mx + h[7]*My + 1.;
            ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
            double _xi = (h[0]*Mx + h[1]*My + h[2])*ww;
            double _yi = (h[3]*Mx + h[4]*My + h[5])*ww;
            double err[] = { _xi - m[i].x, _yi - m[i].y };
            if( _JtJ || _JtErr )
            {
                double J[][8] =
                {
                    { Mx*ww, My*ww, ww, 0, 0, 0, -Mx*ww*_xi, -My*ww*_xi },
                    { 0, 0, 0, Mx*ww, My*ww, ww, -Mx*ww*_yi, -My*ww*_yi }
                };

                for( j = 0; j < 8; j++ )
                {
                    for( k = j; k < 8; k++ )
                        _JtJ->data.db[j*8+k] += J[0][j]*J[0][k] + J[1][j]*J[1][k];
                    _JtErr->data.db[j] += J[0][j]*err[0] + J[1][j]*err[1];
                }
            }
            if( _errNorm )
                *_errNorm += err[0]*err[0] + err[1]*err[1];
        }
    }

    cvCopy( solver.param, &modelPart );
    return true;
}


void cvConvertPointsHomogeneous(const CvMat* src, CvMat* dst)
{
	Ptr<CvMat> temp, denom;

	int i, s_count, s_dims, d_count, d_dims;
	CvMat _src, _dst, _ones;
	CvMat* ones = 0;

	if (!CV_IS_MAT(src))
		CV_Error(!src ? CV_StsNullPtr : CV_StsBadArg,
			"The input parameter is not a valid matrix");

	if (!CV_IS_MAT(dst))
		CV_Error(!dst ? CV_StsNullPtr : CV_StsBadArg,
			"The output parameter is not a valid matrix");

	if (src == dst || src->data.ptr == dst->data.ptr)
	{
		if (src != dst && (!CV_ARE_TYPES_EQ(src, dst) || !CV_ARE_SIZES_EQ(src, dst)))
			CV_Error(CV_StsBadArg, "Invalid inplace operation");
		return;
	}

	if (src->rows > src->cols)
	{
		if (!((src->cols > 1) ^ (CV_MAT_CN(src->type) > 1)))
			CV_Error(CV_StsBadSize, "Either the number of channels or columns or rows must be =1");

		s_dims = CV_MAT_CN(src->type)*src->cols;
		s_count = src->rows;
	}
	else
	{
		if (!((src->rows > 1) ^ (CV_MAT_CN(src->type) > 1)))
			CV_Error(CV_StsBadSize, "Either the number of channels or columns or rows must be =1");

		s_dims = CV_MAT_CN(src->type)*src->rows;
		s_count = src->cols;
	}

	if (src->rows == 1 || src->cols == 1)
		src = cvReshape(src, &_src, 1, s_count);

	if (dst->rows > dst->cols)
	{
		if (!((dst->cols > 1) ^ (CV_MAT_CN(dst->type) > 1)))
			CV_Error(CV_StsBadSize,
				"Either the number of channels or columns or rows in the input matrix must be =1");

		d_dims = CV_MAT_CN(dst->type)*dst->cols;
		d_count = dst->rows;
	}
	else
	{
		if (!((dst->rows > 1) ^ (CV_MAT_CN(dst->type) > 1)))
			CV_Error(CV_StsBadSize,
				"Either the number of channels or columns or rows in the output matrix must be =1");

		d_dims = CV_MAT_CN(dst->type)*dst->rows;
		d_count = dst->cols;
	}

	if (dst->rows == 1 || dst->cols == 1)
		dst = cvReshape(dst, &_dst, 1, d_count);

	if (s_count != d_count)
		CV_Error(CV_StsUnmatchedSizes, "Both matrices must have the same number of points");

	if (CV_MAT_DEPTH(src->type) < CV_32F || CV_MAT_DEPTH(dst->type) < CV_32F)
		CV_Error(CV_StsUnsupportedFormat,
			"Both matrices must be floating-point (single or double precision)");

	if (s_dims < 2 || s_dims > 4 || d_dims < 2 || d_dims > 4)
		CV_Error(CV_StsOutOfRange,
			"Both input and output point dimensionality must be 2, 3 or 4");

	if (s_dims < d_dims - 1 || s_dims > d_dims + 1)
		CV_Error(CV_StsUnmatchedSizes,
			"The dimensionalities of input and output point sets differ too much");

	if (s_dims == d_dims - 1)
	{
		if (d_count == dst->rows)
		{
			ones = cvGetSubRect(dst, &_ones, cvRect(s_dims, 0, 1, d_count));
			dst = cvGetSubRect(dst, &_dst, cvRect(0, 0, s_dims, d_count));
		}
		else
		{
			ones = cvGetSubRect(dst, &_ones, cvRect(0, s_dims, d_count, 1));
			dst = cvGetSubRect(dst, &_dst, cvRect(0, 0, d_count, s_dims));
		}
	}

	if (s_dims <= d_dims)
	{
		if (src->rows == dst->rows && src->cols == dst->cols)
		{
			if (CV_ARE_TYPES_EQ(src, dst))
				cvCopy(src, dst);
			else
				cvConvert(src, dst);
		}
		else
		{
			if (!CV_ARE_TYPES_EQ(src, dst))
			{
				temp = cvCreateMat(src->rows, src->cols, dst->type);
				cvConvert(src, temp);
				src = temp;
			}
			cvTranspose(src, dst);
		}

		if (ones)
			cvSet(ones, cvRealScalar(1.));
	}
	else
	{
		int s_plane_stride, s_stride, d_plane_stride, d_stride, elem_size;

		if (!CV_ARE_TYPES_EQ(src, dst))
		{
			temp = cvCreateMat(src->rows, src->cols, dst->type);
			cvConvert(src, temp);
			src = temp;
		}

		elem_size = CV_ELEM_SIZE(src->type);

		if (s_count == src->cols)
			s_plane_stride = src->step / elem_size, s_stride = 1;
		else
			s_stride = src->step / elem_size, s_plane_stride = 1;

		if (d_count == dst->cols)
			d_plane_stride = dst->step / elem_size, d_stride = 1;
		else
			d_stride = dst->step / elem_size, d_plane_stride = 1;

		denom = cvCreateMat(1, d_count, dst->type);

		if (CV_MAT_DEPTH(dst->type) == CV_32F)
		{
			const float* xs = src->data.fl;
			const float* ys = xs + s_plane_stride;
			const float* zs = 0;
			const float* ws = xs + (s_dims - 1)*s_plane_stride;

			float* iw = denom->data.fl;

			float* xd = dst->data.fl;
			float* yd = xd + d_plane_stride;
			float* zd = 0;

			if (d_dims == 3)
			{
				zs = ys + s_plane_stride;
				zd = yd + d_plane_stride;
			}

			for (i = 0; i < d_count; i++, ws += s_stride)
			{
				float t = *ws;
				iw[i] = fabs((double)t) > FLT_EPSILON ? t : 1.f;
			}

			cvDiv(0, denom, denom);

			if (d_dims == 3)
				for (i = 0; i < d_count; i++)
				{
					float w = iw[i];
					float x = *xs * w, y = *ys * w, z = *zs * w;
					xs += s_stride; ys += s_stride; zs += s_stride;
					*xd = x; *yd = y; *zd = z;
					xd += d_stride; yd += d_stride; zd += d_stride;
				}
			else
				for (i = 0; i < d_count; i++)
				{
					float w = iw[i];
					float x = *xs * w, y = *ys * w;
					xs += s_stride; ys += s_stride;
					*xd = x; *yd = y;
					xd += d_stride; yd += d_stride;
				}
		}
		else
		{
			const double* xs = src->data.db;
			const double* ys = xs + s_plane_stride;
			const double* zs = 0;
			const double* ws = xs + (s_dims - 1)*s_plane_stride;

			double* iw = denom->data.db;

			double* xd = dst->data.db;
			double* yd = xd + d_plane_stride;
			double* zd = 0;

			if (d_dims == 3)
			{
				zs = ys + s_plane_stride;
				zd = yd + d_plane_stride;
			}

			for (i = 0; i < d_count; i++, ws += s_stride)
			{
				double t = *ws;
				iw[i] = fabs(t) > DBL_EPSILON ? t : 1.;
			}

			cvDiv(0, denom, denom);

			if (d_dims == 3)
				for (i = 0; i < d_count; i++)
				{
					double w = iw[i];
					double x = *xs * w, y = *ys * w, z = *zs * w;
					xs += s_stride; ys += s_stride; zs += s_stride;
					*xd = x; *yd = y; *zd = z;
					xd += d_stride; yd += d_stride; zd += d_stride;
				}
			else
				for (i = 0; i < d_count; i++)
				{
					double w = iw[i];
					double x = *xs * w, y = *ys * w;
					xs += s_stride; ys += s_stride;
					*xd = x; *yd = y;
					xd += d_stride; yd += d_stride;
				}
		}
	}
}

int cvFindHomography( const CvMat* objectPoints, const CvMat* imagePoints,
                  CvMat* __H, int method, double ransacReprojThreshold,
                  CvMat* mask )
{
    const double confidence = 0.995;
    const int maxIters = 2000;
    const double defaultRANSACReprojThreshold = 3;
    bool result = false;
    Ptr<CvMat> m, M, tempMask;

    double H[9];
    CvMat matH = cvMat( 3, 3, CV_64FC1, H );
    int count;

    CV_Assert( CV_IS_MAT(imagePoints) && CV_IS_MAT(objectPoints) );

    count = MAX(imagePoints->cols, imagePoints->rows);
    CV_Assert( count >= 4 );
    if( ransacReprojThreshold <= 0 )
        ransacReprojThreshold = defaultRANSACReprojThreshold;

    m = cvCreateMat( 1, count, CV_64FC2 );
    jw::cvConvertPointsHomogeneous( imagePoints, m );

    M = cvCreateMat( 1, count, CV_64FC2 );
    jw::cvConvertPointsHomogeneous( objectPoints, M );

    if( mask )
    {
        CV_Assert( CV_IS_MASK_ARR(mask) && CV_IS_MAT_CONT(mask->type) &&
            (mask->rows == 1 || mask->cols == 1) &&
            mask->rows*mask->cols == count );
    }
    if( mask || count > 4 )
        tempMask = cvCreateMat( 1, count, CV_8U );
    if( !tempMask.empty() )
        cvSet( tempMask, cvScalarAll(1.) );

    CvHomographyEstimator estimator(4);
    if( count == 4 )
        method = 0;
	if (method == CV_LMEDS)
	{
		result = estimator.runRANSAC(M, m, &matH, tempMask, ransacReprojThreshold, confidence, maxIters);
	}
        //result = estimator.runLMeDS( M, m, &matH, tempMask, confidence, maxIters );
    else if( method == CV_RANSAC )
	{
		//double Time = (double)cvGetTickCount();
        result = estimator.runRANSAC( M, m, &matH, tempMask, ransacReprojThreshold, confidence, maxIters);
		//Time = (double)cvGetTickCount() - Time;
		//printf("checkHash time = %gms\n", Time / (cvGetTickFrequency() * 1000));//ºÁÃë
		//printf("checkHash time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//ºÁÃë
	}
	else
        result = estimator.runKernel( M, m, &matH ) > 0;

    if( result && count > 4 )
    {
        icvCompressPoints( (CvPoint2D64f*)M->data.ptr, tempMask->data.ptr, 1, count );
        count = icvCompressPoints( (CvPoint2D64f*)m->data.ptr, tempMask->data.ptr, 1, count );
        M->cols = m->cols = count;
        if( method == CV_RANSAC )
            estimator.runKernel( M, m, &matH );
        estimator.refine( M, m, &matH, 10 );
    }

    if( result )
        cvConvert( &matH, __H );

    if( mask && tempMask )
    {
        if( CV_ARE_SIZES_EQ(mask, tempMask) )
           cvCopy( tempMask, mask );
        else
           cvTranspose( tempMask, mask );
    }

    return (int)result;
}



 /*void cvComputeCorrespondEpilines( const CvMat* points, int pointImageID,
                                          const CvMat* fmatrix, CvMat* lines )
{
    int abc_stride, abc_plane_stride, abc_elem_size;
    int plane_stride, stride, elem_size;
    int i, dims, count, depth, cn, abc_dims, abc_count, abc_depth, abc_cn;
    uchar *ap, *bp, *cp;
    const uchar *xp, *yp, *zp;
    double f[9];
    CvMat F = cvMat( 3, 3, CV_64F, f );

    if( !CV_IS_MAT(points) )
        CV_Error( !points ? CV_StsNullPtr : CV_StsBadArg, "points parameter is not a valid matrix" );

    depth = CV_MAT_DEPTH(points->type);
    cn = CV_MAT_CN(points->type);
    if( (depth != CV_32F && depth != CV_64F) || (cn != 1 && cn != 2 && cn != 3) )
        CV_Error( CV_StsUnsupportedFormat, "The format of point matrix is unsupported" );

    if( cn > 1 )
    {
        dims = cn;
        CV_Assert( points->rows == 1 || points->cols == 1 );
        count = points->rows * points->cols;
    }
    else if( points->rows > points->cols )
    {
        dims = cn*points->cols;
        count = points->rows;
    }
    else
    {
        if( (points->rows > 1 && cn > 1) || (points->rows == 1 && cn == 1) )
            CV_Error( CV_StsBadSize, "The point matrix does not have a proper layout (2xn, 3xn, nx2 or nx3)" );
        dims = points->rows;
        count = points->cols;
    }

    if( dims != 2 && dims != 3 )
        CV_Error( CV_StsOutOfRange, "The dimensionality of points must be 2 or 3" );

    if( !CV_IS_MAT(fmatrix) )
        CV_Error( !fmatrix ? CV_StsNullPtr : CV_StsBadArg, "fmatrix is not a valid matrix" );

    if( CV_MAT_TYPE(fmatrix->type) != CV_32FC1 && CV_MAT_TYPE(fmatrix->type) != CV_64FC1 )
        CV_Error( CV_StsUnsupportedFormat, "fundamental matrix must have 32fC1 or 64fC1 type" );

    if( fmatrix->cols != 3 || fmatrix->rows != 3 )
        CV_Error( CV_StsBadSize, "fundamental matrix must be 3x3" );

    if( !CV_IS_MAT(lines) )
        CV_Error( !lines ? CV_StsNullPtr : CV_StsBadArg, "lines parameter is not a valid matrix" );

    abc_depth = CV_MAT_DEPTH(lines->type);
    abc_cn = CV_MAT_CN(lines->type);
    if( (abc_depth != CV_32F && abc_depth != CV_64F) || (abc_cn != 1 && abc_cn != 3) )
        CV_Error( CV_StsUnsupportedFormat, "The format of the matrix of lines is unsupported" );

    if( abc_cn > 1 )
    {
        abc_dims = abc_cn;
        CV_Assert( lines->rows == 1 || lines->cols == 1 );
        abc_count = lines->rows * lines->cols;
    }
    else if( lines->rows > lines->cols )
    {
        abc_dims = abc_cn*lines->cols;
        abc_count = lines->rows;
    }
    else
    {
        if( (lines->rows > 1 && abc_cn > 1) || (lines->rows == 1 && abc_cn == 1) )
            CV_Error( CV_StsBadSize, "The lines matrix does not have a proper layout (3xn or nx3)" );
        abc_dims = lines->rows;
        abc_count = lines->cols;
    }

    if( abc_dims != 3 )
        CV_Error( CV_StsOutOfRange, "The lines matrix does not have a proper layout (3xn or nx3)" );

    if( abc_count != count )
        CV_Error( CV_StsUnmatchedSizes, "The numbers of points and lines are different" );

    elem_size = CV_ELEM_SIZE(depth);
    abc_elem_size = CV_ELEM_SIZE(abc_depth);

    if( cn == 1 && points->rows == dims )
    {
        plane_stride = points->step;
        stride = elem_size;
    }
    else
    {
        plane_stride = elem_size;
        stride = points->rows == 1 ? dims*elem_size : points->step;
    }

    if( abc_cn == 1 && lines->rows == 3 )
    {
        abc_plane_stride = lines->step;
        abc_stride = abc_elem_size;
    }
    else
    {
        abc_plane_stride = abc_elem_size;
        abc_stride = lines->rows == 1 ? 3*abc_elem_size : lines->step;
    }

    cvConvert( fmatrix, &F );
    if( pointImageID == 2 )
        cvTranspose( &F, &F );

    xp = points->data.ptr;
    yp = xp + plane_stride;
    zp = dims == 3 ? yp + plane_stride : 0;

    ap = lines->data.ptr;
    bp = ap + abc_plane_stride;
    cp = bp + abc_plane_stride;

    for( i = 0; i < count; i++ )
    {
        double x, y, z = 1.;
        double a, b, c, nu;

        if( depth == CV_32F )
        {
            x = *(float*)xp; y = *(float*)yp;
            if( zp )
                z = *(float*)zp, zp += stride;
        }
        else
        {
            x = *(double*)xp; y = *(double*)yp;
            if( zp )
                z = *(double*)zp, zp += stride;
        }

        xp += stride; yp += stride;

        a = f[0]*x + f[1]*y + f[2]*z;
        b = f[3]*x + f[4]*y + f[5]*z;
        c = f[6]*x + f[7]*y + f[8]*z;
        nu = a*a + b*b;
        nu = nu ? 1./sqrt(nu) : 1.;
        a *= nu; b *= nu; c *= nu;

        if( abc_depth == CV_32F )
        {
            *(float*)ap = (float)a;
            *(float*)bp = (float)b;
            *(float*)cp = (float)c;
        }
        else
        {
            *(double*)ap = a;
            *(double*)bp = b;
            *(double*)cp = c;
        }

        ap += abc_stride;
        bp += abc_stride;
        cp += abc_stride;
    }
}*/

cv::Mat findHomography( InputArray _points1, InputArray _points2,
                            int method, double ransacReprojThreshold, OutputArray _mask )
{
    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    int npoints = points1.checkVector(2);
    CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
               points1.type() == points2.type());

    Mat H(3, 3, CV_64F);
    CvMat _pt1 = points1, _pt2 = points2;
    CvMat matH = H, c_mask, *p_mask = 0;
    if( _mask.needed() )
    {
        _mask.create(npoints, 1, CV_8U, -1, true);
        p_mask = &(c_mask = _mask.getMat());
    }
    bool ok = jw::cvFindHomography( &_pt1, &_pt2, &matH, method, ransacReprojThreshold, p_mask ) > 0;
    if( !ok )
        H = Scalar(0);
    return H;
}

// int
//cvFindHomography(const CvMat* objectPoints, const CvMat* imagePoints,
//	CvMat* __H, int method, double ransacReprojThreshold,
//	CvMat* mask)
//{
//	const double confidence = 0.995;
//	const int maxIters = 2000;
//	const double defaultRANSACReprojThreshold = 3;
//	bool result = false;
//	Ptr<CvMat> m, M, tempMask;
//
//	double H[9];
//	CvMat matH = cvMat(3, 3, CV_64FC1, H);
//	int count;
//
//	CV_Assert(CV_IS_MAT(imagePoints) && CV_IS_MAT(objectPoints));
//
//	count = MAX(imagePoints->cols, imagePoints->rows);
//	CV_Assert(count >= 4);
//	if (ransacReprojThreshold <= 0)
//		ransacReprojThreshold = defaultRANSACReprojThreshold;
//
//	m = cvCreateMat(1, count, CV_64FC2);
//	jw::cvConvertPointsHomogeneous(imagePoints, m);
//
//	M = cvCreateMat(1, count, CV_64FC2);
//	jw::cvConvertPointsHomogeneous(objectPoints, M);
//
//	if (mask)
//	{
//		CV_Assert(CV_IS_MASK_ARR(mask) && CV_IS_MAT_CONT(mask->type) &&
//			(mask->rows == 1 || mask->cols == 1) &&
//			mask->rows*mask->cols == count);
//	}
//	if (mask || count > 4)
//		tempMask = cvCreateMat(1, count, CV_8U);
//	if (!tempMask.empty())
//		cvSet(tempMask, cvScalarAll(1.));
//
//	CvHomographyEstimator estimator(4);
//	if (count == 4)
//		method = 0;
//	if (method == CV_LMEDS)
//		result = estimator.runLMeDS(M, m, &matH, tempMask, confidence, maxIters);
//	else if (method == CV_RANSAC)
//	{
//		double Time = (double)cvGetTickCount();
//		result = estimator.runRANSAC(M, m, &matH, tempMask, ransacReprojThreshold, confidence, maxIters);
//		Time = (double)cvGetTickCount() - Time;
//		printf("checkHash time = %gms\n", Time / (cvGetTickFrequency() * 1000));//ºÁÃë
//		printf("checkHash time = %gs\n", Time / (cvGetTickFrequency() * 1000000));//ºÁÃë
//	}
//	else
//		result = estimator.runKernel(M, m, &matH) > 0;
//
//	if (result && count > 4)
//	{
//		icvCompressPoints((CvPoint2D64f*)M->data.ptr, tempMask->data.ptr, 1, count);
//		count = icvCompressPoints((CvPoint2D64f*)m->data.ptr, tempMask->data.ptr, 1, count);
//		M->cols = m->cols = count;
//		if (method == CV_RANSAC)
//			estimator.runKernel(M, m, &matH);
//		estimator.refine(M, m, &matH, 10);
//	}
//
//	if (result)
//		cvConvert(&matH, __H);
//
//	if (mask && tempMask)
//	{
//		if (CV_ARE_SIZES_EQ(mask, tempMask))
//			cvCopy(tempMask, mask);
//		else
//			cvTranspose(tempMask, mask);
//	}
//
//	return (int)result;
//}

//cv::Mat jw::findHomography( InputArray _points1, InputArray _points2,
//                            OutputArray _mask, int method, double ransacReprojThreshold )
//{
//    return cv::findHomography(_points1, _points2, method, ransacReprojThreshold, _mask);
//}

//cv::Mat jw::findFundamentalMat( InputArray _points1, InputArray _points2,
//                               int method, double param1, double param2,
//                               OutputArray _mask )
//{
//    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
//    int npoints = points1.checkVector(2);
//    CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
//              points1.type() == points2.type());
//
//    Mat F(method == CV_FM_7POINT ? 9 : 3, 3, CV_64F);
//    CvMat _pt1 = points1, _pt2 = points2;
//    CvMat matF = F, c_mask, *p_mask = 0;
//    if( _mask.needed() )
//    {
//        _mask.create(npoints, 1, CV_8U, -1, true);
//        p_mask = &(c_mask = _mask.getMat());
//    }
//    int n = cvFindFundamentalMat( &_pt1, &_pt2, &matF, method, param1, param2, p_mask );
//    if( n <= 0 )
//        F = Scalar(0);
//    if( n == 1 )
//        F = F.rowRange(0, 3);
//    return F;
//}
//
//cv::Mat cv::findFundamentalMat( InputArray _points1, InputArray _points2,
//                                OutputArray _mask, int method, double param1, double param2 )
//{
//    return cv::findFundamentalMat(_points1, _points2, method, param1, param2, _mask);
//}
//
//
//void cv::computeCorrespondEpilines( InputArray _points, int whichImage,
//                                    InputArray _Fmat, OutputArray _lines )
//{
//    Mat points = _points.getMat(), F = _Fmat.getMat();
//    int npoints = points.checkVector(2);
//    if( npoints < 0 )
//        npoints = points.checkVector(3);
//    CV_Assert( npoints >= 0 && (points.depth() == CV_32F || points.depth() == CV_32S));
//
//    _lines.create(npoints, 1, CV_32FC3, -1, true);
//    CvMat c_points = points, c_lines = _lines.getMat(), c_F = F;
//    cvComputeCorrespondEpilines(&c_points, whichImage, &c_F, &c_lines);
//}
//
//void cv::convertPointsFromHomogeneous( InputArray _src, OutputArray _dst )
//{
//    Mat src = _src.getMat();
//    int npoints = src.checkVector(3), cn = 3;
//    if( npoints < 0 )
//    {
//        npoints = src.checkVector(4);
//        if( npoints >= 0 )
//            cn = 4;
//    }
//    CV_Assert( npoints >= 0 && (src.depth() == CV_32F || src.depth() == CV_32S));
//
//    _dst.create(npoints, 1, CV_MAKETYPE(CV_32F, cn-1));
//    CvMat c_src = src, c_dst = _dst.getMat();
//    cvConvertPointsHomogeneous(&c_src, &c_dst);
//}

//void cv::convertPointsToHomogeneous( InputArray _src, OutputArray _dst )
//{
//    Mat src = _src.getMat();
//    int npoints = src.checkVector(2), cn = 2;
//    if( npoints < 0 )
//    {
//        npoints = src.checkVector(3);
//        if( npoints >= 0 )
//            cn = 3;
//    }
//    CV_Assert( npoints >= 0 && (src.depth() == CV_32F || src.depth() == CV_32S));
//
//    _dst.create(npoints, 1, CV_MAKETYPE(CV_32F, cn+1));
//    CvMat c_src = src, c_dst = _dst.getMat();
//    cvConvertPointsHomogeneous(&c_src, &c_dst);
//}

//void cv::convertPointsHomogeneous( InputArray _src, OutputArray _dst )
//{
//    int stype = _src.type(), dtype = _dst.type();
//    CV_Assert( _dst.fixedType() );
//
//    if( CV_MAT_CN(stype) > CV_MAT_CN(dtype) )
//        convertPointsFromHomogeneous(_src, _dst);
//    else
//        convertPointsToHomogeneous(_src, _dst);
//}


}//jw
/* End of file. */
