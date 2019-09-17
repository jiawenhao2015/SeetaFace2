#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <time.h>

#include "beauty/skin_smooth.h"
#include <vector>
#include <opencv2/core.hpp>
#include "beauty/commontools.h"

using namespace std;
using namespace cv;

namespace beauty
{
	SkinSmooth::SkinSmooth()
	{
		smoothValue_ = 2;
		detailValue_ = 1;
		opacity_ = 50;
		contrastValue_ = 120;
		brightValue_ = 20;
		scale_ratio_ = 1.0;
		draw_rect_ = false;
	}

	SkinSmooth::SkinSmooth(const int smoothValue,const int detailValue, const int opacity, const int contrastValue, 
		const int brightValue,const float scale_ratio = 1.0f,const bool draw_rect = false)
	{
		smoothValue_ = smoothValue;
		detailValue_ = detailValue;
		opacity_ = opacity;
		contrastValue_ = contrastValue;
		brightValue_ = brightValue;
		scale_ratio_ = scale_ratio;
		draw_rect_ = draw_rect;
	}

	void SkinSmooth::contrastAugement(cv::Mat& img)
	{
		//contrastValue_ = int(contrastValue_ * 0.01);
		//cv::Mat scaleMat(img.rows, img.cols, CV_8UC3,cv::Scalar(contrastValue_,contrastValue_,contrastValue_));
		//cv::Mat addMat(img.rows, img.cols, CV_8UC3, cv::Scalar(brightValue_, brightValue_, brightValue_));
		//img = img.mul(scaleMat);
		//img = img * contrastValue_;
		//cv::add(img, addMat, img);

		//for loop realization
		// for (int y = 0; y < img.rows; y++) {
		// 	for (int x = 0; x < img.cols; x++) {
		// 		for (int c = 0; c < 3; c++) {
		// 			img.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((contrastValue_*0.01)*(img.at<Vec3b>(y, x)[c]) + brightValue_);
		// 		}
		// 	}
		// }


		uchar *pTempData = img.data;
		for (int i = 0; i < img.rows * img.cols; i++, pTempData += 3)
		{
			pTempData[0] = min(pTempData[0] * 1.2 + 24.0, 255.0);
			pTempData[1] = min(pTempData[1] * 1.2 + 24.0, 255.0);
			pTempData[2] = min(pTempData[2] * 1.2 + 24.0, 255.0);
		}
	}
	//双边滤波 https://cloud.tencent.com/developer/article/1094243
	void bilateralFilter2(unsigned char* pSrc, unsigned char* pDest, int width, int height, int radius)
	{
		float delta = 0.1f;
		float eDelta = 1.0f / (2 * delta * delta);

		int colorDistTable[256 * 256] = { 0 }; 
		for (int x = 0; x < 256; x++)
		{
			int  * colorDistTablePtr = colorDistTable + (x * 256);
			for (int y = 0; y < 256; y++)
			{
				int  mod = ((x - y) * (x - y))*(1.0f / 256.0f);
				colorDistTablePtr[y] = 256 * exp(-mod * eDelta);
			}
		} 
		for (int Y = 0; Y < height; Y++)
		{
			int Py = Y * width;
			unsigned char* LinePD = pDest + Py; 
			unsigned char* LinePS = pSrc + Py;
			for (int X = 0; X < width; X++)
			{
				int sumPix = 0;
				int sum = 0;
				int factor = 0;

				for (int i = -radius; i <= radius; i++)
				{
					unsigned char* pLine = pSrc + ((Y + i + height) % height)* width;
					int cPix = 0;
					int  * colorDistPtr = colorDistTable + LinePS[X] * 256;
					for (int j = -radius; j <= radius; j++)
					{
						cPix = pLine[ (X + j+width)%width];
						factor = colorDistPtr[cPix];
						sum += factor;
						sumPix += (factor *cPix);
					}
				}
				LinePD[X] = (sumPix / sum);
			}
		} 
	}
	void SkinSmooth::smooth(cv::Mat& img,const vector<cv::Rect>& rects)
	{
		//set parameters
		Mat dst;
		int dx = smoothValue_ * 5;  //parameter of bilateralFilter
		double fc = smoothValue_ * 12.5;  //parameter of bilateralFilter
		int kernel_size = 2 * detailValue_ - 1;
		Mat tmp1, tmp2, tmp3, tmp4;

		//Smooth whole image if rects.size() == 0, which indicates
		//detect 0 face in image
		if (rects.size() == 0)
		{
			//bilateral filter
			bilateralFilter(img, tmp1, dx, fc, fc);
			//high pass
			tmp2 = (tmp1 - img + 128);
			//gaussian blur
			GaussianBlur(tmp2, tmp3, Size(kernel_size, kernel_size), 0, 0);
			//images fusion by linearity
			tmp4 = img + 2 * tmp3 - 256;
			//opacity fusion
			dst = (img * (100 - opacity_) + tmp4 * opacity_) / 100;
			//copy roi to img, roi is whole image here
			dst.copyTo(img);
		}
		else
		{
			for (vector<Rect>::const_iterator it = rects.begin(); it != rects.end(); it++)
			{
				Scalar color = CV_RGB(0, 0, 255);
				Point top_left, right_down;
				int rect_h = it->height;
				int rect_w = it->width;
				int expand_h = 0; 
				int expand_w = 0;
				if (scale_ratio_ > 1.0f)
				{
					expand_h = int(rect_h / 2.0 * (scale_ratio_ - 1));
					expand_w = int(rect_w / 2.0 * (scale_ratio_ - 1));
				}
				
				top_left.x = it->x - expand_w;
				top_left.y = it->y - expand_h;
				right_down.x = it->x + rect_w + expand_w;
				right_down.y = it->y + rect_h + expand_h;
				if (top_left.x < 0) top_left.x = 0;
				if (top_left.y < 0) top_left.y = 0;
				if (right_down.x > img.cols) right_down.x = img.cols - 1;
				if (right_down.y > img.rows) right_down.y = img.rows - 1;

				top_left.x = min(top_left.x, img.cols -1);
				top_left.y = min(top_left.y, img.rows -1);
				right_down.x = min(right_down.x, img.cols -1);
				right_down.y = min(right_down.y, img.rows -1);

				//draw rectangle
				if (draw_rect_)
					rectangle(img, top_left, right_down, color);
				
				Mat roi = img(Range(top_left.y, right_down.y), Range(top_left.x, right_down.x));
				//same process
				
				if(roi.empty())
				{
					continue;
					assert(roi.data != tmp1.data);

				}
				
				TimeStatic(3,NULL);
				//bilateralFilter(roi, tmp1, dx, fc, fc);//dx=10 fc=25
				//bilateralFilter(roi.data, tmp1.data, roi.cols, roi.rows, 10);
				bilateralFilter(roi, tmp1, 9, 15, 15);
				TimeStatic(3,"bilateral filter");
				tmp2 = (tmp1 - roi + 128);

				
				GaussianBlur(tmp2, tmp3, Size(kernel_size, kernel_size), 0, 0);
				
				tmp4 = roi + 2 * tmp3 - 255;
				dst = (roi * (100 - opacity_) + tmp4 * opacity_) / 100;
				
				dst.copyTo(roi);
				
			}
		}
		// double t0 = (double)getTickCount();
		// contrastAugement(img);
		// double t1 = (double)getTickCount();
		// printf("contrastAugement:%gms\n",(t1 - t0) * 1000 / getTickFrequency());
	}
} //end namespace beauty

/*
怀旧风格算法
R = 0.393 * r + 0.769 * g + 0.189 * b;
G = 0.349 * r + 0.686 * g + 0.168 * b;
B = 0.272 * r + 0.534 * g + 0.131 * b;
*/