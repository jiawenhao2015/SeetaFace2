#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <time.h>

#include "beauty/skin_smooth.h"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "beauty/commontools.h"
#include "beauty/rbf.h"

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


		// uchar *pTempData = img.data;
		// for (int i = 0; i < img.rows * img.cols; i++, pTempData += 3)
		// {
		// 	pTempData[0] = min(pTempData[0] * 1.2 + 24.0, 255.0);
		// 	pTempData[1] = min(pTempData[1] * 1.2 + 24.0, 255.0);
		// 	pTempData[2] = min(pTempData[2] * 1.2 + 24.0, 255.0);
		// }
		uchar *pTempData = img.data;
		for (int i = 0; i < img.rows * img.cols; i++, pTempData += 3)
		{
			pTempData[0] = min(pTempData[0] * 1.05 + 24.0, 255.0);
			pTempData[1] = min(pTempData[1] * 1.05 + 24.0, 255.0);
			pTempData[2] = min(pTempData[2] * 1.05 + 24.0, 255.0);
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
	//导向滤波器
	Mat guidedFilterCore(Mat &srcMat, Mat &guidedMat, int radius, double eps)
	{
		//------------【0】转换源图像信息，将输入扩展为64位浮点型，以便以后做乘法------------
		srcMat.convertTo(srcMat, CV_64FC1);
		guidedMat.convertTo(guidedMat, CV_64FC1);
		//--------------【1】各种均值计算----------------------------------
		Mat mean_p, mean_I, mean_Ip, mean_II;
		boxFilter(srcMat, mean_p, CV_64FC1, Size(radius, radius));//生成待滤波图像均值mean_p	
		boxFilter(guidedMat, mean_I, CV_64FC1, Size(radius, radius));//生成引导图像均值mean_I	
		boxFilter(srcMat.mul(guidedMat), mean_Ip, CV_64FC1, Size(radius, radius));//生成互相关均值mean_Ip
		boxFilter(guidedMat.mul(guidedMat), mean_II, CV_64FC1, Size(radius, radius));//生成引导图像自相关均值mean_II
		//--------------【2】计算相关系数，计算Ip的协方差cov和I的方差var------------------
		Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
		Mat var_I = mean_II - mean_I.mul(mean_I);
		//---------------【3】计算参数系数a、b-------------------
		Mat a = cov_Ip / (var_I + eps);
		Mat b = mean_p - a.mul(mean_I);
		//--------------【4】计算系数a、b的均值-----------------
		Mat mean_a, mean_b;
		boxFilter(a, mean_a, CV_64FC1, Size(radius, radius));
		boxFilter(b, mean_b, CV_64FC1, Size(radius, radius));
		//---------------【5】生成输出矩阵------------------
		Mat dstImage = mean_a.mul(srcMat) + mean_b;
		return dstImage;
	}
	void guidedFilter(const Mat &srcMat, Mat &dstMat, int radius, double eps)
	{
		vector<Mat> vSrcImage, vResultImage;
		split(srcMat, vSrcImage);
		for (int i = 0; i < 3; i++)
		{
			Mat tempImage;
			vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0 / 255.0);//将分通道转换成浮点型数据
			Mat cloneImage = tempImage.clone();	//将tempImage复制一份到cloneImage
			Mat resultImage = guidedFilterCore(tempImage, cloneImage, radius, eps);//对分通道分别进行导向滤波，半径为1、3、5...等奇数 eps 0.01
			vResultImage.push_back(resultImage);//将分通道导向滤波后的结果存放到vResultImage中
		}
		//----------【3】将分通道导向滤波后结果合并-----------------------
		merge(vResultImage, dstMat);
		dstMat.convertTo(dstMat, CV_8UC3, 255);//将归一化的数字，回到0-255区间
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
				
				
				bilateralFilter(roi, tmp1, dx, fc, fc);//dx=10 fc=25
				//bilateralFilter(roi.data, tmp1.data, roi.cols, roi.rows, 10);
				//bilateralFilter(roi, tmp1, 9, 15, 15);
				// imshow("roi", roi);
				// imshow("tmp1", tmp1);
TimeStatic(3,NULL);
				// Mat guidetmp;
				// guidedFilter(roi,guidetmp,1,0.01);
 TimeStatic(3,"bilateral filter");
// 				unsigned char * img_out = 0;

// 				recursive_bf(roi.data, img_out, 0.03, 0.1, roi.cols, roi.rows, roi.channels());

// 				Mat imgbf(Size(roi.cols, roi.rows), CV_8UC3);
// 				for (int i = 0; i < roi.cols * roi.rows * 3; i++)
// 				{
// 					imgbf.at<cv::Vec3b>(i / (roi.cols * 3), (i % (roi.cols * 3)) / 3)[i % 3] = img_out[i];
// 				}

 				//imshow("guidetmp", guidetmp);
				



				tmp2 = (tmp1 - roi + 128);

				
				GaussianBlur(tmp2, tmp3, Size(kernel_size, kernel_size), 0, 0);
				
				tmp4 = roi + 2 * tmp3 - 255;
				dst = (roi * (100 - opacity_) + tmp4 * opacity_) / 100;
				
				dst.copyTo(roi);
				
			}
		}
		double t0 = (double)getTickCount();
		contrastAugement(img);
		double t1 = (double)getTickCount();
		printf("contrastAugement:%gms\n",(t1 - t0) * 1000 / getTickFrequency());
	}
} //end namespace beauty

/*
怀旧风格算法
R = 0.393 * r + 0.769 * g + 0.189 * b;
G = 0.349 * r + 0.686 * g + 0.168 * b;
B = 0.272 * r + 0.534 * g + 0.131 * b;
*/