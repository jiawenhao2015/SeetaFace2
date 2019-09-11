/************************************************************
Author: GuLong
Date: 2019.06.25 15:00
Description: Face Lift 
referred paper: Image Deformation Using Moving Least Squares
				Interactive Image Warping
************************************************************/
#ifndef FACE_LIFT_H_
#define FACE_LIFT_H_

#include <opencv2/core/core.hpp>
#include <vector>

namespace beauty
{
	void face_lift(cv::Mat& img, const std::vector<cv::Point2f>& landmarks,
		int change, const cv::Rect& face);

	/*void general_face_lift(cv::Mat& img, const std::vector<std::vector<cv::Point2f>>& multiLandmarks,
		int change, const std::vector<cv::Rect>& faces);*/

	cv::Mat local_translation_warp(cv::Mat& img, const int start_x, const int start_y,
		const int end_x, const int end_y, const float radius);

	std::vector<int> bilinear_insert(const cv::Mat& img, const float ux, const float uy);
	std::vector<int> bilinear_insert(const cv::Mat& img, const uchar *pSrcData, const float ux, const float uy,const int rows,const int cols);

	//img: face area
	cv::Mat face_lift(cv::Mat& img,const std::vector<cv::Point2f>& landmarks);

	void get_offset_landmarks(const std::vector<cv::Point2f>& landmarks, const cv::Point2f& start_point,
		std::vector<cv::Point2f>& offset_landmaks);

	void general_face_lift(cv::Mat& img, const std::vector<cv::Rect>& faces,
		const std::vector<std::vector<cv::Point2f>>& multiLandmark);
}

#endif