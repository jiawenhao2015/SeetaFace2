/************************************************************
Author: GuLong
Date: 2019.06.27 15:00
Description: Eye Enlarge Class
************************************************************/
#ifndef EYE_ENLARGE_H_
#define EYE_ENLARGE_H_

#include <vector>
#include <opencv2/core/core.hpp>

namespace beauty
{
	class EyeEnlarge
	{
	public:
		EyeEnlarge() :strength_(20), radius_(50.0f),use_bilinear_(true) {}
		EyeEnlarge(const int strength,const float radius,const bool use_bilinear):
			strength_(strength),radius_(radius),use_bilinear_(use_bilinear) {}

		void generalEyeEnlarge(cv::Mat& img, const std::vector<cv::Rect>& faces,
			const std::vector<std::vector<cv::Point2f>>& multiLandmarks);

	private:

		void enlarge(const cv::Mat& face_img,cv::Mat& dst, const cv::Point2f& center_point);

		void getLeftCenterPoint(const std::vector<cv::Point2f>& offset_landmarks, cv::Point2f& left_center_point);

		void getRightCenterPoint(const std::vector<cv::Point2f>& offset_landmarks, cv::Point2f& right_center_point);

		void singleLandmarkEyeEnlarge(cv::Mat& face_img, const std::vector<cv::Point2f>& offset_landmarks);

		void getOffsetLandmarks(const std::vector<cv::Point2f>& landmarks,const cv::Rect& face,
			std::vector<cv::Point2f>& offset_landmarks);

	private:
		int strength_;
		float radius_;
		bool use_bilinear_;
	};
}

#endif