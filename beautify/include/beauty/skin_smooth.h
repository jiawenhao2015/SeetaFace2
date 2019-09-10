/************************************************************
Author: GuLong
Date: 2019.06.18 16:30
Description: Skin Smooth Class
************************************************************/

#ifndef BEAUTY_H_
#define BEAUTY_H_

#include <opencv2/core.hpp>
#include <vector>

namespace beauty
{
	class SkinSmooth
	{
	public:
		SkinSmooth();
		SkinSmooth(const int smoothValue, const int detailValue, const int opacity, const int contrastValue,
			const int brightValue,const float scale_ratio,const bool draw_rect);

		SkinSmooth(SkinSmooth&) = delete;
		SkinSmooth &operator=(const SkinSmooth&) = delete;

		void contrastAugement(cv::Mat& img);
		void smooth(cv::Mat& img,const std::vector<cv::Rect>& rects);

	private:
		int smoothValue_;  //set soomthValue_ = 3 * detailValue_, result will be better
		int detailValue_;
		int opacity_;
		int contrastValue_;
		int brightValue_;
		float scale_ratio_;
		bool draw_rect_;
	};
	
	
} //namespace beauty

#endif
