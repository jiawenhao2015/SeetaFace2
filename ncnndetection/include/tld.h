/************************************************************
* Author: Long Gu
* Date: 2019.07.23
************************************************************/
#ifndef __TLD_H__
#define __TLD_H__

#include <opencv2/core/core.hpp>

class TLD
{
public:
	TLD();
	void defineLastBox(const cv::Rect &box);
	void processFrame(const cv::Mat& img1, const cv::Mat& img2, cv::Rect& bbox_next, bool& is_last_box_found);
	void track(const cv::Mat& img1, const cv::Mat& img2);
	bool trackf2f(const cv::Mat& img1, const cv::Mat& img2);

private:
	void bboxPoints(const cv::Rect& bbox);
	void bboxPredict(const cv::Rect& bbox1, cv::Rect& bbox2);
	void normCrossCorrelation(const cv::Mat& img1, const cv::Mat& img2);

	bool filterPts();

	//last frame data
	cv::Rect lastbox_;
	//track data
	bool tracked_;
	cv::Rect tbb_;
	std::vector<cv::Point2f> points1_;
	std::vector<cv::Point2f> points2_;
	std::vector<cv::Point2f> pointsFB_;
	std::vector<uchar> status_;
	std::vector<uchar> FB_status_;
	std::vector<float> similarity_;
	std::vector<float> FB_error_;
	float fbmed_;

	float lambda_ = 0.5;
	int level_ = 5;
	cv::Size window_size_ = cv::Size(4, 4);
	cv::TermCriteria term_criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT +
		cv::TermCriteria::EPS, 20, 0.03);

};

#endif