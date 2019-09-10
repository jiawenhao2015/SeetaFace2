/************************************************************
* Author: Long Gu
* Date: 2019.07.23
************************************************************/
#ifndef __TRACK_H__
#define __TRACK_H__

#include <opencv2/core/core.hpp>

class FaceTrack
{
public:
	FaceTrack();
	void initTrack(const std::string& model_path, const int& minFace);
	void detectTrackFace(cv::Rect& result, cv::Mat& img);

private:
	void* face_impl_;
};

#endif