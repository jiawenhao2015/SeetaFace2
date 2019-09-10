#include <opencv2/imgproc/imgproc.hpp>
#include "ncnn/net.h"
#include "ncnn_mtcnn.h"
#include "tld.h"
#include "face_track.h"


using namespace std;
using namespace cv;

struct Bbox;

class FaceImpl
{
public:
	void initMtcnn(const string& model_path, const int& minFace);
	void Detect(Rect& result, Mat& img);

	vector<Bbox> finalBbox_;
	int skip_ = 0;
	Mat first_;
	Rect bbox_;
	Mat current_gray_;
	Rect pbox_;
	bool status_ = true;
	Mat last_gray_;
	TLD tld_;
	MTCNN mtcnn_;
};

void FaceImpl::initMtcnn(const string& model_path, const int& minFace)
{
	mtcnn_.init(model_path);
	mtcnn_.SetMinFace(minFace);
}

void FaceImpl::Detect(Rect& result, Mat& img)
{
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(img.data,
		ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);

	if (int(finalBbox_.size()) == 0 || status_ == false)
	{
		finalBbox_.clear();
		mtcnn_.detect(ncnn_img, finalBbox_);
		if (finalBbox_.size() > 0)
		{
			bbox_.x = finalBbox_[0].x1;
			bbox_.y = finalBbox_[0].y1;
			bbox_.height = finalBbox_[0].x2 - finalBbox_[0].x1;
			bbox_.width = finalBbox_[0].y2 - finalBbox_[0].y1;
			cvtColor(img, last_gray_, CV_BGR2GRAY);
			tld_.defineLastBox(bbox_);
			status_ = true;
		}
	}

	if (finalBbox_.size() > 0)
	{
		cvtColor(img, current_gray_, CV_BGR2GRAY);
		double t2 = (double)getTickCount();
		tld_.processFrame(last_gray_, current_gray_, pbox_, status_);
#if LOG_OUTPUT
		printf("Tracking costs: %gms\n", ((double)getTickCount() - t2) * 1000 / getTickFrequency());
#endif
		if (status_)
		{
			pbox_.width = bbox_.width;
			pbox_.height = bbox_.height;
			if (skip_ > 2)
			{
				double t1 = (double)getTickCount();
				Bbox tmpBbox;
				tmpBbox.x1 = pbox_.x;
				tmpBbox.y1 = pbox_.y;
				tmpBbox.x2 = pbox_.x + pbox_.width;
				tmpBbox.y2 = pbox_.y + pbox_.height;
				if (mtcnn_.getRnetScore(ncnn_img, tmpBbox) < 0.99)
					finalBbox_.clear();
#if LOG_OUTPUT
				printf("rnet costs: %gms\n", ((double)getTickCount() - t1) * 1000 / getTickFrequency());
#endif
				skip_ = 0;
			}
			result = pbox_;
			swap(last_gray_, current_gray_);
		}
		skip_++;
	}
}

FaceTrack::FaceTrack() : face_impl_(new FaceImpl()) 
{
	return;
}

void FaceTrack::initTrack(const string& model_path, const int& minFace)
{
	FaceImpl *p = (FaceImpl*)face_impl_;
	p->initMtcnn(model_path, minFace);
}

void FaceTrack::detectTrackFace(Rect& result, Mat& img)
{
	FaceImpl *p = (FaceImpl*)face_impl_;
	p->Detect(result, img);
}

