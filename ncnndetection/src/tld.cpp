#include <opencv2/video/tracking.hpp>
#include "tld.h"

using namespace std;
using namespace cv;

float median(vector<float> vec)
{
	int n = floor(vec.size() / 2);
	nth_element(vec.begin(), vec.begin() + n, vec.end());
	return vec[n];
}

TLD::TLD() {}

void TLD::defineLastBox(const Rect& bbox)
{
	lastbox_ = bbox;
}

void TLD::processFrame(const Mat& img1, const Mat& img2, Rect& bbox_next, bool& is_last_box_found)
{
	points1_.clear();
	points2_.clear();

	if (is_last_box_found) track(img1, img2);
	else tracked_ = false;

	if (tracked_)
	{
		bbox_next = tbb_;
		lastbox_ = bbox_next;
	}
	else
		is_last_box_found = false;
}

void TLD::track(const Mat& img1, const Mat& img2)
{
	bboxPoints(lastbox_);
	if (points1_.size() < 1)
	{
		tracked_ = false;
		return;
	}

	tracked_ = trackf2f(img1, img2);
	if (tracked_) bboxPredict(lastbox_, tbb_);
}

void TLD::bboxPoints(const Rect& bbox)
{
	int max_pts = 10;
	int margin_h = 0;
	int margin_w = 0;
	int step_x = ceil((bbox.width - 2 * margin_w) / max_pts);
	int step_y = ceil((bbox.height - 2 * margin_h) / max_pts);
	for (int y = bbox.y + margin_h; y < bbox.y + bbox.height - margin_h; y += step_y)
	{
		for (int x = bbox.x + margin_w; x < bbox.x + bbox.width - margin_w; x += step_x)
		{
			points1_.push_back(Point2f(x, y));
		}
	}
}

void TLD::bboxPredict(const Rect& bbox1, Rect& bbox2)
{
	int num_points = (int)points1_.size();
	vector<float> x_off(num_points);
	vector<float> y_off(num_points);
	for (int i = 0; i < num_points; ++i)
	{
		x_off[i] = points2_[i].x - points1_[i].x;
		y_off[i] = points2_[i].y - points1_[i].y;
	}

	float dx = median(x_off);
	float dy = median(y_off);
	float s;
	if (num_points > 1)
	{
		vector<float> d;
		d.reserve(num_points * (num_points - 1) / 2);
		for (int i = 0; i < num_points; ++i)
		{
			for (int j = i + 1; j < num_points; ++j)
			{
				d.push_back(norm(points2_[i] - points2_[j]) / norm(points1_[i] - points1_[j]));
			}
		}
		s = median(d);
	}
	else
		s = 1.0;

	float s1 = 0.5 * (s - 1) * bbox1.width;
	float s2 = 0.5 * (s - 1) * bbox1.height;
	bbox2.x = round(bbox1.x + dx - s1);
	bbox2.y = round(bbox1.y + dy - s2);
	bbox2.width = round(bbox1.width * s);
	bbox2.height = round(bbox1.height * s);
}

bool TLD::trackf2f(const Mat& img1, const Mat& img2)
{
	//Forward-Backward tracking
	calcOpticalFlowPyrLK(img1, img2, points1_, points2_, status_, similarity_,
		window_size_, level_, term_criteria_, lambda_, 0);
	calcOpticalFlowPyrLK(img2, img1, points2_, pointsFB_, FB_status_, FB_error_, window_size_,
		level_, term_criteria_, lambda_, 0);

	//compute real FB-error
	for (uint i = 0; i < points1_.size(); ++i)
	{
		FB_error_[i] = norm(pointsFB_[i] - points1_[i]);
	}
	//filter out points with FB_error[i] > median(FB_error) &&
	//points with sim_error[i] > median(sim_error)
	normCrossCorrelation(img1, img2);
	return filterPts();
}

void TLD::normCrossCorrelation(const Mat& img1,const Mat& img2)
{
	int sub_img_size = 10;
	Mat rec0(sub_img_size, sub_img_size, CV_8U);
	Mat rec1(sub_img_size, sub_img_size, CV_8U);
	Mat res(1, 1, CV_32F);

	for (uint i = 0; i < points1_.size(); ++i)
	{
		if (status_[i] == 1)
		{
			getRectSubPix(img1, Size(sub_img_size, sub_img_size), points1_[i], rec0);
			getRectSubPix(img2, Size(sub_img_size, sub_img_size), points2_[i], rec1);
			matchTemplate(rec0, rec1, res, CV_TM_CCOEFF_NORMED);
			similarity_[i] = ((float*)(res.data))[0];
		}
		else
			similarity_[i] = 0.0;
	}

	rec0.release();
	rec1.release();
	res.release();
}

bool TLD::filterPts()
{
	//get error medians
	float sim_med = median(similarity_);
	size_t i = 0, k = 0;
	for (; i < points2_.size(); ++i)
	{
		if(!status_[i]) continue;
		if (similarity_[i] >= sim_med)
		{
			points1_[k] = points1_[i];
			points2_[k] = points2_[i];
			FB_error_[k] = FB_error_[i];
			++k;
		}
	}

	if (k == 0) return false;
	points1_.resize(k);
	points2_.resize(k);
	FB_error_.resize(k);
	fbmed_ = median(FB_error_);
	for (i = k = 0; i < points2_.size(); ++i)
	{
		if(!status_[i]) continue;
		if (FB_error_[i] <= fbmed_)
		{
			points1_[k] = points1_[i];
			points2_[k] = points2_[i];
			++k;
		}
	}
	points1_.resize(k);
	points2_.resize(k);
	if (k > 0) return true;
	else return false;
}


