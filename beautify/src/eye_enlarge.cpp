#include "beauty/face_lift.h"
#include "beauty/eye_enlarge.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace beauty
{
	void EyeEnlarge::enlarge(const Mat& face_img,Mat& dst, const Point2f& center_point)
	{
		if (face_img.size() != dst.size())
		{
			printf("Size of face_img should be equal to dst!\n");
			return;
		}

		if (face_img.channels() != 3 && face_img.channels() != 1)
		{
			printf("Channels of face_img and dst should be 1 or 3!\n");
			printf("This happens in line %d in file %s!\n", __LINE__, __FILE__);
			return;
		}

		int left = center_point.x - radius_ < 0 ? 0 : int(center_point.x - radius_);
		int top = center_point.y - radius_ < 0 ? 0 : int(center_point.y - radius_);
		int right = center_point.x + radius_ >= face_img.cols ?  
			face_img.cols - 1 : int(center_point.x + radius_);
		int bottom = center_point.y + radius_ >= face_img.rows ?
			face_img.rows - 1 : int(center_point.y + radius_);

		float pow_radius = radius_ * radius_;

		for (int y = top; y < bottom; ++y)
		{
			float offset_y = y - center_point.y;
			for (int x = left; x < right; ++x)
			{
				float offset_x = x - center_point.x;
				float offset_xy = offset_x*offset_x + offset_y*offset_y;
				if (offset_xy <= pow_radius)
				{
					float scale_factor = 1.0f - offset_xy / pow_radius;
					scale_factor = 1.0f - strength_ / 100.0f *scale_factor;

				//	printf("scale_factor: %f   ---%f\n", scale_factor, offset_xy / pow_radius);

					float pos_x = offset_x * scale_factor + center_point.x;
					float pos_y = offset_y * scale_factor + center_point.y;
					pos_x = pos_x < 0 ? 0 : pos_x;
					pos_y = pos_y < 0 ? 0 : pos_y;

					//use bilinear insert method
					if (use_bilinear_ && 0)
					{
						vector<int> insert_val = bilinear_insert(face_img, pos_x, pos_y);
						if (face_img.channels() == 1 && insert_val.size() == 1)
						{
							dst.at<uchar>(y, x) = insert_val[0];
							return;
						}
							
						if (face_img.channels() == 3 && insert_val.size() == 3)
						{
							for (int ch = 0; ch < face_img.channels(); ++ch)
								dst.at<Vec3b>(y, x)[ch] = insert_val[ch];
							return;
						}

					//	printf("face_img: %d   ---%d\n", face_img.channels(), insert_val.size());
					}
					// printf("use_bilinear_: %d\n", use_bilinear_);
					// assert(0);

					//use default method
					int int_pos_x = int(pos_x);
					int int_pos_y = int(pos_y);
					int_pos_x = int_pos_x >= face_img.cols ? face_img.cols - 1 : int_pos_x;
					int_pos_y = int_pos_y >= face_img.rows ? face_img.rows - 1 : int_pos_y;
					//printf("int_pos_x: %d,int_pos_y: %d\n", int_pos_x, int_pos_y);
					//printf("face_img_h: %d,face_img_w: %d\n", face_img.rows, face_img.cols);

					if (face_img.channels() == 1)
						dst.at<uchar>(y, x) = face_img.at<uchar>(int_pos_y, int_pos_x);
					else
					{
						for (int ch = 0; ch < face_img.channels(); ++ch)
						{
							dst.at<Vec3b>(y, x)[ch] = face_img.at<Vec3b>(int_pos_y, int_pos_x)[ch];
						}
					}
				}
			}
		}
	}

	bool is_number_of_landmark_68(const vector<Point2f>& landmarks)
	{
		if (landmarks.size() != 68)
		{
			printf("Number of landmarks should be 68!\n");
			return false;
		}
		return true;
	}

	void EyeEnlarge::getLeftCenterPoint(const vector<Point2f>& offset_landmarks, Point2f& left_center_point)
	{
	//	if (!is_number_of_landmark_68(offset_landmarks)) return;

		//left eye landmark index: 37-42 => 36-41(start from 0)
		//use mean value of (x_i,y_i) as left_center_point, 36 <= i <= 41
		float sum_x = 0.f, sum_y = 0.f;
		int i = 0;
		for (int idx = 1; idx <= 8; ++idx)
		{
			sum_x += offset_landmarks[idx].x;
			sum_y += offset_landmarks[idx].y;
			++i;
		}

		left_center_point.x = float(sum_x / i);
		left_center_point.y = float(sum_y / i);
		//printf("left_center_point: %.2f %.2f\n", left_center_point.x, left_center_point.y);
	}

	void EyeEnlarge::getRightCenterPoint(const vector<Point2f>& offset_landmarks, Point2f& right_center_point)
	{
		//if (!is_number_of_landmark_68(offset_landmarks)) return;

		//right eye landmark index: 43-48 => 42-47(start from 0)
		//use mean value of (x_i,y_i) as right_center_point, 42 <= i <= 47
		float sum_x = 0.f, sum_y = 0.f;
		int i = 0;
		for (int idx = 10; idx <= 17; ++idx)
		{
			sum_x += offset_landmarks[idx].x;
			sum_y += offset_landmarks[idx].y;
			++i;
		}

		right_center_point.x = float(sum_x / i);
		right_center_point.y = float(sum_y / i);
		//printf("right_center_point: %.2f %.2f\n", right_center_point.x, right_center_point.y);
	}

	void EyeEnlarge::singleLandmarkEyeEnlarge(Mat& face_img, const vector<Point2f>& offset_landmarks)
	{
		Point2f left_center_point, right_center_point;
		getLeftCenterPoint(offset_landmarks, left_center_point);
		getRightCenterPoint(offset_landmarks, right_center_point);


		// cv::circle(face_img, left_center_point, 3, CV_RGB(255, 0, 0), -1);
		// cv::circle(face_img, right_center_point, 3, CV_RGB(255, 0, 0), -1);
		// imshow("face_img",face_img);
		// waitKey(1);


		Mat dst_face = face_img.clone();
		enlarge(face_img, dst_face, left_center_point);
		Mat final_face = dst_face.clone();
		enlarge(dst_face, final_face, right_center_point);

		face_img = final_face.clone();
	}

	void EyeEnlarge::getOffsetLandmarks(const vector<Point2f>& landmarks,
		const Rect& face, vector<Point2f>& offset_landmarks)
	{
		Point2f start_point = Point2f(float(face.x), float(face.y));
		offset_landmarks.clear();
		get_offset_landmarks(landmarks, start_point, offset_landmarks);
	}

	void EyeEnlarge::generalEyeEnlarge(Mat& img, const vector<Rect>& faces,
		const vector<vector<Point2f>>& multiLandmarks)
	{
		if (faces.size() != multiLandmarks.size())
		{
			printf("Size of faces should be equal to size of multiLandmarks!\n");
			printf("This happens in line %d in file %s!\n", __LINE__, __FILE__);
			return;
		}

		for (int i = 0; i < faces.size(); ++i)
		{
			vector<Point2f> offset_landmarks;
			getOffsetLandmarks(multiLandmarks[i], faces[i], offset_landmarks);
			Point top_left = Point(faces[i].x, faces[i].y);
			Point right_down = Point(faces[i].x + faces[i].width, faces[i].y + faces[i].width);
			
			top_left.x = max(0, top_left.x);
			top_left.x = min(img.cols -1, top_left.x);
			top_left.y = max(0, top_left.y);
			top_left.y = min(img.rows - 1, top_left.y);


			right_down.x = max(0, right_down.x);
			right_down.x = min(img.cols -1, right_down.x);
			right_down.y = max(0, right_down.y);
			right_down.y = min(img.rows - 1, right_down.y);

			Mat face_img = img(Range(top_left.y, right_down.y), Range(top_left.x, right_down.x));
			Mat processed_face_img = face_img.clone();
			singleLandmarkEyeEnlarge(processed_face_img, offset_landmarks);
			processed_face_img.copyTo(face_img);

			offset_landmarks.clear();
		}
	}

} //end namespace beauty