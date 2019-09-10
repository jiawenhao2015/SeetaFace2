#include <math.h>

#include "beauty/face_lift.h"
#include "beauty/commontools.h"
#include <opencv2/highgui/highgui.hpp>


#define MAXNUM 0.5

using namespace std;
using namespace cv;

namespace beauty
{
	/*
	void face_lift(Mat& img, const vector<Point2f>& landmarks, 
		int change, const Rect& face)
	{
		vector<Point2f> control_p = { landmarks[0],
			landmarks[5],
			Point2f(face.x,face.y + face.height),
			Point2f(face.x + face.width,face.y + face.height),
			landmarks[11],
			landmarks[16],
			landmarks[33],
			landmarks[62]
		};
		vector<Point2f> control_q = { landmarks[0],
			landmarks[5],
			Point2f(landmarks[5].x + change,landmarks[5].y),					      
			Point2f(face.x,face.y + face.height),
			Point2f(face.x + face.width,face.y + face.height),
			Point2f(landmarks[11].x - change,landmarks[11].y),
			landmarks[16],
			landmarks[33],
			landmarks[62],
		};

		Mat pic = img.clone();

		for (int i = face.x; i <= face.x + face.width; ++i)
		{
			for (int j = landmarks[0].y; j < landmarks[8].y; ++j)
			{
				{
					vector<float> weight_p;								        
					vector<Point2f>::iterator itcp = control_p.begin();
					while (itcp != control_p.end())
					{
						double tmp;
						if (itcp->x != i || itcp->y != j)
							tmp = 1 / ((itcp->x - i)*(itcp->x - i) + (itcp->y - j)*(itcp->y - j));
						else
							tmp = MAXNUM;
						weight_p.push_back(tmp);
						++itcp;
					}

					double px = 0, py = 0, qx = 0, qy = 0, tw = 0;
					itcp = control_p.begin();

					vector<float>::iterator itwp = weight_p.begin();
					vector<Point2f>::iterator itcq = control_q.begin();
					while (itcp != control_p.end())
					{
						px += (*itwp)*(itcp->x);
						py += (*itwp)*(itcp->y);
						qx += (*itwp)*(itcq->x);
						qy += (*itwp)*(itcq->y);

						tw += *itwp;
						++itcp;
						++itcq;
						++itwp;

					}
					px = px / tw;
					py = py / tw;
					qx = qx / tw;
					qy = qy / tw;

					Mat A = Mat::zeros(2, 1, CV_32FC1);
					Mat B = Mat::zeros(1, 2, CV_32FC1);
					Mat C = Mat::zeros(1, 2, CV_32FC1);
					Mat sumL = Mat::zeros(2, 2, CV_32FC1);
					Mat sumR = Mat::zeros(2, 2, CV_32FC1);
					Mat M, pos;
					for (int i = 0; i < weight_p.size(); ++i)
					{
						A.at<float>(0, 0) = (control_p[i].x - px);
						A.at<float>(1, 0) = (control_p[i].y - py);
						B.at<float>(0, 0) = weight_p[i] * ((control_p[i].x - px));
						B.at<float>(0, 1) = weight_p[i] * ((control_p[i].y - py));
						sumL += A * B;
						C.at<float>(0, 0) = weight_p[i] * (control_q[i].x - qx);
						C.at<float>(0, 1) = weight_p[i] * (control_q[i].y - qy);
						sumR += A * C;
					}
					M = sumL.inv()*sumR;

					B.at<float>(0, 0) = i - px;
					B.at<float>(0, 1) = j - py;
					C.at<float>(0, 0) = qx;
					C.at<float>(0, 1) = qy;
					pos = B * M + C;
					int row = pos.at<float>(0, 0);
					int col = pos.at<float>(0, 1);

					pic.at<Vec3b>(col, row)[0] = img.at<Vec3b>(j, i)[0];
					pic.at<Vec3b>(col, row)[1] = img.at<Vec3b>(j, i)[1];
					pic.at<Vec3b>(col, row)[2] = img.at<Vec3b>(j, i)[2];

				}
			}
		}
		//img = pic.clone();
	}
	*/

	void face_lift(Mat &src, const vector<Point2f>& landmarks, int change, const Rect &face)
	{
		vector<Point2f> control_p = { landmarks[0],
			landmarks[5],
			Point2f(float(face.x),float(face.y + face.height)),
			Point2f(float(face.x + face.width),float(face.y + face.height)),
			landmarks[11],
			landmarks[16],
			landmarks[33],
			landmarks[62]
		};
		vector<Point2f> control_q = { landmarks[0],
			landmarks[5],
			Point2f(landmarks[5].x + change,landmarks[5].y),
			Point2f(float(face.x),float(face.y + face.height)),
			Point2f(float(face.x + face.width),float(face.y + face.height)),
			Point2f(landmarks[11].x - change,landmarks[11].y),
			landmarks[16],
			landmarks[33],
			landmarks[62],
		};

		Mat pic = src.clone();

		for (int i = face.x; i < face.x + face.width; ++i)
		{
			printf("i: %d\n", i);
			for (int j = int(landmarks[0].y); j < int(landmarks[8].y); ++j)
			{
				printf("j: %d\n", j);
				{
					vector<float> weight_p;
					vector<Point2f>::iterator itcp = control_p.begin();
					while (itcp != control_p.end())
					{
						float tmp;
						if (itcp->x != i || itcp->y != j)
							tmp = 1 / ((itcp->x - i)*(itcp->x - i) + (itcp->y - j)*(itcp->y - j));
						else
							tmp = MAXNUM;
						weight_p.push_back(tmp);
						++itcp;
					}

					float px = 0, py = 0, qx = 0, qy = 0, tw = 0;
					itcp = control_p.begin();

					vector<float>::iterator itwp = weight_p.begin();
					vector<Point2f>::iterator itcq = control_q.begin();
					while (itcp != control_p.end())
					{
						px += (*itwp)*(itcp->x);
						py += (*itwp)*(itcp->y);
						qx += (*itwp)*(itcq->x);
						qy += (*itwp)*(itcq->y);

						tw += *itwp;
						++itcp;
						++itcq;
						++itwp;
					}

					px = px / tw;
					py = py / tw;
					qx = qx / tw;
					qy = qy / tw;

					Mat A = Mat::zeros(2, 1, CV_32FC1);
					Mat B = Mat::zeros(1, 2, CV_32FC1);
					Mat C = Mat::zeros(1, 2, CV_32FC1);
					Mat sumL = Mat::zeros(2, 2, CV_32FC1);
					Mat sumR = Mat::zeros(2, 2, CV_32FC1);
					Mat M, pos;
					for (int i = 0; i < weight_p.size(); ++i)
					{
						A.at<float>(0, 0) = (control_p[i].x - px);
						A.at<float>(1, 0) = (control_p[i].y - py);
						B.at<float>(0, 0) = weight_p[i] * ((control_p[i].x - px));
						B.at<float>(0, 1) = weight_p[i] * ((control_p[i].y - py));
						sumL += A * B;
						C.at<float>(0, 0) = weight_p[i] * (control_q[i].x - qx);
						C.at<float>(0, 1) = weight_p[i] * (control_q[i].y - qy);
						sumR += A * C;
					}

					M = sumL.inv()*sumR;

					B.at<float>(0, 0) = i - px;
					B.at<float>(0, 1) = j - py;
					C.at<float>(0, 0) = qx;
					C.at<float>(0, 1) = qy;

					pos = B * M + C;
					int row = int(pos.at<float>(0, 0));
					int col = int(pos.at<float>(0, 1));

					printf("row: %d, col: %d\n", row, col);

					if (row < 0) row = 0;
					if (row >= face.y + face.height) row = face.y + face.height - 1;
					if (col < 0) col = 0;
					if (col >= face.x + face.width) col = face.x + face.width - 1;

					/*pic.at<Vec3b>(col, row)[0] = src.at<Vec3b>(j, i)[0];
					pic.at<Vec3b>(col, row)[1] = src.at<Vec3b>(j, i)[1];
					pic.at<Vec3b>(col, row)[2] = src.at<Vec3b>(j, i)[2];*/

					pic.at<Vec3b>(row, col)[0] = src.at<Vec3b>(j, i)[0];
					pic.at<Vec3b>(row, col)[1] = src.at<Vec3b>(j, i)[1];
					pic.at<Vec3b>(row, col)[2] = src.at<Vec3b>(j, i)[2];
				}

			}
		}
	}

	/*void general_face_lift(Mat& img, const vector<vector<Point2f>>& multiLandmarks,
		int change, const vector<Rect>& faces)
	{
		printf("Size of multiLandmark: %d\n", multiLandmarks.size());
		printf("Size of faces: %d\n", faces.size());

		if (multiLandmarks.size() != faces.size())
		{
			printf("Size of multiLandmark should be equal to size of faces!\n");
			return;
		}

		for (int i = 0; i < multiLandmarks.size(); i++)
		{
			face_lift(img, multiLandmarks[i], change, faces[i]);
		}
	}*/

	Mat local_translation_warp(Mat& img, const int start_x, const int start_y,
		const int end_x, const int end_y, const float radius)
	{
		float radius_pow = radius * radius;
		Mat dst = img.clone();

		imshow("dst",dst);

		//|m-c|^2 in formula
		//float ddmc = (end_x - start_x) * (end_x - start_x) + (end_y - start_y) * (end_y - start_y);
		float ddmc = float(powf((end_x - start_x)*1.0f, 2) + powf((end_y - start_y)*1.0, 2));
		for (int j = 0; j < img.rows; ++j)
		{
			for (int i = 0; i < img.cols; ++i)
			{
				//is point located in range
				//which correspond a rectangle (start_x ,start_y)
				if (fabs(i - start_x) > radius && fabs(j - start_y) > radius)
					continue;

				//float distance = (i - start_x) * (i - start_x) + (j - start_y) * (j - start_y);
				float distance = float(pow(i - start_x, 2) + pow(j - start_y, 2));
				if (distance < radius_pow)
				{
					float ratio = (radius_pow - distance) / (radius_pow - distance + ddmc);
					ratio *= ratio;

					//mapping
					float ux = i - ratio * (end_x - start_x);
					float uy = j - ratio * (end_y - start_y);

					vector<int> insert_val = bilinear_insert(img, ux, uy);


					
					if (insert_val.size() != 3 && insert_val.size() != 1)
						continue;

					if (insert_val.size() == 3)
					{
						for (int ch = 0; ch < insert_val.size(); ++ch)
							dst.at<Vec3b>(j, i)[ch] = insert_val[ch];
					}
					else
					{
						dst.at<uchar>(j, i) = insert_val[0];
					}
				}
			}
		}
		return dst;
	}
	vector<int> bilinear_insert(const Mat& img, const float ux, const float uy)
	{
		vector<int> result;

		if (img.channels() != 3 && img.channels() != 1)
		{			 
			return result;
		}
			

		int x1 = int(ux);
		int y1 = int(uy);

		x1 = x1 < 0 ? 0 : x1;
		y1 = y1 < 0 ? 0 : y1;
		x1 = x1 >= img.cols ? img.cols - 1 : x1;
		y1 = y1 >= img.rows ? img.rows - 1 : y1;

		int x2 = x1 + 1;
		int y2 = y1 + 1;
		x2 = x2 >= img.cols ? img.cols - 1 : x2;
		y2 = y2 >= img.rows ? img.rows - 1 : y2;

		if (x2 >= img.cols || y2 >= img.rows || x1 < 0 || y1 < 0)
		{
			printf("height,width: %d %d\n", img.rows, img.cols);
			printf("x1,y1: %d %d\n", x1, y1);
			printf("x2,y2: %d %d\n", x2, y2);
			printf("x2 or y2 exceeds out img.rows or img.cols\n");
		}

		if (img.channels() == 3)
		{
			for (int i = 0; i < img.channels(); i++)
			{
				float part1 = img.at<Vec3b>(y1, x1)[i] * (float(x2) - ux) * (float(y2) - uy);
				float part2 = img.at<Vec3b>(y1, x2)[i] * (ux - float(x1)) * (float(y2) - uy);
				float part3 = img.at<Vec3b>(y2, x1)[i] * (float(x2) - ux) * (uy - float(y1));
				float part4 = img.at<Vec3b>(y2, x2)[i] * (ux - float(x1)) * (uy - float(y1));

				float insert_val = part1 + part2 + part3 + part4;

				result.push_back(int(insert_val));
			}
		}
		else
		{
			float part1 = img.at<uchar>(y1, x1) * (float(x2) - ux) * (float(y2) - uy);
			float part2 = img.at<uchar>(y1, x2) * (ux - float(x1)) * (float(y2) - uy);
			float part3 = img.at<uchar>(y2, x1) * (float(x2) - ux) * (uy - float(y1));
			float part4 = img.at<uchar>(y2, x2) * (ux - float(x1)) * (uy - float(y1));

			float insert_val = part1 + part2 + part3 + part4;

			result.push_back(int(insert_val));
		}
		return result;
	}

	Mat face_lift(Mat& img,const vector<Point2f>& landmarks)
	{
		if (landmarks.size() == 0) return img;

		// Point2f left_landmark = landmarks[3];
		// Point2f left_landmark_down = landmarks[6];
		// Point2f right_landmark = landmarks[10];
		// Point2f right_landmark_down = landmarks[13];
		// Point2f end_point = landmarks[30];

		Point2f left_landmark = landmarks[69];
		Point2f left_landmark_down = landmarks[72];
		Point2f right_landmark = landmarks[80];
		Point2f right_landmark_down = landmarks[77];
		Point2f end_point = landmarks[34];

		//calculate distance of 4th point to 6/7th point as left face-lift distance
		float left_lift = sqrtf(powf(left_landmark.x - left_landmark_down.x, 2) +
			powf(left_landmark.y - left_landmark_down.y, 2));

		//calculate distance of 12th point to 16th point as right face-lift distance
		float right_lift = sqrtf(powf(right_landmark.x - right_landmark_down.x, 2) +
			powf(right_landmark.y - left_landmark_down.y, 2));

		TimeStatic(1,NULL);
		//left face lift
		Mat inter_res = local_translation_warp(img, int(left_landmark.x), int(left_landmark.y),
			int(end_point.x), int(end_point.y), left_lift);

		//right face lift
		Mat dst = local_translation_warp(inter_res, int(right_landmark.x), int(right_landmark.y),
			int(end_point.x), int(end_point.y), right_lift);
		TimeStatic(1,"inner facelift");
		return dst;
	}

	void get_offset_landmarks(const vector<Point2f>& landmarks,const Point2f& start_point,
		vector<Point2f>& offset_landmaks)
	{
		for (int i = 0; i < landmarks.size(); ++i)
		{
			Point2f tmp = landmarks[i];
			tmp.x = tmp.x - start_point.x > 0 ? tmp.x - start_point.x : 0;
			tmp.y = tmp.y - start_point.y > 0 ? tmp.y - start_point.y : 0;
			offset_landmaks.push_back(tmp);
		}
	}

	void general_face_lift(Mat& img, const vector<Rect>& faces,
		const vector<vector<Point2f>>& multiLandmarks)
	{
		if (faces.size() != multiLandmarks.size())
		{
			printf("Size of faces should be equal to Size of multiLandmarks!\n");
			return;
		}

		for (int i = 0; i < faces.size(); ++i)
		{
			Point2f top_left = Point2f(float(faces[i].x), float(faces[i].y));
			Point2f right_down = Point2f(faces[i].x + float(faces[i].width), faces[i].y + float(faces[i].height));

			// printf("int(top_left.y), int(right_down.y),int(top_left.x), int(right_down.x) %d %d %d %d\n",int(top_left.y), int(right_down.y), int(top_left.x), int(right_down.x));
			// printf("%d %d\n",img.rows,img.cols);
			Mat roi = img(Range(int(top_left.y), int(right_down.y)), Range(int(top_left.x), int(right_down.x)));

			//to replace gap between roi and source image
			/*int border_width = 50;
			int border_start_x = int(right_down.x - border_width);
			border_start_x = border_start_x > 0 ? border_start_x : 0;
			int border_end_x = int(right_down.x + border_width);
			border_end_x = border_end_x < img.cols ? border_end_x : img.cols - 1;
			Mat border_overlay = img(Range(int(top_left.y), int(right_down.y)),
				Range(int(border_start_x), int(border_end_x)));*/

			vector<Point2f> offset_landmarks;
			get_offset_landmarks(multiLandmarks[i], top_left,offset_landmarks);
			TimeStatic(0,NULL);
			Mat dst = face_lift(roi, offset_landmarks);
			TimeStatic(0,"face_lift");
			dst.copyTo(roi);


			/*Mat tmp = border_overlay.clone();
			tmp.copyTo(border_overlay);*/
		}
	}


} //end namespace beauty