#pragma warning(disable: 4819)

#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>

#include <seeta/Struct_cv.h>
#include <seeta/Struct.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/core/core.hpp>



#include "beauty/face_lift.h"
#include "beauty/skin_smooth.h"
#include "beauty/eye_enlarge.h"


#include <array>

#include <vector>
#include <map>
#include <iostream>

using namespace std;
using namespace cv;


int test_image(seeta::FaceDetector &FD, seeta::FaceLandmarker &FL)
{
	std::string image_path = "1.jpg";
	std::cout << "Loading image: " << image_path << std::endl;
	auto frame = cv::imread(image_path);
	seeta::cv::ImageData simage = frame;

	if (simage.empty()) {
		std::cerr << "Can not open image: " << image_path << std::endl;
		return EXIT_FAILURE;
	}


	auto faces = FD.detect(simage);

	for (int i = 0; i < faces.size; ++i)
	{
		auto &face = faces.data[i];
		auto points = FL.mark(simage, face.pos);

		cv::rectangle(frame, cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height), CV_RGB(128, 128, 255), 3);
		for (auto &point : points)
		{
			cv::circle(frame, cv::Point(point.x, point.y), 3, CV_RGB(128, 255, 128), -1);
		}
	}

	auto output_path = image_path + ".pts81.png";
	cv::imwrite(output_path, frame);
	std::cerr << "Saving result into: " << output_path << std::endl;

	return EXIT_SUCCESS;
}

int testVideo()
{
    seeta::ModelSetting::Device device = seeta::ModelSetting::GPU;
    int id = 0;
    seeta::ModelSetting FD_model( "../models/fd_2_00.dat", device, id );
    seeta::ModelSetting FL_model( "../models/pd_2_00_pts81.dat", device, id );

	seeta::FaceDetector FD(FD_model);
	seeta::FaceLandmarker FL(FL_model);

	FD.set(seeta::FaceDetector::PROPERTY_VIDEO_STABLE, 1);
	//FD.set(seeta::FaceDetector::PROPERTY_THRESHOLD3, 0.8);

	int camera_id = 0;
	cv::VideoCapture capture(camera_id);
	if (!capture.isOpened())
	{
		std::cerr << "Can not open camera(" << camera_id << "), testing image..." << std::endl;
		return test_image(FD, FL);
	}

	auto video_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	auto video_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	std::cout << "Open camera(" << camera_id << ")" << std::endl;
	std::cout<<"video_width:"<<video_width<<"	video_height:"<<video_height<<std::endl;

	cv::Mat frame;
	int cnt = 0;
	float avgTime = 0;
	float totalTime = 0;
	while (capture.isOpened())
	{
		
		capture.grab();
		capture.retrieve(frame);

		if (frame.empty()) break;

	//	cv::resize(frame,frame,cv::Size(320,240));
		seeta::cv::ImageData simage = frame;

		double t1 = (double)cv::getTickCount();
		auto faces = FD.detect(simage);

		double t2 = (double)cv::getTickCount();
		int dettime = (int)((t2 - t1) * 1000 / cv::getTickFrequency());


		vector<cv::Rect> bbox;
		vector<vector<Point2f>> multiLandmarks;
		for (int i = 0; i < faces.size; ++i)
		{
			auto &face = faces.data[i];
			auto points = FL.mark(simage, face.pos);
			vector<Point2f> landmarks;

			cv::rectangle(frame, cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height), CV_RGB(128, 128, 255), 3);
			for (int i = 0; i < points.size(); i++)
			{
				auto point = points[i];
				landmarks.push_back(cv::Point(point.x, point.y));
				if(i >= 0 && i <= 8 || i >= 9 && i <= 17)
				{
					cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(128, 255, 128), -1);
				}
				
				if(i == 69 || i == 72 || i == 77 || i == 80 || i == 34)
				{
					cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(255, 0, 0), -1);
				//	printf("-----%d  %d %d\n",i,(int)point.x,(int)point.y);
					
				}
				// cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(255, 0, 0), -1);
				// cv::putText(frame, to_string(i), Point(point.x, point.y),
				// 	cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0));
			}
			bbox.push_back(cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height));
			multiLandmarks.push_back(landmarks);
		}


		double t3 = (double)cv::getTickCount();
		int cost_time = (int)((t3 - t1) * 1000 / cv::getTickFrequency());
		totalTime += cost_time;
		cnt++;

		avgTime = totalTime /cnt;
		printf("detect:%d    det&&align:%d  avg:%.2f\n", 
			dettime, cost_time,avgTime);


		cv::imshow("Frame", frame);
 


		#if 1
		//face lift 
		double tfacelift = (double)cv::getTickCount();
		
		beauty::general_face_lift(frame, bbox, multiLandmarks); 

		double tfacelift2 = (double)cv::getTickCount();
		printf("tfacelift %gms\n", (tfacelift2 - tfacelift) * 1000 / cv::getTickFrequency());

		//eye enlarge
		beauty::EyeEnlarge eyeEnlarge;
		eyeEnlarge.generalEyeEnlarge(frame, bbox, multiLandmarks); 

		double teyeEnlarge = (double)cv::getTickCount();
		printf("teyeEnlarge %gms\n", (teyeEnlarge - tfacelift2) * 1000 / cv::getTickFrequency());

		//smooth
		beauty::SkinSmooth skin_smooth;
		skin_smooth.smooth(frame,bbox);  

		double tskin_smooth = (double)cv::getTickCount();
		printf("tskin_smooth %gms\n", (tskin_smooth - teyeEnlarge) * 1000 / cv::getTickFrequency());

		printf("alltime %gms\n", (tskin_smooth - t1) * 1000 / cv::getTickFrequency());
	#endif

		cv::imshow("beautify", frame);

		auto key = cv::waitKey(1);
		if (key == 27)
		{
			break;
		}
	}

	return EXIT_SUCCESS;
}

int main()
{
testVideo();
}