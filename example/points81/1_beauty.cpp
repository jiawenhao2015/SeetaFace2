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
#include "beauty/image_brighten.h"
#include "beauty/commontools.h"
#include "face_track.h"
#include "ncnn_mtcnn.h"

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
    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
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


int g_C = 3 ,g_CMax = 300;
int g_n = 3, g_nMax = 30;
int g_maxCG = 75, g_maxCGMax = 100;
Mat frameOri;
//定义回调函数
void on_bilateralFilterTrackbar(int,void*);
int testmtcnn()
{
    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
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




	FaceTrack tracker;
	std::string modelPath = "/home/jiawenhao/beauty/beautify_linux/models/detection/mtcnn";

	int minFace = 40;
	tracker.initTrack(modelPath, minFace);

	//	VideoWriter video("test3_gulong.mp4", CV_FOURCC('M', 'J', 'P', 'G'), 20.0, Size(1280, 480),1);
	
	while (capture.isOpened())
	{
		
		capture.grab();
		capture.retrieve(frame);
		//frame = imread("/home/jiawenhao/beauty/beautify_linux/imgs/2.jpg");
		if (frame.empty()) break;

		//cv::resize(frame,frame,cv::Size(320,240));
		seeta::cv::ImageData simage = frame;

		double t1 = (double)cv::getTickCount();
		//auto faces = FD.detect(simage);
		vector<Rect> faceRects(1);
		tracker.detectTrackFace(faceRects[0], frame);

		double t2 = (double)cv::getTickCount();
		int dettime = (int)((t2 - t1) * 1000 / cv::getTickFrequency());

		//cv::rectangle(frame, faceRects[0], CV_RGB(128, 128, 255), 3);

		

		vector<SeetaFaceInfo> facesVec;
		SeetaFaceInfo f;
		f.pos.x = faceRects[0].x;
		f.pos.y = faceRects[0].y;
		f.pos.width = faceRects[0].width;
		f.pos.height = faceRects[0].height;
		facesVec.push_back(f);




		SeetaFaceInfoArray  faces;
		faces.size = facesVec.size();
		faces.data = facesVec.data();


		vector<cv::Rect> bbox;
		vector<vector<Point2f>> multiLandmarks;
		#if 1
		for (int i = 0; i < faces.size; ++i)
		{
			auto &face = faces.data[i];
			auto points = FL.mark(simage, face.pos);
			vector<Point2f> landmarks;

			//cv::rectangle(frame, cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height), CV_RGB(128, 128, 255), 3);
			for (int i = 0; i < points.size(); i++)
			{
				auto point = points[i];
				landmarks.push_back(cv::Point(point.x, point.y));
				if(i >= 0 && i <= 8 || i >= 9 && i <= 17)
				{
					//cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(128, 255, 128), -1);
				}
				
				if(i == 69 || i == 72 || i == 77 || i == 80 || i == 34)
				{
				//	cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(255, 0, 0), -1);
				//	printf("-----%d  %d %d\n",i,(int)point.x,(int)point.y);
					
				}
				// cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(255, 0, 0), -1);
				// cv::putText(frame, to_string(i), Point(point.x, point.y),
				// 	cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0));
			}
			bbox.push_back(cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height));
			multiLandmarks.push_back(landmarks);
		}
		#endif

		double t3 = (double)cv::getTickCount();
		int cost_time = (int)((t3 - t1) * 1000 / cv::getTickFrequency());
		totalTime += cost_time;
		cnt++;

		avgTime = totalTime /cnt;
		printf("detect:%d    det&&align:%d  avg:%.2f\n", 
			dettime, cost_time,avgTime);


		//	cv::imshow("Frame", frame);
 
		frameOri = frame.clone();

		#if 1
		//face lift 
		double tfacelift = (double)cv::getTickCount();
		
		beauty::general_face_lift(frame, bbox, multiLandmarks); 

		double tfacelift2 = (double)cv::getTickCount();
		printf("tfacelift %gms\n", (tfacelift2 - tfacelift) * 1000 / cv::getTickFrequency());

		//eye enlarge
		beauty::EyeEnlarge eyeEnlarge;
		eyeEnlarge.generalEyeEnlarge(frame, bbox, multiLandmarks); 
		#endif
		double teyeEnlarge = (double)cv::getTickCount();
		printf("teyeEnlarge %gms\n", (teyeEnlarge - tfacelift2) * 1000 / cv::getTickFrequency());

		//smooth
		beauty::SkinSmooth skin_smooth;
		skin_smooth.smooth(frame,bbox);  

		double tskin_smooth = (double)cv::getTickCount();
		printf("tskin_smooth %gms\n", (tskin_smooth - teyeEnlarge) * 1000 / cv::getTickFrequency());

		printf("alltime %gms\n", (tskin_smooth - t1) * 1000 / cv::getTickFrequency());
		

		//brighten
		Mat dehazeMat;
		TimeStatic(4,NULL);
		enhance::brighten(frameOri, dehazeMat);
		TimeStatic(4,"dehaze");
		//cv::imshow("dehaze", dehazeMat);

		// imwrite("./diff/1_ori_"+ to_string(cnt)+".jpg",frameOri,vector<int>{99});
		// imwrite("./diff/1_our_"+ to_string(cnt)+".jpg",frame,vector<int>{99});
			
		

		// enhance::anotherEnhanced(frameOri,brightenMat);
		// cv::imshow("brighten2", brightenMat);
		// enhance::Enhancement(frameOri,0.8, brightenMat);
		// cv::imshow("brighten3", brightenMat);
		// enhance::BrightnessAndContrastAuto(frameOri, brightenMat,1.5);
		// cv::imshow("brighten4", brightenMat);

		Mat dehazegammaMat;
		enhance::MyGammaCorrection(dehazeMat, dehazegammaMat,0.7);
		//cv::imshow("dehaze+gamma", dehazegammaMat);

		Mat gammaMat;
		enhance::MyGammaCorrection(frameOri, gammaMat,0.7);
		//cv::imshow("MyGammaCorrection", gammaMat);


		Mat gammadehaze;
		enhance::brighten(gammaMat, gammadehaze);
		//cv::imshow("gamma + dehaze", gammadehaze);


		//创建轨迹条
		// createTrackbar("c", "ACE", &g_C, g_CMax, on_bilateralFilterTrackbar);
		// on_bilateralFilterTrackbar(g_C,0);

		// createTrackbar("n", "ACE", &g_n, g_nMax, on_bilateralFilterTrackbar);
		// on_bilateralFilterTrackbar(g_n,0);

		// createTrackbar("g_maxCG", "ACE", &g_maxCG, g_maxCGMax, on_bilateralFilterTrackbar);
		// on_bilateralFilterTrackbar(g_n,0);

		// Mat aceMat;
		// enhance::ACE(frameOri, aceMat);
		// cv::imshow("aceMat", gammaMat);

		//拼接结果对比
		Mat firstRow,secondRow,diff;
		putText(frameOri,"original",Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255),3);
		putText(frame,"contrast",Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255),3);
		putText(dehazeMat,"dehaze",Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255),3);
		putText(gammaMat,"gamma",Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255),3);
		hconcat(frameOri,frame,firstRow);
		hconcat(dehazeMat,gammaMat,secondRow);
		vconcat(firstRow,secondRow,diff);

		//video << diff;




		imshow("diff",diff);
		
	
		

		//video << saveMat;
		

		auto key = cv::waitKey(1);
		if (key == 27)
		{
			break;
		}
	}

	return EXIT_SUCCESS;
}
int testlocalvideo()
{
    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
    int id = 0;
    seeta::ModelSetting FD_model( "../models/fd_2_00.dat", device, id );
    seeta::ModelSetting FL_model( "../models/pd_2_00_pts81.dat", device, id );

	seeta::FaceDetector FD(FD_model);
	seeta::FaceLandmarker FL(FL_model);

	FD.set(seeta::FaceDetector::PROPERTY_VIDEO_STABLE, 1);

	 int camera_id = 0;
	// cv::VideoCapture capture(camera_id);

	cv::VideoCapture capture;
	capture.open("../0.mp4");

	if (!capture.isOpened())
	{
		std::cerr << "Can not open camera(" << camera_id << "), testing image..." << std::endl;
		return test_image(FD, FL);
	}

	auto video_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	auto video_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	auto video_fps = capture.get(CAP_PROP_FPS);
	auto video_fourcc = capture.get(CAP_PROP_FOURCC);

	std::cout << "Open camera(" << camera_id << ")" << std::endl;
	std::cout<<"video_width:"<<video_width<<"	video_height:"<<video_height<<std::endl;

	cv::Mat frame;
	int cnt = 0;
	float avgTime = 0;
	float totalTime = 0;




	FaceTrack tracker;
	std::string modelPath = "/home/jiawenhao/beauty/beautify_linux/models/detection/mtcnn";

	int minFace = 40;
	tracker.initTrack(modelPath, minFace);

	//	VideoWriter video("test_0_no_mopi.mp4", CV_FOURCC('M', 'J', 'P', 'G'), 20.0, Size(1600, 450),1);
	

		VideoWriter video("0919_0.mp4", video_fourcc, video_fps, Size(1600, 450*2),1);
	
	while (capture.isOpened())
	{
		
		capture.grab();
		capture.retrieve(frame);
		//frame = imread("/home/jiawenhao/beauty/beautify_linux/imgs/2.jpg");
		if (frame.empty()) break;



		cv::resize(frame,frame,cv::Size(800,450));
		seeta::cv::ImageData simage = frame;

		double t1 = (double)cv::getTickCount();
		
		vector<Rect> faceRects(1);
		tracker.detectTrackFace(faceRects[0], frame);

		double t2 = (double)cv::getTickCount();
		int dettime = (int)((t2 - t1) * 1000 / cv::getTickFrequency());

		//cv::rectangle(frame, faceRects[0], CV_RGB(128, 128, 255), 3);

		

		vector<SeetaFaceInfo> facesVec;
		SeetaFaceInfo f;
		f.pos.x = faceRects[0].x;
		f.pos.y = faceRects[0].y;
		f.pos.width = faceRects[0].width;
		f.pos.height = faceRects[0].height;
		facesVec.push_back(f);




		SeetaFaceInfoArray  faces;
		faces.size = facesVec.size();
		faces.data = facesVec.data();


		vector<cv::Rect> bbox;
		vector<vector<Point2f>> multiLandmarks;
		#if 0
		for (int i = 0; i < faces.size; ++i)
		{
			auto &face = faces.data[i];
			auto points = FL.mark(simage, face.pos);
			vector<Point2f> landmarks;

			//cv::rectangle(frame, cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height), CV_RGB(128, 128, 255), 3);
			for (int i = 0; i < points.size(); i++)
			{
				auto point = points[i];
				landmarks.push_back(cv::Point(point.x, point.y));
				if(i >= 0 && i <= 8 || i >= 9 && i <= 17)
				{
					//cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(128, 255, 128), -1);
				}
				
				if(i == 69 || i == 72 || i == 77 || i == 80 || i == 34)
				{
				//	cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(255, 0, 0), -1);
				//	printf("-----%d  %d %d\n",i,(int)point.x,(int)point.y);
					
				}
				// cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(255, 0, 0), -1);
				// cv::putText(frame, to_string(i), Point(point.x, point.y),
				// 	cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0));
			}
			bbox.push_back(cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height));
			multiLandmarks.push_back(landmarks);
		}
		#endif

		double t3 = (double)cv::getTickCount();
		int cost_time = (int)((t3 - t1) * 1000 / cv::getTickFrequency());
		totalTime += cost_time;
		cnt++;

		avgTime = totalTime /cnt;
		printf("detect:%d    det&&align:%d  avg:%.2f\n", 
			dettime, cost_time,avgTime);


		//	cv::imshow("Frame", frame);
 
		Mat frameOri = frame.clone();

		#if 0
		//face lift 
		double tfacelift = (double)cv::getTickCount();
		
		beauty::general_face_lift(frame, bbox, multiLandmarks); 

		double tfacelift2 = (double)cv::getTickCount();
		printf("tfacelift %gms\n", (tfacelift2 - tfacelift) * 1000 / cv::getTickFrequency());

		//eye enlarge
		beauty::EyeEnlarge eyeEnlarge;
		eyeEnlarge.generalEyeEnlarge(frame, bbox, multiLandmarks); 
		#endif
		double teyeEnlarge = (double)cv::getTickCount();
		//	printf("teyeEnlarge %gms\n", (teyeEnlarge - tfacelift2) * 1000 / cv::getTickFrequency());

		//smooth
		beauty::SkinSmooth skin_smooth;
		skin_smooth.smooth(frame,bbox);  

		double tskin_smooth = (double)cv::getTickCount();
		printf("tskin_smooth %gms\n", (tskin_smooth - teyeEnlarge) * 1000 / cv::getTickFrequency());

		printf("alltime %gms\n", (tskin_smooth - t1) * 1000 / cv::getTickFrequency());
		

		//	cv::imshow("beautify", frame);
		

		

		//brighten
		Mat dehazeMat;
		TimeStatic(4,NULL);
		enhance::brighten(frameOri, dehazeMat);
		TimeStatic(4,"dehaze");
		cv::imshow("dehaze", dehazeMat);

		// imwrite("./diff/1_ori_"+ to_string(cnt)+".jpg",frameOri,vector<int>{99});
		// imwrite("./diff/1_our_"+ to_string(cnt)+".jpg",frame,vector<int>{99});
			
		

		// enhance::anotherEnhanced(frameOri,brightenMat);
		// cv::imshow("brighten2", brightenMat);
		// enhance::Enhancement(frameOri,0.8, brightenMat);
		// cv::imshow("brighten3", brightenMat);
		// enhance::BrightnessAndContrastAuto(frameOri, brightenMat,1.5);
		// cv::imshow("brighten4", brightenMat);

		Mat dehazegammaMat;
		enhance::MyGammaCorrection(dehazeMat, dehazegammaMat,0.7);
		cv::imshow("dehaze+gamma", dehazegammaMat);



		Mat gammaMat;
		enhance::MyGammaCorrection(frameOri, gammaMat,0.7);
		cv::imshow("MyGammaCorrection", gammaMat);


		Mat gammadehaze;
		enhance::brighten(gammaMat, gammadehaze);
		cv::imshow("gamma + dehaze", gammadehaze);

		//拼接结果对比
		Mat firstRow,secondRow,diff;

		putText(frameOri,"original",Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255),3);
		putText(frame,"contrast",Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255),3);
		putText(dehazeMat,"dehaze",Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255),3);
		putText(gammaMat,"gamma",Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255),3);


		hconcat(frameOri,frame,firstRow);
		hconcat(dehazeMat,gammaMat,secondRow);
		vconcat(firstRow,secondRow,diff);

		video << diff;

		imshow("diff",diff);
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
	//testmtcnn();

	testlocalvideo();
	//testVideo();
}

static int cnt = 1;
void on_bilateralFilterTrackbar(int,void *)
{
    double t1 = (double)cv::getTickCount();
    //bilateralFilter(g_srcImage, g_dstImage, g_ndValue, g_nsigmaColorValue, g_nsigmaSpaceValue);
    Mat aceMat;
	enhance::ACE(frameOri, aceMat, g_C, g_n, g_CMax*1.0/10);
    double t2 = (double)cv::getTickCount();
	//int dettime = (int)((t2 - t1) * 1000 / cv::getTickFrequency());
    printf("%d,current costtime %gms\n", cnt++,((t2 - t1) * 1000 / cv::getTickFrequency()));
    imshow("ACE", aceMat);
	auto key = cv::waitKey(1);
}
