#pragma once
#include <opencv2/core/core.hpp>



//https://github.com/rzwm/IBEABFHR
//https://blog.csdn.net/u012494876/article/details/80637323
//Image Brightness Enhancement Automatically Based on Fast Haze Removal
namespace enhance
{
    class ImageBrighten
    {
    public:
        void brighten(const cv::Mat& src, cv::Mat& dst);

    private:
        void fastHazeRemoval(const cv::Mat& src, cv::Mat& dst);
        void fastHazeRemoval_1Channel(const cv::Mat& src, cv::Mat& dst);
        void fastHazeRemoval_3Channel(const cv::Mat& src, cv::Mat& dst);
    };

    // wrapper
    void brighten(const cv::Mat& src, cv::Mat& dst);
 


    void anotherEnhanced(const cv::Mat src,cv::Mat& dst);

    void Enhancement(const cv::Mat image, double alpha,cv::Mat image_enhancement);

    void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent);
    void MyGammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma);

}
