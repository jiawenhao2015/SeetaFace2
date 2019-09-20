#include "beauty/image_brighten.h"
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void enhance::ImageBrighten::brighten(const cv::Mat &src, cv::Mat &dst)
{
    CV_Assert(src.type() == CV_8UC3 || src.type() == CV_8UC1);

    cv::Mat src_inverse = ~src;

    cv::Mat temp;
    fastHazeRemoval(src_inverse, temp);

    dst = ~temp;
}

void enhance::ImageBrighten::fastHazeRemoval(const cv::Mat &src, cv::Mat &dst)
{
    CV_Assert(src.type() == CV_8UC3 || src.type() == CV_8UC1);

    if (src.channels() == 1)
    {
        fastHazeRemoval_1Channel(src, dst);
    }
    else
    {
        fastHazeRemoval_3Channel(src, dst);
    }
}

void enhance::ImageBrighten::fastHazeRemoval_1Channel(const cv::Mat &src, cv::Mat &dst)
{
    CV_Assert(src.type() == CV_8UC1);

    // step 1. input H(x)
    const cv::Mat &H = src;

    // step 2. calc M(x)
    const cv::Mat &M = H;

    // step 3. calc M_ave(x)
    cv::Mat M_ave;
    const int radius = std::max(50, std::max(H.rows, H.cols) / 20); // should not be too small, or there will be a halo artifact
    cv::boxFilter(M, M_ave, -1, cv::Size(2 * radius + 1, 2 * radius + 1));

    // step 4. calc m_av
    const float m_av = float(cv::mean(M)[0] / 255.0);

    // step 5. calc L(x)
    const float p = 1.0f - m_av + 0.9f; // a simple parameter selection strategy, for reference only
    const float coeff = std::min(p * m_av, 0.9f);
    cv::Mat L(H.size(), CV_32FC1);
    for (int y = 0; y < L.rows; ++y)
    {
        const uchar *M_line = M.ptr<uchar>(y);
        const uchar *M_ave_line = M_ave.ptr<uchar>(y);
        float *L_line = L.ptr<float>(y);
        for (int x = 0; x < L.cols; ++x)
        {
            L_line[x] = std::min(coeff * M_ave_line[x], float(M_line[x]));
        }
    }

    // step 6. calc A
    double max_H = 0.0;
    cv::minMaxLoc(H, nullptr, &max_H);
    double max_M_ave = 0.0;
    cv::minMaxLoc(M_ave, nullptr, &max_M_ave);
    const float A = 0.5f * float(max_H) + 0.5f * float(max_M_ave);

    // step 7. get F(x)
    cv::Mat F(H.size(), CV_8UC1);
    for (int y = 0; y < F.rows; ++y)
    {
        const uchar *H_line = H.ptr<uchar>(y);
        const float *L_line = L.ptr<float>(y);
        uchar *F_line = F.ptr<uchar>(y);
        for (int x = 0; x < F.cols; ++x)
        {
            const float l = L_line[x];
            const float factor = 1.0f / (1.0f - l / A);
            F_line[x] = cv::saturate_cast<uchar>((float(H_line[x]) - l) * factor);
        }
    }

    dst = F;
}

void enhance::ImageBrighten::fastHazeRemoval_3Channel(const cv::Mat &src, cv::Mat &dst)
{
    CV_Assert(src.type() == CV_8UC3);

    // step 1. input H(x)
    const cv::Mat &H = src;

    // step 2. calc M(x)
    cv::Mat M(H.size(), CV_8UC1);
    uchar max_H = 0; // used in step 6
    for (int y = 0; y < M.rows; ++y)
    {
        const cv::Vec3b *H_line = H.ptr<cv::Vec3b>(y);
        uchar *M_line = M.ptr<uchar>(y);
        for (int x = 0; x < M.cols; ++x)
        {
            const cv::Vec3b &h = H_line[x];
            M_line[x] = std::min(h[2], std::min(h[0], h[1]));
            max_H = std::max(std::max(h[0], h[1]), std::max(h[2], max_H));
        }
    }

    // step 3. calc M_ave(x)
    cv::Mat M_ave;
    const int radius = std::max(50, std::max(H.rows, H.cols) / 20); // should not be too small, or there will be a halo artifact
    cv::boxFilter(M, M_ave, -1, cv::Size(2 * radius + 1, 2 * radius + 1));

    // step 4. calc m_av
    const float m_av = float(cv::mean(M)[0] / 255.0);

    // step 5. calc L(x)
    // const float p = 1.0f - m_av + 0.9f; // a simple parameter selection strategy, for reference only
    // const float coeff = std::min(p * m_av, 0.9f);

    const float p = 1.0f - m_av + 0.5f; // a simple parameter selection strategy, for reference only
    const float coeff = std::min(p * m_av, 0.5f);

    cv::Mat L(H.size(), CV_32FC1);
    for (int y = 0; y < L.rows; ++y)
    {
        const uchar *M_line = M.ptr<uchar>(y);
        const uchar *M_ave_line = M_ave.ptr<uchar>(y);
        float *L_line = L.ptr<float>(y);
        for (int x = 0; x < L.cols; ++x)
        {
            L_line[x] = std::min(coeff * M_ave_line[x], float(M_line[x]));
        }
    }

    // step 6. calc A
    double max_M_ave = 0.0;
    cv::minMaxLoc(M_ave, nullptr, &max_M_ave);
    const float A = 0.5f * max_H + 0.5f * float(max_M_ave);

    // step 7. get F(x)
    cv::Mat F(H.size(), CV_8UC3);
    for (int y = 0; y < F.rows; ++y)
    {
        const cv::Vec3b *H_line = H.ptr<cv::Vec3b>(y);
        const float *L_line = L.ptr<float>(y);
        cv::Vec3b *F_line = F.ptr<cv::Vec3b>(y);
        for (int x = 0; x < F.cols; ++x)
        {
            const cv::Vec3b &h = H_line[x];
            const float l = L_line[x];
            cv::Vec3b &f = F_line[x];
            const float factor = 1.0f / (1.0f - l / A);
            f[0] = cv::saturate_cast<uchar>((float(h[0]) - l) * factor);
            f[1] = cv::saturate_cast<uchar>((float(h[1]) - l) * factor);
            f[2] = cv::saturate_cast<uchar>((float(h[2]) - l) * factor);
        }
    }

    dst = F;
}

void enhance::brighten(const cv::Mat &src, cv::Mat &dst)
{
    ImageBrighten().brighten(src, dst);
}

///
namespace enhance
{

void calhistOfRGB(cv::Mat &src)
{

    cv::Mat dst;

    //分割图像为3个通道即：B, G and R
    std::vector<cv::Mat> bgr_planes;
    split(src, bgr_planes);

    //创建箱子的数目
    int histSize = 256;

    //设置范围 ( for B,G,R) )
    float range[] = {0, 256}; //不包含上界256
    const float *histRange = {range};

    //归一化，起始位置直方图清除内容
    bool uniform = true;
    bool accumulate = false;

    cv::Mat b_hist, g_hist, r_hist;

    //计算每个平面的直方图
    //&bgr_planes[]原数组，1原数组个数，0只处理一个通道，
    //Mat()用于处理原来数组的掩膜，b_hist将要用来存储直方图的Mat对象
    //1直方图的空间尺寸，histsize每一维的箱子数目，histrange每一维的变化范围
    //uniform和accumulate箱子的大小一样，直方图开始的位置清除内容

    calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    //画直方图（ B, G and R）
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    //归一化结果为 [ 0, histImage.rows ]
    //b_hist输入数组，b_hist输出数组，
    //0和histImage.rows归一化的两端限制值，
    //NORM_MINMAX归一化类型 -1输出和输入类型一样，Mat()可选掩膜
    normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    //为每个通道画图
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
             cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
             cv::Scalar(255, 0, 0), 2, 8, 0);

        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
             cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
             cv::Scalar(0, 255, 0), 2, 8, 0);

        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
             cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
             cv::Scalar(0, 0, 255), 2, 8, 0);
    }
}

float GetGamma(cv::Mat &src)
{
    CV_Assert(src.data);
    CV_Assert(src.depth() != sizeof(uchar));

    int height = src.rows;
    int width = src.cols;
    long size = height * width;

    //!< histogram
    float histogram[256] = {0};
    uchar pvalue = 0;
    cv::MatIterator_<uchar> it, end;
    for (it = src.begin<uchar>(), end = src.end<uchar>(); it != end; it++)
    {
        pvalue = (*it);
        histogram[pvalue]++;
    }

    int threshold = 0;       //otsu阈值
    long sum0 = 0, sum1 = 0; //前景的灰度总和和背景灰度总和
    long cnt0 = 0, cnt1 = 0; //前景的总个数和背景的总个数

    double w0 = 0, w1 = 0;  //前景和背景所占整幅图像的比例
    double u0 = 0, u1 = 0;  //前景和背景的平均灰度
    double u = 0;           //图像总平均灰度
    double variance = 0;    //前景和背景的类间方差
    double maxVariance = 0; //前景和背景的最大类间方差

    int i, j;
    for (i = 1; i < 256; i++) //一次遍历每个像素
    {
        sum0 = 0;
        sum1 = 0;
        cnt0 = 0;
        cnt1 = 0;
        w0 = 0;
        w1 = 0;
        for (j = 0; j < i; j++)
        {
            cnt0 += histogram[j];
            sum0 += j * histogram[j];
        }

        u0 = (double)sum0 / cnt0;
        w0 = (double)cnt0 / size;

        for (j = i; j <= 255; j++)
        {
            cnt1 += histogram[j];
            sum1 += j * histogram[j];
        }

        u1 = (double)sum1 / cnt1;
        w1 = 1 - w0; // (double)cnt1 / size;

        u = u0 * w0 + u1 * w1;

        //variance =  w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);
        variance = w0 * w1 * (u0 - u1) * (u0 - u1);

        if (variance > maxVariance)
        {
            maxVariance = variance;
            threshold = i;
        }
    }

    // convert threshold to gamma.
    float gamma = 0.0;
    gamma = threshold / 255.0;

    // return
    return gamma;
}

void GammaCorrection(cv::Mat &src, cv::Mat &dst, float fGamma)
{
    CV_Assert(src.data);

    // accept only char type matrices
    CV_Assert(src.depth() != sizeof(uchar));

    // build look up table
    unsigned char lut[256];
    for (int i = 0; i < 256; i++)
    {
        lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
    }

    // case 1 and 3 for different channels
    dst = src.clone();
    const int channels = dst.channels();
    switch (channels)
    {
    case 1:
    {

        cv::MatIterator_<uchar> it, end;
        for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
            *it = lut[(*it)];

        break;
    }
    case 3:
    {

        cv::MatIterator_<cv::Vec3b> it, end;
        for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)
        {
            (*it)[0] = lut[((*it)[0])]; // B
            (*it)[1] = lut[((*it)[1])]; // G
            (*it)[2] = lut[((*it)[2])]; // R
        }
        break;
    }
    } // end for switch
}

void anotherEnhanced(const cv::Mat img, cv::Mat &dst)
{
    if (!img.data)
    {
        return;
    }

    int height = img.rows;
    int width = img.cols;
    cv::Mat simg = img.clone();

    //cv::resize(img, simg, cv::Size(height/4, width/4));

    // R,G,B 通道分析
    calhistOfRGB(simg);

    // Gray
    cv::Mat Grayimg;
    cvtColor(simg, Grayimg, cv::COLOR_BGR2GRAY);

    float gamma = GetGamma(Grayimg);

    GammaCorrection(simg, dst, gamma);
}

//另外一个增强方法
void Enhancement(const cv::Mat image, double alpha, cv::Mat image_enhancement)
{

    image.copyTo(image_enhancement);
    for (int i = 1; i < image.rows - 1; i++)
        for (int j = 1; j < image.cols - 1; j++)
            image_enhancement.at<uchar>(i, j) = image.at<uchar>(i, j) + 4 * alpha * (image.at<uchar>(i, j) - (image.at<uchar>(i + 1, j) + image.at<uchar>(i - 1, j) + image.at<uchar>(i, j + 1) + image.at<uchar>(i, j - 1)) / 4);
}

/****************************************/
/*   实现自动对比度的函数                  */
/*   目前只有前后中通道调用                */
/*   彩色的已经加入到了函数内部             */
/*原文链接：https://blog.csdn.net/u010684134/article/details/69246115*/
//貌似不管事儿呢。。
/*****************************************/
void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent)
{
    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    int histSize = 256;
    float alpha, beta;
    double minGray = 0, maxGray = 0;

    //to calculate grayscale histogram
    cv::Mat gray;
    if (src.type() == CV_8UC1)
        gray = src;
    else if (src.type() == CV_8UC3)
        cvtColor(src, gray, CV_BGR2GRAY);
    else if (src.type() == CV_8UC4)
        cvtColor(src, gray, CV_BGRA2GRAY);
    if (clipHistPercent == 0)
    {
        // keep full available range
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        cv::Mat hist; //the grayscale histogram

        float range[] = {0, 256};
        const float *histRange = {range};
        bool uniform = true;
        bool accumulate = false;
        cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        // locate points that cuts at required value
        float max = accumulator.back();
        clipHistPercent *= (max / 100.0); //make percent as absolute
        clipHistPercent /= 2.0;           // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent)
            minGray++;

        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent))
            maxGray--;
    }

    // current range
    float inputRange = maxGray - minGray;

    alpha = (histSize - 1) / inputRange; // alpha expands current range to histsize range
    beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

    // Apply brightness and contrast normalization
    // convertTo operates with saurate_cast
    src.convertTo(dst, -1, alpha, beta);

    // restore alpha channel from source
    if (dst.type() == CV_8UC4)
    {
        int from_to[] = {3, 3};
        cv::mixChannels(&src, 4, &dst, 1, from_to, 1);
    }
    return;
}

void MyGammaCorrection(cv::Mat &src, cv::Mat &dst, float fGamma)
{
    CV_Assert(src.data);

    // accept only char type matrices
    CV_Assert(src.depth() != sizeof(uchar));

    // build look up table
    unsigned char lut[256];
    for (int i = 0; i < 256; i++)
    {
        lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
    }

    dst = src.clone();
    const int channels = dst.channels();
    switch (channels)
    {
    case 1:
    {

        cv::MatIterator_<uchar> it, end;
        for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
            //*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
            *it = lut[(*it)];

        break;
    }
    case 3:
    {

        cv::MatIterator_<cv::Vec3b> it, end;
        for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)
        {
            //(*it)[0] = pow((float)(((*it)[0])/255.0), fGamma) * 255.0;
            //(*it)[1] = pow((float)(((*it)[1])/255.0), fGamma) * 255.0;
            //(*it)[2] = pow((float)(((*it)[2])/255.0), fGamma) * 255.0;
            (*it)[0] = lut[((*it)[0])];
            (*it)[1] = lut[((*it)[1])];
            (*it)[2] = lut[((*it)[2])];
        }

        break;
    }
    }
}

//https://blog.csdn.net/hjimce/article/details/45421299 导向滤波
// float Guidedfiler(float *inimg, float *guidedimg, int height, int widht, int Radius, float eps)
// {
//     int lenght = height * widht;
//     float *mult = new float[lenght];
//     float *oned = new float[lenght];
//     for (int i = 0; i < lenght; i++)
//     {
//         mult[i] = inimg[i] * guidedimg[i];
//         oned[i] = 1;
//     }
//     float *covmult = new float[lenght];
//     float *covone = new float[lenght];
//     FastGetAVG(covmult, mult, widht, height, Radius);
//     FastGetAVG(covone, oned, widht, height, Radius);
//     for (int i = 0; i < lenght; i++)
//     {
//         covmult[i] /= covone[i];
//     }
//     delete[] mult;
//     delete[] oned;

//     //计算导向图、原图的窗口均值
//     float *mean_inimg = new float[lenght];
//     FastGetAVG(mean_inimg, inimg, widht, height, Radius);
//     float *mean_guideimg = new float[lenght];
//     FastGetAVG(mean_guideimg, guidedimg, widht, height, Radius);
//     for (int i = 0; i < lenght; i++)
//     {
//         mean_guideimg[i] /= covone[i];
//         mean_inimg[i] /= covone[i];
//     }

//     //计算ak的除数
//     float *var_guideimg = new float[lenght];
//     float *sqr_guideimg = new float[lenght];
//     for (int i = 0; i < lenght; i++)
//     {
//         sqr_guideimg[i] = guidedimg[i] * guidedimg[i];
//     }
//     FastGetAVG(var_guideimg, sqr_guideimg, widht, height, Radius);
//     delete[] sqr_guideimg;
//     for (int i = 0; i < lenght; i++)
//     {
//         var_guideimg[i] = var_guideimg[i] / covone[i] - mean_guideimg[i] * mean_guideimg[i];
//     }
//     //计算ak
//     float *a = new float[lenght];
//     for (int i = 0; i < lenght; i++)
//     {
//         a[i] = (covmult[i] - mean_guideimg[i] * mean_inimg[i]) / (var_guideimg[i] + eps);
//     }
//     //计算bk
//     float *b = new float[lenght];
//     for (int i = 0; i < lenght; i++)
//     {
//         b[i] = mean_inimg[i] - a[i] * mean_guideimg[i];
//     }
//     delete[] covmult;
//     delete[] mean_guideimg;
//     delete[] mean_inimg;
//     delete[] var_guideimg;
//     float *mean_a = new float[lenght];
//     float *mean_b = new float[lenght];
//     FastGetAVG(mean_a, a, widht, height, Radius);
//     FastGetAVG(mean_b, b, widht, height, Radius);
//     for (int i = 0; i < lenght; i++)
//     {
//         mean_a[i] /= covone[i];
//         mean_b[i] /= covone[i];
//     }
//     delete[] a;
//     delete[] b;
//     //输出图像
//     float *outimg = new float[lenght];
//     for (int i = 0; i < lenght; i++)
//     {
//         outimg[i] = mean_a[i] * guidedimg[i] + mean_b[i];
//     }
//     delete[] mean_a;
//     delete[] mean_b;
//     return outimg;
// }



void ACEcore(const Mat& src, Mat& dst,int C = 3, int n = 3, float MaxCG = 7.5)
{
    int row = src.rows;
    int col = src.cols;
    Mat meanLocal; //图像局部均值
    Mat varLocal; //图像局部方差
    Mat meanGlobal; //全局均值
    Mat varGlobal; //全局标准差
    blur(src.clone(), meanLocal, Size(n, n));
    Mat highFreq = src - meanLocal;
    varLocal = highFreq.mul(highFreq);
    varLocal.convertTo(varLocal, CV_32F);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            varLocal.at<float>(i, j) = (float)sqrt(varLocal.at<float>(i, j));
        }
    }
    meanStdDev(src, meanGlobal, varGlobal);
    Mat gainArr = meanGlobal / varLocal; //增益系数矩阵
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(gainArr.at<float>(i, j) > MaxCG){
                gainArr.at<float>(i, j) = MaxCG;
            }
        }
    }
    printf("%d %d\n", row, col);
    gainArr.convertTo(gainArr, CV_8U);
    gainArr = gainArr.mul(highFreq);
    Mat dst1 = meanLocal + gainArr;
    Mat dst2 = meanLocal + C * highFreq;
    
    dst = dst1;
}


void ACE(const cv::Mat& src, cv::Mat& dst,int C, int n, float MaxCG)
{
    std::vector <cv::Mat> now;
    split(src, now);
    // C = 150;
    // n = 5;
    // MaxCG = 3;
    cv::Mat dst1, dst2, dst3;
    ACEcore(now[0], dst1, C, n, MaxCG);
    ACEcore(now[1], dst2, C, n, MaxCG);
    ACEcore(now[2], dst3, C, n, MaxCG);
    now.clear();

    now.push_back(dst1);
    now.push_back(dst2);
    now.push_back(dst3);
    cv::merge(now, dst);
}




} // namespace enhance