#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;


//调整BGR为HSV
int main(){
    Mat img = imread("../pictures/8.png");
    Mat img_hsv ,img_hsv_red;
    //将 BGR 图片转化为 HSV 后分离通道得到的就是 H色度  S饱和度  V明度
    cvtColor(img,img_hsv,COLOR_BGR2HSV);
    imshow("HSV",img_hsv);
    //--------------将 HSV 分离----------------
    // vector<Mat>mv;
    // split(img_hsv,mv);
    // imshow("H",mv[0]);
    // imshow("S",mv[1]);
    // imshow("V",mv[2]);
    //---------------------------------------------
    //-----------用 inRange() 提取红色像素----------------
    inRange(img_hsv,Scalar(60,0,0),Scalar(80,255,255),img_hsv_red);
    imshow("HSV_RED",img_hsv_red);
    waitKey(0);
    return 0;
}