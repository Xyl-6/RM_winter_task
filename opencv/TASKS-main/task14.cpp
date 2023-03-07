#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/photo.hpp>
#include <iostream>
using namespace std;
using namespace cv;
#define g_d 20.0


void test01(){
    Mat src = imread("../pictures/14-2.jpg");
    namedWindow("orgimg",0);
    imshow("orgimg",src);
    //------------------------------
    Mat dst1 = src.clone();
    Mat output;
    blur(dst1,dst1,Size(5,5));
    bilateralFilter(dst1,output,g_d,g_d*2,g_d/2);
    namedWindow("dst1",0);
    imshow("dst1",output);
    //-----------------------------------

}


void test02(){
    Mat src = imread("../pictures/14-1.jpg");
    // imshow("orgimg",src);
    Mat srccpy = src.clone();
    //
    Mat srcROI = srccpy(Rect2d(120,170,220,340));
    Mat srcROIcpy = srcROI.clone();
    medianBlur(srcROI,srcROI,7);
    
    imshow("dst",srccpy);
}

void test03(){
    Mat src = imread("../pictures/14-1.jpg");
    imshow("org",src);

    blur(src,src,Size(5,5));
    Mat dst;
    bilateralFilter(src,dst,g_d,g_d*2,g_d/2);
    imshow("dst",dst);
}


int main(){
    test01();
    test03();
    waitKey(0);
    return 0;
}