#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
//使用不同形状或大小的算子

//腐蚀、膨胀要配合 getStructingElement() 函数来使用  用于第三个 kernel 参数
//--------腐蚀erode-------------
//求局部最小值
void test01(Mat& img){
    Mat element = getStructuringElement( MORPH_RECT ,Size(15,15),Point(-1,-1));  //矩形核
    Mat dst = img.clone();
    erode(img,dst,element);
    imshow("erodeimg",dst);
}




//-------膨胀dilate--------------
//求局部最大值
void test02(Mat& img){
    Mat element = getStructuringElement( MORPH_CROSS ,Size(10,10),Point(-1,-1));  //交叉型核
    Mat dst = img.clone();
    dilate(img,dst,element);
    imshow("dilateimg",dst);   
}


// 核心函数 morphologyEx() 
//-----------开运算(先腐蚀，后膨胀的过程)-----------
void test03(Mat& img){
    Mat element = getStructuringElement(MORPH_ELLIPSE,Size(10,10),Point(-1,-1));  // 核型为椭圆
    Mat dst;
    img.copyTo(dst);
    morphologyEx(img,dst,MORPH_OPEN,element);
    imshow("OPEN",dst);
}

//-----------闭运算--------------
void test04(Mat& img){
    Mat element = getStructuringElement(MORPH_ELLIPSE,Size(10,10),Point(-1,-1));
    Mat dst;
    img.copyTo(dst);
    morphologyEx(img,dst,MORPH_CLOSE,element);
    imshow("CLOSE",dst);
}

int main(){
    Mat img = imread("../pictures/3.png");
    imshow("orgimg",img);
    test01(img);
    test02(img);
    test03(img);
    test04(img);
    waitKey(0);
    return  0;
}
