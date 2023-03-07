#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

//实现canny算子 并注意其原理   了解对canny算子进一步改进的方法


//canny边缘检测
void test01(){
    Mat img = imread("../pictures/2.jpg");
    // Mat CannyImg = img.clone();   //用来存放输出后的图像且大小与原图相同
    Mat CannyImg = Mat::zeros(img.size(),img.type());  //效果同上
    //Canny(src,dst,threshold1阈值1,threshold2阈值2，int sobel算子的大小 默认=3)
    //高低阈值的比为 2：1 到 3：1 之间
    Canny(img,CannyImg,30,60,3);
    namedWindow("CannyImg",0);
    imshow("CannyImg",CannyImg);
}



//Canny算子caise
//1.转化成灰度 2.降噪 3.canny 4.将得到的边缘作为掩码拷贝到效果图上  得到彩色的边缘图
void test02(){
    //导入图片
    Mat src,dst,src_gray,edge;   //原图  输出图  原图的灰度图  边缘效果图
    src = imread("../pictures/2.jpg");
    dst = Mat::zeros(src.size(),src.type());
    src_gray.create(src.size(),src.type());     
    //1.转化成灰度
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
    //2.降噪
    blur(src_gray,edge,Size(3,3));
    //3.canny
    Canny(edge,edge,30,60,3);
    //4.
    src.copyTo(dst,edge);

    imshow("Canny边缘检测(彩色)",dst);


}

int main(){
    test01();
    test02();
    waitKey(0);
    return 0;
}