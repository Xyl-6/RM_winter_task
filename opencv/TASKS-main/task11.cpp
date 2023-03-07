#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;



//图像连通域计算
//steps：1.二值化  2.转灰度  3.计算
void test01(){
    Mat src , src_gray;
    src = imread("../pictures/11.png");
    //--------去噪、二值化、化为灰度图像-----------------
    blur(src,src,Size(10,10));
    threshold(src,src,80,255,THRESH_BINARY_INV);
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
    Mat lables,status,centroids;
    int cnt = connectedComponentsWithStats(src_gray,lables,status,centroids,8);
    //框出每个回型针
    for(int i=0 ; i<cnt ;i++){
        int x = status.at<int>(i,0);
        int y = status.at<int>(i,1);
        int width = status.at<int>(i,2);
        int height = status.at<int>(i,3);
        rectangle(src,Rect(x,y,width,height),Scalar(0,0,255),1,8,0);
    }
    cout << "回型针的个数为： " << cnt-1 << endl;
    //输出每个中心点的坐标看看那
    // for(int i=1 ; i<cnt ;i++){
    //     double x = centroids.at<double>(i,0); 
    //     double y = centroids.at<double>(i,1);
    //     printf("x = %.2f  y = %.2f\n",x,y);
    // }
    imshow("srcwithstatus",src);
    waitKey(0);
}




int main(){
    test01();
    return 0;
}