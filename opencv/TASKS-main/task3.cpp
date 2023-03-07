#include <opencv2/opencv.hpp>

using namespace cv;


int main(){
    Mat srcImg = imread("../pictures/1.png");
    Mat deepMat,shallowMat;
    //-------------shallow copy---------------
    //浅拷贝会随愿图变化而变化
    shallowMat = srcImg;   
    
    //-------------deep copy------------------
    //深拷贝另辟空间储存图像 不会随原图变化而变化
    // srcImg.copyTo(deepMat);      方法一
    deepMat = srcImg.clone();     //方法二
    //改变原图srcImg
    int rowNumber = srcImg.rows;
    int colNumebr = srcImg.cols;
    for(int i = 0;i < rowNumber; i++){
        for(int j = 0; j < colNumebr; j++){
            //访问24位彩色图像Vec3b     灰度图像是uchar
            srcImg.at<Vec3b>(i,j)[0] = (srcImg.at<Vec3b>(i,j)[0]+srcImg.at<Vec3b>(i,j)[1]+srcImg.at<Vec3b>(i,j)[2])/3;//bule通道
            srcImg.at<Vec3b>(i,j)[1] = (srcImg.at<Vec3b>(i,j)[0]+srcImg.at<Vec3b>(i,j)[1]+srcImg.at<Vec3b>(i,j)[2])/3;//green通道
            srcImg.at<Vec3b>(i,j)[2] = (srcImg.at<Vec3b>(i,j)[0]+srcImg.at<Vec3b>(i,j)[1]+srcImg.at<Vec3b>(i,j)[2])/3;//red通道
        }
    }

    //---------------------------------------------
    imshow("shallowCopy",shallowMat);
    imshow("deepCopy",deepMat);
    waitKey(0);

}