#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;


//------图像通道分离函数 split----------
void ImgSplit(Mat& img){
    vector<Mat> mv;
    split(img,mv);
    imshow("blue channel",mv[0]);
    imshow("green channel",mv[1]);
    imshow("red channel",mv[2]);


}


int main(){
    Mat img = imread("../pictures/3.png");
    ImgSplit(img);
    waitKey(0);
    return 0;
}