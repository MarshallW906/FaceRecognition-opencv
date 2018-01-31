#include "face-recog.cpp"
#include "stdc++.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

#ifndef DEBUG
#define DEBUG true
#endif

void test() {
    cv::Mat ttt;
    ttt = cv::imread("./att_faces/s1/1.pgm", cv::IMREAD_GRAYSCALE);
    if (ttt.empty()) {
        cout << "Could not open or find the image" << std::endl;
        return;
    }
    cv::namedWindow("Display", WINDOW_AUTOSIZE);
    cv::imshow("Display", ttt);
    waitKey(0);
}

int main() {
    FaceRecognitionPCA fr;
    fr.init();
}