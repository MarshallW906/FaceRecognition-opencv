#include "face-recog.cpp"
#include "stdc++.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

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
    fr.SetDebugEnable(false);
    int best_k = 50;
    int max_correct_count = 0;
    std::map<int, std::vector<int>> k_correct_rate;
    for (int k = 95; k <= 100; k++) {
        cout << "Current k: " << k << endl;
        int cur_correct_count = fr.SetKandStartTest(k);
        // for statistic
        k_correct_rate[cur_correct_count].push_back(k);
        if (cur_correct_count > max_correct_count) {
            max_correct_count = cur_correct_count;
            best_k = k;
        }
        cout << "Correct count: " << cur_correct_count << endl;
    }
    double max_correct_rate = (double)max_correct_count / 120;
    cout << endl;
    cout << "One of Best K: [" << best_k << "], Max Correct Rate: ["
         << max_correct_rate << "]." << endl;
    cout << endl;
    cout << "All k tests:" << endl;
    for (int cnt = max_correct_count; cnt >= 0; cnt--) {
        if (!k_correct_rate[cnt].empty()) {
            double cur_rate = (double)cnt / 120;
            cout << "Correct Count: [" << cnt << "], Correct Rate: ["
                 << cur_rate << "]: ";
            for (auto &tmpk : k_correct_rate[cnt]) cout << tmpk << ", ";
            cout << endl;
        }
    }
}