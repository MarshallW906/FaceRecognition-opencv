#include "./face-recog.hpp"
#include "./stdc++.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

#ifndef DEBUG
#define DEBUG true
#endif

void FaceRecognitionPCA::init() {
    cout << "[INFO] PCA init" << endl;
    face_training_images_.clear();
    LoadTrainingImg();
    cout << "[INFO] PCA init completed." << endl;
}

void FaceRecognitionPCA::LoadTrainingImg() {
    cout << "[INFO] Start Load Training Images" << endl;
    vector<cv::Mat> trainingFaces;
    int success_count = 0;
    // 1~7 as training images, 8~9 as test images
    string filePrefix = "./att_faces/s";
    for (int i = 1; i <= 40; i++) {
        string outer_number = to_string(i);
        string tmp = filePrefix + outer_number + '/';
        for (int j = 1; j <= 7; j++) {
            char number = j + '0';
            string filename = tmp + number + ".pgm";
            Mat tmpMat = cv::imread(filename, IMREAD_GRAYSCALE);
            if (tmpMat.empty()) {
                // Check if the image is loaded successfully
                if (DEBUG) cout << "Load " << filename << "failed" << endl;
            } else {
                success_count++;
            }
            tmpMat.reshape(0, 1).t();
            trainingFaces.push_back(std::move(tmpMat));
        }
    }
    if (success_count == 280) {
        cout << "[INFO] Load training images successfully." << endl;
    } else {
        cout << "[ERROR] Not all training images are successfully loaded."
             << endl;
        return;
    }
    face_training_images_ = std::move(trainingFaces);
}