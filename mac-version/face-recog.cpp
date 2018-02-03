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
    cout << "[INFO] PCA init..." << endl << endl;
    N_ = 280;
    d_ = 10304;
    k_ = 75;
    LoadTrainingImg();
    CalcParams();
    cout << "[COMPLETED] PCA init completed." << endl << endl;
}

void FaceRecognitionPCA::LoadTrainingImg() {
    cout << "[INFO] Load Training Images..." << endl;
    cv::Mat trainingFaces;
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
                cout << "[ERROR] Load " << filename << "failed" << endl;
            } else {
                success_count++;
            }
            tmpMat = tmpMat.reshape(1, 1);
            trainingFaces.push_back(std::move(tmpMat));
        }
    }
    if (success_count == 280) {
        cout << "[INFO] Load training images successfully." << endl;
        if (DEBUG) {
            cout << "[DEBUG] TrainingFaces Mat size: " << trainingFaces.size()
                 << endl;
        }
    } else {
        cout << "[ERROR] Not all training images are successfully loaded."
             << endl;
        return;
    }
    trainingFaces.convertTo(trainingFaces, CV_64FC1, 1.0 / 255);
    Xt_face_training_images_ = std::move(trainingFaces);
    cout << "[COMPLETED] Load Training Images Completed." << endl << endl;
}

void FaceRecognitionPCA::CalcParams() {
    cout << "[INFO] Calc Necessary Params..." << endl;
    // x_avg_
    Mat tmp_x_avg;
    cv::reduce(Xt_face_training_images_, tmp_x_avg, 0,
               cv::ReduceTypes::REDUCE_AVG);
    if (DEBUG) {
        cout << "[DEBUG] x_avg_ size(width * height): " << tmp_x_avg.size()
             << endl;
        // cv::imshow("average face", tmp_x_avg.reshape(0, 112));
        // waitKey(0);
    }
    x_avg_ = std::move(tmp_x_avg);

    // x_avg_i_
    Mat tmp_x_avg_i;
    for (int i = 0; i < N_; i++) {
        Mat tmprow = Xt_face_training_images_.row(0) - x_avg_;
        tmp_x_avg_i.push_back(tmprow);
    }
    if (DEBUG) {
        // cv::imshow("x_avg_i", tmp_x_avg_i);
        // waitKey(0);
    }
    x_avg_i_ = std::move(tmp_x_avg_i);

    // L_: the equivalent covariance matrix of C
    Mat tmp_L = Xt_face_training_images_ * Xt_face_training_images_.t();
    if (DEBUG) {
        cout << "[DEBUG] L_ matrix size: " << tmp_L.size() << endl;
    }
    L_ = std::move(tmp_L);

    // L_'s eigenVectors & eigenValues
    cv::eigen(L_, Tao_L_eigenValues_, W_L_eigenVectors_);
    if (DEBUG) {
        cout << "[DEBUG] L_ eigenVectors size: " << W_L_eigenVectors_.size()
             << endl;
        cout << "[DEBUG] L_ eigenValues size: " << Tao_L_eigenValues_.size()
             << endl;
    }

    // V_kt_;
    Mat tmp_Vt = (Xt_face_training_images_.t() * W_L_eigenVectors_).t();
    // cut & normalize
    Mat tmp_Vkt_cutk = tmp_Vt.rowRange(0, k_);
    Mat tmp_V_kt;
    // Mat towrite, towrite_normalize;
    for (int i = 0; i < k_; i++) {
        Mat tmprow;
        cv::normalize(tmp_Vkt_cutk.row(i), tmprow);
        // towrite_normalize.push_back(tmp_Vkt_cutk.row(i).reshape(0, 112));
        tmp_V_kt.push_back(std::move(tmprow));
        // towrite.push_back(tmp_Vkt_cutk.row(i).reshape(0, 112));
    }
    // imwrite("./eigenfaces.pgm", (towrite * 255.0));
    // imwrite("./eigenfaces_normalized.png", (towrite_normalize * 255.0));

    if (DEBUG) {
        cout << "[DEBUG] tmp_Vt size: " << tmp_Vt.size() << endl;
        cout << "[DEBUG] tmp_V_kt size: " << tmp_V_kt.size() << endl;
    }

    V_kt_ = std::move(tmp_V_kt);

    // alpha_ik_
    Mat tmp_alpha_ik = V_kt_ * x_avg_i_.t();
    cout << tmp_alpha_ik.size() << endl;
    alpha_ik_ = std::move(tmp_alpha_ik);

    cout << "[COMPLETED] Calc Necessary Params Completed." << endl << endl;
}