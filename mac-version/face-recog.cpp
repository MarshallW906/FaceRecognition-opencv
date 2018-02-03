#include "./face-recog.hpp"
#include "./stdc++.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

void FaceRecognitionPCA::init() {
    cout << "[INFO] PCA init..." << endl << endl;
    N_ = 280;
    d_ = 10304;
    P_ = 120;
    DEBUG_ = false;
    cout << "Load Training Images..." << endl;
    LoadTrainingImg();
    cout << "Load Test Images..." << endl;
    LoadTestImg();
    cout << "Calc Reuseable Params..." << endl;
    CalcReuseableParams();
    cout << "[COMPLETED] PCA init completed." << endl << endl;
}

int FaceRecognitionPCA::SetKandStartTest(int k) {
    SetK(k);
    CalcParams();
    return TestRecognition();
}

void FaceRecognitionPCA::LoadTrainingImg() {
    if (DEBUG_) cout << "[INFO] Load Training Images..." << endl;
    cv::Mat trainingFaces;
    int success_count = 0;
    // 1~7 as training images, 8~10 as test images
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
    if (success_count == N_) {
        if (DEBUG_) {
            cout << "[INFO] Load Training Images successfully." << endl;
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
    if (DEBUG_)
        cout << "[COMPLETED] Load Training Images Completed." << endl << endl;
}

void FaceRecognitionPCA::LoadTestImg() {
    if (DEBUG_) cout << "[INFO] Load Test Images..." << endl;
    cv::Mat testFaces;
    int success_count = 0;
    // 1~7 as training images, 8~10 as test images
    string filePrefix = "./att_faces/s";
    for (int i = 1; i <= 40; i++) {
        string outer_number = to_string(i);
        string tmp = filePrefix + outer_number + '/';
        for (int j = 8; j <= 10; j++) {
            ostringstream ostr;
            ostr << tmp << j << ".pgm";
            string filename = ostr.str();
            Mat tmpMat = cv::imread(filename, IMREAD_GRAYSCALE);
            if (tmpMat.empty()) {
                // Check if the image is loaded successfully
                cout << "[ERROR] Load " << filename << "failed" << endl;
            } else {
                success_count++;
            }
            tmpMat = tmpMat.reshape(1, 1);
            testFaces.push_back(std::move(tmpMat));
        }
    }
    if (success_count == P_) {
        if (DEBUG_) {
            cout << "[INFO] Load Test Images successfully." << endl;
            cout << "[DEBUG] TestFaces Mat size: " << testFaces.size() << endl;
        }
    } else {
        cout << "[ERROR] Not all test images are successfully loaded." << endl;
        return;
    }
    testFaces.convertTo(testFaces, CV_64FC1, 1.0 / 255);
    test_images_ = std::move(testFaces);
    if (DEBUG_)
        cout << "[COMPLETED] Load Test Images Completed." << endl << endl;
}

void FaceRecognitionPCA::CalcReuseableParams() {
    if (DEBUG_) cout << "[INFO] Calc Reuseable Params..." << endl;
    // x_avg_
    if (DEBUG_) cout << "[INFO] Calc x_avg_..." << endl;
    Mat tmp_x_avg;
    cv::reduce(Xt_face_training_images_, tmp_x_avg, 0,
               cv::ReduceTypes::REDUCE_AVG);
    if (DEBUG_) {
        cout << "[DEBUG] x_avg_ size(width * height): " << tmp_x_avg.size()
             << endl;
        // cv::imshow("average face", tmp_x_avg.reshape(0, 112));
        // waitKey(0);
    }
    x_avg_ = std::move(tmp_x_avg);
    if (DEBUG_) cout << "[COMPLETED] Calc x_avg Completed." << endl;

    // x_avg_i_
    if (DEBUG_) cout << "[INFO] Calc x_avg_i_..." << endl;
    Mat tmp_x_avg_i;
    for (int i = 0; i < N_; i++) {
        Mat tmprow = Xt_face_training_images_.row(i) - x_avg_;
        tmp_x_avg_i.push_back(tmprow);
    }
    if (DEBUG_) {
        cout << "[DEBUG] x_avg_i_ size(width * height): " << tmp_x_avg_i.size()
             << endl;
    }
    x_avg_i_ = std::move(tmp_x_avg_i);
    if (DEBUG_) cout << "[COMPLETED] Calc x_avg_i_ Completed." << endl;

    // zp_avg
    if (DEBUG_) cout << "[INFO] Calc zp_avg..." << endl;
    for (int i = 0; i < P_; i++) {
        Mat tmp = test_images_.row(i) - x_avg_;
        zp_avg_test_images_.push_back(std::move(tmp));
    }
    if (DEBUG_) {
        cout << "[DEBUG] zp_avg_test_images_.size(): "
             << zp_avg_test_images_.size() << endl;
    }
    if (DEBUG_) cout << "[COMPLETED] Calc zp_avg Completed." << endl;

    // L_: the equivalent covariance matrix of C
    if (DEBUG_) cout << "[INFO] Calc L_ matrix..." << endl;
    // Mat tmp_L = Xt_face_training_images_ * Xt_face_training_images_.t();
    Mat tmp_L = x_avg_i_ * x_avg_i_.t();
    if (DEBUG_) {
        cout << "[DEBUG] L_ matrix size: " << tmp_L.size() << endl;
    }
    L_ = std::move(tmp_L);
    if (DEBUG_) cout << "[COMPLETED] Calc L_ matrix Completed." << endl;

    // L_'s eigenVectors & eigenValues
    if (DEBUG_) cout << "[INFO] Calc L_ eigenValues & eigenVectors..." << endl;
    cv::eigen(L_, Tao_L_eigenValues_, W_L_eigenVectors_);
    if (DEBUG_) {
        cout << "[DEBUG] L_ eigenVectors size: " << W_L_eigenVectors_.size()
             << endl;
        cout << "[DEBUG] L_ eigenValues size: " << Tao_L_eigenValues_.size()
             << endl;
    }
    if (DEBUG_)
        cout << "[COMPLETED] Calc L_'s eigenValues & eigenVectors Completed."
             << endl;

    if (DEBUG_)
        cout << "[INFO] Calc Reuseable Params Completed." << endl << endl;
}

void FaceRecognitionPCA::CalcParams() {
    if (DEBUG_) cout << "[INFO] Calc Necessary Params..." << endl;

    // V_kt_;
    if (DEBUG_) cout << "[INFO] Calc V_kt_..." << endl;
    Mat tmp_Vt = (x_avg_i_.t() * W_L_eigenVectors_.t()).t();
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

    if (DEBUG_) {
        cout << "[DEBUG] tmp_Vt size: " << tmp_Vt.size() << endl;
        cout << "[DEBUG] tmp_V_kt size: " << tmp_V_kt.size() << endl;
    }

    V_kt_ = std::move(tmp_V_kt);
    if (DEBUG_) cout << "[COMPLETED] Calc V_kt_ Completed." << endl;

    // alpha_ik_
    if (DEBUG_) cout << "[INFO] Calc alpha_ik_" << endl;
    Mat tmp_alpha_ikt = (V_kt_ * x_avg_i_.t()).t();
    // Mat tmp_alpha_ikt;
    // for (int i = 0; i < N_; i++) {
    //     Mat tmprow = (V_kt_ * x_avg_i_.row(i).t()).t();
    //     tmp_alpha_ikt.push_back(std::move(tmprow));
    // }
    if (DEBUG_)
        cout << "[DEBUG] tmp_alpha_ik size: " << tmp_alpha_ikt.size() << endl;
    alpha_ik_t_ = std::move(tmp_alpha_ikt);
    if (DEBUG_) cout << "[COMPLETED] Calc alpha_ik_ Completed." << endl;

    if (DEBUG_)
        cout << "[COMPLETED] Calc Necessary Params Completed." << endl << endl;
}

int FaceRecognitionPCA::TestRecognition() {
    if (DEBUG_) cout << "[INFO] Start TestRecognition with k = " << k_ << endl;
    // alpha_p
    if (DEBUG_) cout << "[INFO] Calc alpha_p..." << endl;
    cv::Mat alpha_p;
    for (int i = 0; i < P_; i++) {
        Mat tmp = (V_kt_ * zp_avg_test_images_.row(i).t()).t();
        alpha_p.push_back(std::move(tmp));
    }
    if (DEBUG_) cout << "[DEBUG] alpha_p size: " << alpha_p.size() << endl;
    if (DEBUG_) cout << "[COMPLETED] Calc alpha_p Completed." << endl;

    // comparison: Face Recognition
    if (DEBUG_)
        cout << "[INFO] Start Comparing alpha_p with alpha_ik..." << endl;
    int correct_count = 0;
    for (int p = 0; p < P_; p++) {
        int min_jp_idx = 0;
        double min_jp = 999999;
        const Mat& tmp_alpha_p = alpha_p.row(p);
        // find match
        for (int l = 0; l < N_; l++) {
            const Mat& tmp_alpha_l = alpha_ik_t_.row(l);
            Mat tmp_distance_vec = tmp_alpha_p - tmp_alpha_l;
            auto tmp_distance = cv::norm(tmp_distance_vec);
            // cout << tmp_distance << endl;
            if (tmp_distance < min_jp) {
                min_jp = tmp_distance;
                min_jp_idx = l;
            }
        }
        // check match
        // cout << "p = " << p << ", match_idx = " << min_jp_idx << endl;
        if ((p / 3) == (min_jp_idx / 7)) correct_count++;
    }
    double correct_rate = (correct_count + 1.0 - 1.0) / P_;
    if (DEBUG_) {
        cout << "[RESULT] k = [" << k_ << "], correct_count: [" << correct_count
             << "], Correct rate: [" << correct_rate << "]." << endl;
    }
    return correct_count;
}