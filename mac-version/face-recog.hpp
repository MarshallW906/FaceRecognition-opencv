#include "./stdc++.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

class FaceRecognitionPCA {
   public:
    FaceRecognitionPCA() = default;
    ~FaceRecognitionPCA() = default;
    void init();
    void SetK(int k) { k_ = k; }
    int GetK() { return k_; }
    void SetDebugEnable(bool d) { DEBUG_ = d; }
    void SetKandStartTest(int);

   protected:
    void LoadTrainingImg();
    void LoadTestImg();
    void CalcReuseableParams();
    void CalcParams();
    void TestRecognition();
    bool DEBUG_;

   private:
    // face_training_images_ : stored 40*7 training images as row-vectors
    // get one image vector: face_training_images_.row(i).t();
    // also known as X.t() matrix (because it's stored in row-vectors)
    cv::Mat Xt_face_training_images_;
    // x_avg: x average
    cv::Mat x_avg_;
    // x_avg_i_: deduct the mean from each point
    cv::Mat x_avg_i_;
    // L_: equivalent covariance matrix of C
    cv::Mat L_;
    // eigens of L_
    cv::Mat W_L_eigenVectors_;
    cv::Mat Tao_L_eigenValues_;
    // V_k_: k eigenVectors corresponding to the largest k eigenValues
    cv::Mat V_kt_;
    // alpha_ik_t_: alpha_ik's transpose
    cv::Mat alpha_ik_t_;

    int k_, N_, d_;

    // Test Images
    int P_;
    cv::Mat test_images_;
    cv::Mat zp_avg_test_images_;
};