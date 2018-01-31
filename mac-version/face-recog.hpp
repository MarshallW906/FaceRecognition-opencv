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

   protected:
    void LoadTrainingImg();
    std::vector<cv::Mat> face_training_images_;
};