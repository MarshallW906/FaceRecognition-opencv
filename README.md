# FaceRecognition-opencv : Mac-Version

This is a simple demo, using raw PCA Algorithm.

Thanks to [Cambrige ORL face database](http://www.cl.cam.ac.uk/Research/DTG/attarchive:pub/data/att_faces.zip).

## Compile & Run

You can use `Makefile` to automatically compile it.

```bash
git clone https://github.com/MarshallW906/FaceRecognization-opencv.git
git checkout mac-version
make
make run
```

Or you can compile it manually, it's still simple.

```bash
g++ main.cpp -std=c++14 `pkg-config opencv --cflags --libs`
```

## Test Result

Use every first 7 images as training images, and the rest 3 as test images. (1~7 training, 8~10 test)

Here is the test result:

```bash
One of Best K: [51], Max Correct Rate: [0.966667].

All k tests:
Correct Count: [116], Correct Rate: [0.966667]: 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
Correct Count: [115], Correct Rate: [0.958333]: 50, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
```