compile:
	g++ main.cpp -o face_recog.out -std=c++14 `pkg-config opencv --cflags --libs`

run: compile
	./face_recog.out

clean:
	rm face_recog.out
