#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace std;

class MouseEnterBox {
private:
    VideoCapture cap;
    bool display;
    Rect box;
    Ptr<BackgroundSubtractor> fgbg;
    int frame_count;
    double fps;
    double mouse_enter_time;

public:
    MouseEnterBox(const string& video_path, int bx = 50, int by = 50, int bw = 600, int bh = 800, bool display_video = false)
        : cap(video_path), display(display_video), box(bx, by, bw, bh), frame_count(0), mouse_enter_time(-1) {
        if (!cap.isOpened()) {
            cerr << "Error: Cannot open video file." << endl;
            exit(EXIT_FAILURE);
        }
        fps = cap.get(CAP_PROP_FPS);
        fgbg = createBackgroundSubtractorMOG2(500, 50, false);
    }

    double get_time() {
        Mat frame, gray, fgmask, thresh;
        while (cap.isOpened()) {
            if (!cap.read(frame)) break;
            frame_count++;

            cvtColor(frame, gray, COLOR_BGR2GRAY);
            fgbg->apply(gray, fgmask);
            threshold(fgmask, thresh, 50, 255, THRESH_BINARY);

            vector<vector<Point>> contours;
            findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            for (const auto& cnt : contours) {
                if (contourArea(cnt) < 500) continue;
                Rect bounding_box = boundingRect(cnt);
                if ((box & bounding_box) == bounding_box) {
                    if (mouse_enter_time < 0) mouse_enter_time = frame_count / fps;
                    break;
                }
            }

            if (display && display_image(frame)) break;
        }
        cap.release();
        destroyAllWindows();
        return mouse_enter_time;
    }

    bool display_image(Mat& image) {
        rectangle(image, box, Scalar(0, 255, 0), 2);
        if (mouse_enter_time >= 0) {
            putText(image, to_string(mouse_enter_time) + " sec", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        resize(image, image, Size(900, 600));
        imshow("Frame", image);
        return (waitKey(1000 / fps) & 0xFF) == 27;
    }
};

int main(int argc, char* argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <video_path> [box_x box_y box_w box_h] [-V]" << endl;
        return EXIT_FAILURE;
    }

    string filepath = argv[1];
    int bx = 50, by = 50, bw = 600, bh = 800;
    bool display_video = false;

    if (argc > 5) {
        bx = stoi(argv[2]);
        by = stoi(argv[3]);
        bw = stoi(argv[4]);
        bh = stoi(argv[5]);
        if (argc > 6 && string(argv[6]) == "-V") display_video = true;
    }

    MouseEnterBox MEB(filepath, bx, by, bw, bh, display_video);
    double enter_time = MEB.get_time();

    if (enter_time >= 0)
        cout << "Mouse fully entered the box at " << enter_time << " seconds." << endl;
    else
        cout << "Mouse never fully entered the box." << endl;

    return 0;
}