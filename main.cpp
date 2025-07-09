#include "inference.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace cv;

float focalLength = -1.0f;
const float REAL_HEIGHT_CM = 23.0f;
bool calibrated = false;

int main(int argc, char **argv)
{
    bool runOnGPU = true;
    std::chrono::time_point<std::chrono::system_clock> start_time;
    bool flag_start = false;

    Inference inf("/home/bima/Monocular Distance Estimation/best.onnx", Size(640, 640), "/home/bima/Monocular Distance Estimation/classes.txt", runOnGPU);

    VideoCapture cap(0, CAP_V4L2);
    cap.set(CAP_PROP_FRAME_WIDTH, 800);
    cap.set(CAP_PROP_FRAME_HEIGHT, 600);
    cap.set(CAP_PROP_FOURCC , VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(CAP_PROP_FPS, 60);
    
    if (!cap.isOpened()) {
        std::cout << "Tidak bisa membuka kamera.\n";
        return -1;
    }

    std::ofstream logFile("/home/bima/Monocular Distance Estimation/distance_log.csv");
    if (logFile.is_open()) {
        logFile << "timestamp_ms,pixel_height,estimated_distance_cm,confidence\n";
    } else {
        cout << "[ERROR] Gagal membuka file log.\n";
    }

    std::ifstream focalFileIn("/home/bima/Monocular Distance Estimation/focal.txt");
    if (focalFileIn.is_open()) {
        focalFileIn >> focalLength;
        focalFileIn.close();
        if (focalLength > 0) {
            calibrated = true;
            cout << "[INFO] focalLength dibaca dari file: " << focalLength << " px" << endl;
        }
    } else {
        cout << "[INFO] Tidak ditemukan file 'focal.txt'. Silakan kalibrasi (tekan 'c')." << endl;
    }

    double frame_counter = 0;
    double second_passed = 0;
    double fps = 20;

    cout << "[INFO] Tekan 'c' untuk kalibrasi saat objek ada di jarak tertentu.\n";
    cout << "[INFO] Tekan ESC untuk keluar.\n";

    while (true) {
        if (!flag_start) {
            start_time = std::chrono::system_clock::now();
            flag_start = true;
        }
        
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Frame kosong, keluar.\n";
            break;
        }

        frame_counter++;

        std::vector<Detection> output = inf.runInference(frame);

        int detected_obj = output.size();
        float max_conf = 0.0;
        int index_max = -1;

        for (int i = 0; i < detected_obj; ++i) {
            Detection detection = output[i];
            if (detection.confidence >= max_conf) {
                max_conf = detection.confidence;
                index_max = i;
            }
        }

        if (index_max != -1) {
            Detection detection = output[index_max];
            Rect box = detection.box;

            rectangle(frame, box, Scalar(0, 255, 0), 2);

            if (calibrated) {
                float pixelHeight = box.height;
                float estimated_distance = (REAL_HEIGHT_CM * focalLength) / pixelHeight;

                if (logFile.is_open()) {
                    auto now = chrono::system_clock::now();
                    auto ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();
                
                    logFile << ms << "," << pixelHeight << "," << estimated_distance << "," << max_conf << "\n";
                }                

                string label = "Jarak: " + to_string(int(estimated_distance)) + " cm";
                putText(frame, label, Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
            } else {
                putText(frame, "Tekan 'c' untuk kalibrasi", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            }
        }

        std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        second_passed = duration / 1000;
        if (second_passed >= 1.0) {
            fps = frame_counter / second_passed;
            frame_counter = 0;
            start_time = end_time;
        }

        putText(frame, "FPS: " + to_string(int(fps)), Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
        imshow("Estimasi Jarak YOLO", frame);

        char key = (char)waitKey(1);
        if (key == 27) break; // ESC
        else if (key == 'c') {
            if (index_max != -1) {
                float pixelHeight = output[index_max].box.height;
                float jarakasli;
                cout << "\n[INPUT] Masukkan jarak sebenarnya dari objek ke kamera (cm): ";
                cin >> jarakasli;
                focalLength = (pixelHeight * jarakasli) / REAL_HEIGHT_CM;
                calibrated = true;
                cout << "[OK] Focal Length dikalibrasi: " << focalLength << " px\n";

                std::ofstream focalFileOut("/home/bima/Monocular Distance Estimation/focal.txt");
                if (focalFileOut.is_open()) {
                    focalFileOut << focalLength;
                    focalFileOut.close();
                    cout << "[INFO] focalLength disimpan ke 'focal.txt'\n";
                } else {
                    cout << "[ERROR] Gagal menyimpan focalLength ke file.\n";
                }
            } else {
                cout << "[WARN] Tidak ada objek terdeteksi untuk kalibrasi.\n";
            }
        }
    }

    if (logFile.is_open()) {
        logFile.close();
        cout << "[INFO] File log disimpan ke distance_log.csv\n";
    }
    
    cap.release();
    destroyAllWindows();
    return 0;
}
