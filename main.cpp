#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream> // Tiene dos clases una para leer archivos ifstream y otra para escribir archivos que es ofstream
#include <sstream>
// Las librerias de C puro acaban en .h (cabeceras)
#include <dirent.h>
#include <cstdlib> 
#include <string>
#include <filesystem>
#include <list>
#include <map>
// Cuando se carga la cabecer opencv.hpp automáticamente se cargan las demás cabeceras
//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> // Contiene los elementos básicos como el objeto Mat (matriz que representa la imagen)
#include <opencv2/highgui/highgui.hpp> // Contiene los elementos para crear una interfaz gráfica básica
// OpenCV no está pensado para crear interfaces gráficas potentes. Se centra en la visión artificial y PDI. Si se desea crear una interfaz gráfica completa, se debe usar QT
#include <opencv2/imgcodecs/imgcodecs.hpp> // Contiene las funcionalidad para acceder a los códecs que permiten leer diferentes formatos de imagen (JPEG, JPEG-2000, PNG, TIFF, GIF, etc.)
// Librerías para acceder al video y para poder escribir vídeos en disco
#include <opencv2/video/video.hpp> 
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp> // Librería para realizar operaciones de PDI 
using namespace cv;
using namespace std;

Mat frameActual, frame, frameAnterior, resta,resultado;


int rratio = 3;
int minT = 50;
int maxT = rratio*minT;   

// Variable que almacena el valor del track bar (Threshold)
int mThreshold = 16;

// Variable que almacena el valor del track bar (Contrast Stretching)
int mContrast = 0;

// Variable que almacena el valor de K del (Contrast Stretching)
int kContrast = 0;
//Variable kernel
int dimensionK=5;
// kernel
Mat elementoCruz = getStructuringElement(MORPH_CROSS, Size(dimensionK, dimensionK), Point(-1, -1));
Mat elementoRecto = getStructuringElement(MORPH_RECT, Size(dimensionK, dimensionK), Point(-1, -1));
Mat elementoEplip = getStructuringElement(MORPH_ELLIPSE, Size(dimensionK, dimensionK), Point(-1, -1));

int logistic(int r){
    double rd = (double) r;
    double dK = ((double)kContrast)/100.0;
    double res = 1.0/(1.0+exp(-dK*(rd-((double)mContrast))));
    res*=255.0;
    return ((int) res);
}


Mat aplyDilate(Mat img){
    Mat salida=img.clone();
    if(dimensionK%2!=0){
        Mat salida=img.clone();        
        morphologyEx(salida, salida, MORPH_DILATE, elementoRecto); 
    }   
    return salida;
}

Mat aplyHerode(Mat img){
    Mat salida=img.clone();
    if(dimensionK%2!=0){
        Mat salida=img.clone();
        morphologyEx(salida, salida, MORPH_ERODE, elementoEplip);        
    }   
    return salida;
}
Mat aplyeApertura(Mat img){
    Mat salida=img.clone();
    if(dimensionK%2!=0){
        Mat salida=img.clone();
        morphologyEx(salida, salida, MORPH_OPEN, elementoEplip);        
    }   
    return salida;
}
Mat aplyCierre(Mat img){
    Mat salida=img.clone();
    if(dimensionK%2!=0){
        Mat salida=img.clone();
        morphologyEx(salida, salida, MORPH_CLOSE, elementoRecto);        
    }   
    return salida;
}

Mat aplyGradient(Mat img){
    Mat salida=img.clone();
    if(dimensionK%2!=0){
        Mat salida=img.clone();
        morphologyEx(salida, salida, MORPH_GRADIENT, elementoEplip);        
    }   
    return salida;
}

Mat aplyTopHat(Mat img){
    Mat salida=img.clone();
    if(dimensionK%2!=0){
        Mat salida=img.clone();
        morphologyEx(salida, salida, MORPH_TOPHAT, elementoEplip);        
    }   
    return salida;
}

Mat aplyBackHat(Mat img){
    Mat salida=img.clone();
    if(dimensionK%2!=0){
        Mat salida=img.clone();
        morphologyEx(salida, salida, MORPH_BLACKHAT, elementoEplip);        
    }   
    return salida;
}

Mat aplyFilterGaussian(Mat img){
    Mat salida=img.clone();
    if(dimensionK%2!=0){             
        GaussianBlur(salida,salida,Size(dimensionK,dimensionK),1.7,1.7);//Borra Ruido;
    }    
    return salida;
}
 
Mat aplyCanny(Mat img){    
    Mat salida=img.clone();        
    Canny(salida, salida, minT, maxT, 3);
    return salida;
}

void mostrarSinFondo(){    
    resultado=frame.clone();    
    for (int i = 0; i < resta.rows; ++i) {
        for (int j = 0; j < resta.cols; ++j) {
            if((int) resta.at<uchar>(i, j) != 0)                
                break;            
            resultado.at<Vec3b>(i, j) =Vec3b(0,0,0);            
        }

        for (int j = resta.cols-1; j >= 0; --j) {
            if((int) resta.at<uchar>(i, j) != 0)  
                break;
            resultado.at<Vec3b>(i, j) =Vec3b(0,0,0);            
        }

    }
    imshow("resultado", resultado);
}

Mat aplyContrast_Threshold(Mat img){
    Mat gris=img.clone();
    int pixel=0;
    for(int i=0;i<gris.rows;i++){
                for(int j=0;j<gris.cols;j++){
                    pixel = gris.at<uchar>(i,j);

                    // Pixeles para el Contrast Stretching
                    gris.at<uchar>(i,j) = logistic(pixel);

                    // Pixeles para el Threshold
                    if(pixel>mThreshold)
                        pixel = 255;
                    else
                        pixel = 0;
                    
                    gris.at<uchar>(i,j) = pixel;
                }
            }
    return gris;            
}

Mat aplySobel(Mat img){
    Mat gris;
    Sobel(img,gris, CV_32F, 1, 0);
    return gris;            
}

void detectarMovimiento() {
    namedWindow("Movimiento", WINDOW_AUTOSIZE);
    Mat gray;
    resize(frame, frame, Size(), 0.45, 0.45);//Escalado
    cvtColor(frame, gray, COLOR_BGR2GRAY);    
    // Método para clonar imágenes
    frameActual = gray.clone();

    if (frameAnterior.rows == 0 || frameAnterior.cols == 0) {
        frameAnterior = gray.clone();
    }        
    resta = cv::abs(frameActual - frameAnterior);
    frameAnterior = frameActual.clone();

        
    resta=aplyContrast_Threshold(resta);
    resta=aplyFilterGaussian(resta);
    //resta=aplyCanny(resta);
    resta=aplyeApertura(resta);
    //resta=aplyeApertura(resta);
    resta=aplyHerode(resta);    
    //resta=aplyHerode(resta);    
    resta=aplyDilate(resta);    
    
    
    resta=aplyGradient(resta);
    Mat top=aplyTopHat(resta);
    Mat black=aplyBackHat(resta);
    Mat t_b=abs(top-black);


    Mat f;
    add(resta,t_b,f);
    //
    //resta=aplyHerode(resta);
    //
        
    mostrarSinFondo();            
    imshow("Movimiento", f);
    
}

void crearTrackbars(const string &nombre_ventana) {

    createTrackbar("Threshold (m)",nombre_ventana, &mThreshold, 255, nullptr, nullptr);        
    createTrackbar("cStretching (m)",nombre_ventana, &mContrast, 255, nullptr, nullptr);
    createTrackbar("cStretching (k)",nombre_ventana, &kContrast, 200, nullptr, nullptr);
    createTrackbar("Tam Kernel",nombre_ventana, &dimensionK, 60, nullptr, nullptr);
    /*
    createTrackbar("threshold", nombre_ventana, &_threshold, 100, nullptr, nullptr);
    createTrackbar("k", nombre_ventana, &k, 100, nullptr, nullptr);
    createTrackbar("sigma", nombre_ventana, &sigma, 100, nullptr, nullptr);
    createTrackbar("erosion", nombre_ventana, &erosion, 100, nullptr, nullptr);
    createTrackbar("dilation", nombre_ventana, &dilatacion, 100, nullptr, nullptr);*/
}



void leerArchivo(const string &path) {
    VideoCapture video(path);
    if (video.isOpened()) {
        
        while (true) {
            video >> frame;
            
            if (frame.rows == 0 || frame.cols == 0)
                break;
            crearTrackbars("video");
            detectarMovimiento();
            //aplicarEro_Dinata();            
            imshow("video", frame);
            if (waitKey(23) == 27)
                break;
        }
        destroyAllWindows();
    }
}

void activarCamara() {
    VideoCapture video(0);
    //VideoCapture video("/home/trigun/Documentos/Integrador1/panda.mp4");
    if (video.isOpened()) {
        while (true) {
            video >> frame;

            if (frame.rows == 0 || frame.cols == 0)
                break;
            crearTrackbars("video");
            detectarMovimiento();            
            imshow("video", frame);

            if (waitKey(23) == 27)
                break;
        }
        destroyAllWindows();
    }
}


int main() {
    //string path;
    //cin >> path;
    //leerArchivo(path);
    activarCamara();


    return 0;
}
// /home/leo/videoplayback.mp4
// /home/trigun/Documentos/Integrador1/panda.mp4

