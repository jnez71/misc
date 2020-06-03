/*************************************************
Solving the Schrodinger Equation in 2D because why not? This is a rewrite
of quantum.py in C++. To compile with g++ and then run on Linux, first
install OpenCV, then navigate to this directory and do:
    mkdir -p build && g++ quantum.cpp `pkg-config --libs opencv` -o build/quantum -std=c++11 -O3 -ffast-math -march=native && build/quantum
*************************************************/
#include <iostream> // for giving user instructions and debugging
#include <chrono> // for timing
#include <complex> // for complex arithmetic
#include <opencv2/opencv.hpp> // for handling and displaying solution field as image

////////////////////////////////////////////////// ALIASES AND HELPERS

// Python-style printing
template <class T>
inline void print(T const & obj) {
    std::cout << obj << std::endl;
}

// Python-style time.time
inline double time() {
    return 1e-6*std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// // Python-style time.sleep
// #include <thread>
// inline void sleep(double seconds) {
//    std::this_thread::sleep_for(std::chrono::microseconds(int(1e6*seconds)));
// }

////////////////////////////////////////////////// CLASSES

// The wave function solution field is a 2-dimensional grid of complex numbers
struct Wavef {
    int const rows;
    int const cols;
    cv::Mat real;
    cv::Mat imag;

    Wavef(int rows, int cols) :
        rows(rows),
        cols(cols),
        real(rows, cols, CV_64F),
        imag(rows, cols, CV_64F) {
    }

    Wavef(cv::Mat const & real, cv::Mat const & imag) :
        rows(real.rows),
        cols(real.cols),
        real(real),
        imag(imag) {
        if((real.rows != imag.rows) || (real.cols != imag.cols)) {
            throw std::runtime_error("Real and imaginary parts of a Wavef must be the same shape!");
        }
    }

    void operator=(Wavef const & other) {
        real = other.real;
        imag = other.imag;
    }

    Wavef operator-(Wavef const & other) {
        return Wavef(real - other.real, imag - other.imag);
    }

    Wavef operator*(double const scalar) {
        return Wavef(scalar*real, scalar*imag);
    }
};

////////////////////////////////////////////////// INTERFACES

// Scales the L2-norm of a given wave function to 1
inline void normalize(Wavef & F);

// Returns the given wave function divided by i
inline Wavef over_i(Wavef const & F);

// Returns the complex product of two wave functions
inline Wavef cmult(Wavef const & A, Wavef const & B);

// Returns the complex exponential of a given wave function
inline Wavef cexp(Wavef const & F);

// Returns the finite complex Laplacian a given wave function
inline Wavef claplacian(Wavef const & F);

// Displays magnitude and angle information of a given wave function as an image,
// overlays the potential field, and does automatic intensity balancing if bal is true
inline void cdisp(int const img_shape[2], Wavef const & F, cv::Mat const & V, bool bal=true);

////////////////////////////////////////////////// MAIN

int main(int argc, char ** argv) {
    std::cout.precision(4);

    // Physical constants
    double constexpr hbar = 1;
    double constexpr mass = 0.1;
    double constexpr alph = -(hbar*hbar)/(2*mass);
    std::complex<double> constexpr _i(0, 1);

    // Discrete solution space
    double constexpr dt = 0.001;
    int constexpr Lx = 110;
    int constexpr Ly = 110;
    int constexpr img_shape[2] = {3*Lx, 3*Ly};
    int constexpr fig_coords[2] = {750, 250};

    // Initial wave function, Gaussian packet with momentum
    Wavef F(Lx, Ly);
    for(int x=0; x<Lx; ++x) {
        for(int y=0; y<Ly; ++y) {
            std::complex<double> gauss = exp(-0.5*sqrt(5*(pow(x-Lx/2, 2) + pow(y-Ly/6, 2)) - (100.0*y)*_i));
            F.real.at<double>(x, y) = real(gauss);
            F.imag.at<double>(x, y) = imag(gauss);
        }
    }
    normalize(F);

    // Potential function, double slit anyone?
    cv::Mat V(Lx, Ly, CV_64F);
    cv::line(V, cv::Point(Ly/3, 0), cv::Point(Ly/3, Lx/2-12), 50, 2);
    cv::line(V, cv::Point(Ly/3, Lx/2-3), cv::Point(Ly/3, Lx/2+3), 50, 2);
    cv::line(V, cv::Point(Ly/3, Lx/2+12), cv::Point(Ly/3, Lx), 50, 2);
    cv::line(V, cv::Point(Ly, 0), cv::Point(Ly, Lx), 75, 5);

    // Dissipation function on boundaries to effectively "open the box"
    cv::Mat B(Lx, Ly, CV_64F);
    double bthick = 10;
    for(int i=0; i<bthick; ++i) {
        double s = 50*pow(i/bthick, 3);
        B.row(bthick-1-i) = s;
        B.row(Lx-(bthick-1-i)-1) = s;
        B.col(bthick-1-i) = s;
        B.col(Ly-(bthick-1-i)-1) = s;
    }

    // Initialize display figure
    cv::Mat img_boundary = V + B/1000;
    cdisp(img_shape, F, img_boundary);
    cv::moveWindow("Solution", fig_coords[0], fig_coords[1]);

    // Simulation
    print("Press esc to quit.");
    double t = time();
    while(true) {

        // Draw frame
        cdisp(img_shape, F, img_boundary);
        int key = cv::waitKey(1);
        if(key == 27) {
            break;
        }

        // Simulate up to real time
        while(t < time()) {
            F = cmult(cexp(Wavef(-dt*B, -(dt/hbar)*V)), F - over_i(claplacian(F))*(hbar*dt/(2*mass)));
            normalize(F);
            t += dt;
        }
    }

    return 0;
}

////////////////////////////////////////////////// IMPLEMENTATIONS

inline void normalize(Wavef & F) {
    double invmag = 1 / sqrt(pow(cv::norm(F.real), 2) + pow(cv::norm(F.imag), 2));
    F.real *= invmag;
    F.imag *= invmag;
}

/////////////////////////

inline Wavef over_i(Wavef const & F) {
    return Wavef(F.imag, -F.real);
}

/////////////////////////

inline Wavef cmult(Wavef const & A, Wavef const & B) {
    return Wavef(A.real.mul(B.real) - A.imag.mul(B.imag), A.real.mul(B.imag) + A.imag.mul(B.real));
}

/////////////////////////

inline Wavef cexp(Wavef const & F) {
    Wavef eF(F.rows, F.cols);
    std::complex<double> c;
    for(int x=0; x<F.rows; ++x) {
        for(int y=0; y<F.cols; ++y) {
            c = std::exp(std::complex<double>(F.real.at<double>(x, y), F.imag.at<double>(x, y)));
            eF.real.at<double>(x, y) = c.real();
            eF.imag.at<double>(x, y) = c.imag();
        }
    }
    return eF;
}

/////////////////////////

inline Wavef claplacian(Wavef const & F) {
    Wavef DDF(F.rows, F.cols);
    cv::Laplacian(F.real, DDF.real, CV_64F);
    cv::Laplacian(F.imag, DDF.imag, CV_64F);
    return DDF;
}

/////////////////////////

inline void cdisp(int const img_shape[2], Wavef const & F, cv::Mat const & V, bool bal) {
    cv::Mat mags(F.rows, F.cols, CV_64F);
    cv::Mat angs(F.rows, F.cols, CV_64F);
    for(int x=0; x<F.rows; ++x) {
        for(int y=0; y<F.cols; ++y) {
            mags.at<double>(x, y) = sqrt(pow(F.real.at<double>(x, y), 2) + pow(F.imag.at<double>(x, y), 2));
            angs.at<double>(x, y) = (90/M_PI) * (atan2(F.imag.at<double>(x, y), F.real.at<double>(x, y)) + M_PI);
        }
    }
    cv::Mat vals(F.rows, F.cols, CV_64F);
    if(bal) {
        double max_mag; cv::minMaxLoc(mags, nullptr, &max_mag);
        vals = 255*mags/max_mag;
    } else {
        vals = 10000*mags;
    }
    cv::Mat img(F.rows, F.cols, CV_8UC3);
    cv::Mat chn[3]; cv::split(img, chn);
    angs.convertTo(chn[0], CV_8UC1);
    chn[1].setTo(255);
    vals.convertTo(chn[2], CV_8UC1);
    cv::merge(chn, 3, img);
    cv::cvtColor(img, img, cv::COLOR_HSV2BGR);
    double max_v; cv::minMaxLoc(V, nullptr, &max_v);
    cv::Mat Vimg(F.rows, F.cols, CV_64F);
    Vimg = (255/max_v)*V;
    Vimg.convertTo(Vimg, CV_8UC1);
    cv::cvtColor(Vimg, Vimg, cv::COLOR_GRAY2BGR);
    img = img + Vimg;
    cv::resize(img, img, cv::Size(img_shape[0], img_shape[1]));
    cv::imshow("Solution", img);
}
