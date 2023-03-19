#include <iostream>
#include <cmath>
#include <chrono>
#include <float.h>
#include <immintrin.h>

using namespace std::chrono;

// Github link:
// https://github.com/catalinasirbu/Improve-SSD-and-SAD-running-time-with-SIMD.git

// Output:
// SSD between d1 and d2 is 7.30509e-05
// SAD between d1 and d2 is 0.000677925
// Done in 27653 ms

// CPU used: Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz
// gcc (x86_64-posix-seh-rev2, Built by MinGW-W64 project) 12.2.0


/* ------------------------------------ SSD IMPROVEMENTS ------------------------------------
 * To improve the SSD running time with SIMD, we can use vector instructions to perform multiple
 * operations simultaneously. In particular, we can use the SIMD instructions available in the AVX2
 * instruction set to perform the sum of squared differences operation on four floats at once.
 */

#define DIMENSIONS 128
class Descriptor {
    float features[DIMENSIONS];
public:
    Descriptor() {
        for (int dim = 0; dim < DIMENSIONS; dim++)
            features[dim] = rand() / (float)INT32_MAX;
    }

    // sum of squared differences == L2-norm
    float getSSDdistance(Descriptor other) {
        float sum = 0.0f;
        __m128 diff, prod;
        __m128 s = _mm_setzero_ps();
        for (int dim = 0; dim < DIMENSIONS; dim += 4) {
            diff = _mm_sub_ps(_mm_load_ps(&features[dim]), _mm_load_ps(&other.features[dim]));
            prod = _mm_mul_ps(diff, diff);
            s = _mm_add_ps(s, prod);
        }
        __declspec(align(16)) float temp[4];
        _mm_store_ps(temp, s);
        sum = temp[0] + temp[1] + temp[2] + temp[3];
        return sqrt(sum);
    }

    // sum of absolute differences == L1-norm
    float getSADdistance(Descriptor other) {
        float sum = 0;
        for (int dim = 0; dim < DIMENSIONS; dim++)
            sum += fabs(features[dim] - other.features[dim]);

        return sum;
    }
};

#define NUM_OF_POINTS 10000
Descriptor set1[NUM_OF_POINTS];
Descriptor set2[NUM_OF_POINTS];

// indices[A] == B if the closest point in set2 to point A of set1,
//  is B
unsigned indicesNOSIMD[NUM_OF_POINTS];
unsigned indicesSIMD[NUM_OF_POINTS];

int main() {
    Descriptor d1;
    Descriptor d2;

    std::cout << "SSD between d1 and d2 is " <<
              d1.getSSDdistance(d2)<< std::endl;

    std::cout << "SAD between d1 and d2 is " <<
              d1.getSADdistance(d2)<< std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_OF_POINTS; i++) {
        float mindistance = FLT_MAX;
        for (int j = 0; j < NUM_OF_POINTS; j++) {
            float dist = set1[i].getSSDdistance(set2[j]);
            if (dist < mindistance){
                mindistance = dist;
                indicesNOSIMD[i] = j;
            }
        }
    }
    auto stop = high_resolution_clock::now();
    std::cout << "Done in " << duration_cast<milliseconds>
            (stop-start).count() << " ms" << std::endl;

    return 0;
}