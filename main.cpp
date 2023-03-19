#include <iostream>
#include <cmath>
#include <chrono>
#include <float.h>

using namespace std::chrono;

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
        float sum = 0;
        for (int dim = 0; dim < DIMENSIONS; dim++)
            sum += pow(features[dim] - other.features[dim], 2);

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



// SSD between d1 and d2 is 7.30509e-05
// SAD between d1 and d2 is 0.000677925