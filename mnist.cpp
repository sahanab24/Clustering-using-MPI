#include <fstream>
#include <iostream>
#include <vector>

#include "mpi.h"
#include "Pixel.h"
#include "MnistKMeansMPI.h"

using namespace std;

const int K = 10;

// main test (k-means clustering of Mnist data set)
int main() {

    Pixel *pixels;
    string *pixelLabels;

    int nPixels;
    Pixel::setPixels(&pixels, &pixelLabels, &nPixels);

    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "rank " << rank << " is starting" << std::endl;

    // Set up k-means
    MnistKMeansMPI<K> kMeans;

    if (rank == 0) {
        int nPixels;
        Pixel::setPixels(&pixels, &pixelLabels, &nPixels);
        std::cout << nPixels << " total numbers" << std::endl;
        kMeans.fit(pixels, nPixels);
    } else {
        std::cout << "rank " << rank << " is working" << std::endl;
        kMeans.fitWork(rank);
        MPI_Finalize();
        return 0;
    }

    // get the result
    MnistKMeansMPI<K>::Clusters clusters = kMeans.getClusters();

    // Report the result to console
    int i = 0;
    for (const auto &cluster: clusters) {
        cout << endl
             << endl
             << "cluster #" << ++i
             << endl;
        for (int j: cluster.elements) {
            cout << pixelLabels[j] << " ";
        }
        cout << endl;
    }
    delete[] pixels;
    delete[] pixelLabels;
    MPI_Finalize();
    return 0;
}
