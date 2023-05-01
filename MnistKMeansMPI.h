#pragma once

#include "KMeansMPI.h"
#include "Pixel.h"

template<int k>
class MnistKMeansMPI : public KMeansMPI<k, 784> {
public:
    void fit(Pixel *pixels, int n) {
        // We know that a Color is actually just an array of three bytes so the cast is ok
        // NOTE: this will stop working correctly if the Color data layout is changed in any way
        KMeansMPI<k, 784>::fit(reinterpret_cast<std::array<u_char, 784> *>(pixels), n);
    }

protected:
    typedef std::array<u_char, 784> Element;

    /**
     * We supply the distance method to the abstract KMeansMPI class
     * We use the Euclidean distance between the colors interpreted as 3-d vectors in R,G,B space
     * @param a one color
     * @param b and another
     * @return distance between a and b; 0.0 <= distance <= 441.67 (sqrt(255^2 + 255^2 + 255^2))
     */
    double distance(const Element &a, const Element &b) const override {
        return Pixel(a).euclidDistance(Pixel(b));
    }
};
