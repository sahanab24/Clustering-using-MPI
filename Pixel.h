#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <array>

using namespace std;

class Pixel {
public:
    std::array<u_char, 784> pixels;

    Pixel();

    Pixel(std::array<u_char, 784> as_784array);

    // distance, etc.
    double euclidDistance(const Pixel &other) const;

    // color set
    static void setPixels(Pixel **data, std::string **labels, int *size);
};
