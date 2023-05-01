#include "Pixel.h"

#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

Pixel::Pixel() {}

Pixel::Pixel(const array<u_char, 784> as_784array) : pixels(as_784array) {}

double Pixel::euclidDistance(const Pixel &other) const {
    double total = 0;
    for (int i = 0; i < 784; i++) {
        double temp = 0;
        temp = (double) this->pixels[i] - (double) other.pixels[i];
        temp = temp * temp;
        total += temp;
    }
    return sqrt(total);
}

void setPixel(Pixel *pixels, string *names, int i, string *pixel_array, int size, string name) {
    Pixel each_pixel;
    for (int i = 0; i < size; i++) {
        int temp = std::stoi(pixel_array[i]);
        each_pixel.pixels[i] = (u_char) temp;
    }
    pixels[i] = each_pixel;
    names[i] = std::move(name);
}

vector <vector<string>> parseCSV(string filename) {
    vector <vector<string>> data;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        vector <string> row;
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }
    file.close();
    return data;
}

void Pixel::setPixels(Pixel **data, string **labels, int *size) {
    const int MAX = 200000;
    auto *pixels = new Pixel[MAX];
    auto *names = new string[MAX];
    std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type i = 0;
    vector <vector<string>> csvData = parseCSV("train.csv");
    for (; i < csvData.size(); i++) {
        if (i == 0) {
            continue;
        }
        string label;
        string *csv_pixels = new string[csvData.at(i).size() - 1];
        for (std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type j = 0; j < csvData.at(i).size(); j++) {
            if (j == 0) {
                label = csvData.at(i).at(j);
            } else {
                csv_pixels[j - 1] = csvData.at(i).at(j);
            }

        }
        setPixel(pixels, names, i - 1, csv_pixels, csvData.at(i).size() - 1, label);
    }
    *data = pixels;
    *labels = names;
    *size = i - 1;
}
