#include "basicnn/data/data_loader.hpp"
#include <fstream>
#include <assert.h>
#include <iostream>

namespace basicnn::data{

    MnistLoader::MnistLoader(std::string data_file, std::string label_file, int num) : 
        
        size_(0), rows_(0), cols_(0){

            loadImages(data_file, num);
            loadLabels(label_file, num);
    }

    MnistLoader::MnistLoader(std::string data_file, std::string label_file) :

        MnistLoader(data_file, label_file, 0) {}

    int MnistLoader::to_int(char* p){

        return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
            ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
    }

    std::vector<size_t> MnistLoader::indices(){

        std::vector<size_t> ids;
        ids.reserve(size_);
        for(size_t i=0; i<size_; ++i){

            ids.push_back(i);
        }

        return ids;
    }

    void MnistLoader::loadImages(std::string image_file, int num){

        std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
        char p[4];

        ifs.read(p, 4);
        int magic_number = to_int(p);
        assert(magic_number == 0x803);

        ifs.read(p, 4);
        size_ = to_int(p);
        // limit
        if (num != 0 && num < size_) size_ = num;

        ifs.read(p, 4);
        rows_ = to_int(p);

        ifs.read(p, 4);
        cols_ = to_int(p);

        char* q = new char[rows_ * cols_];
        for (size_t i=0; i<size_; ++i){

            ifs.read(q, rows_ * cols_);
            std::vector<float> image(rows_ * cols_);
            for (size_t j=0; j<rows_ * cols_; ++j){

                image[j] = q[j] / 255.0;
            }
            images_.push_back(image);
        }
        delete[] q;

        ifs.close();
    }

    void MnistLoader::loadLabels(std::string label_file, int num){

        std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);
        char p[4];

        ifs.read(p, 4);
        int magic_number = to_int(p);
        assert(magic_number == 0x801);

        ifs.read(p, 4);
        int size = to_int(p);
        // limit
        if (num != 0 && num < size_) size = num;

        for (size_t i=0; i<size; ++i){
            
            ifs.read(p, 1);
            size_t label = p[0];
            labels_.push_back(label);
        }

        ifs.close();
    }
}