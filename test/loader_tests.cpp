#include "basicnn/data/data_loader.hpp"
#include <gtest/gtest.h>

using namespace basicnn::data;

TEST(LoaderTest, Indices){

    MnistLoader train("dataset/train-images-idx3-ubyte",
                "dataset/train-labels-idx1-ubyte", 100);
    
    std::vector<size_t> ids = train.indices();
    EXPECT_EQ(ids[0], 0);
    EXPECT_EQ(ids[99], 99);
}

TEST(LoaderTest, LabelValue){

    MnistLoader train("dataset/train-images-idx3-ubyte",
                "dataset/train-labels-idx1-ubyte", 100);
                
    double label = train.label(0);
    EXPECT_EQ(label, 5);
}

TEST(LoaderTest, ImageSize){

    MnistLoader train("dataset/train-images-idx3-ubyte",
                "dataset/train-labels-idx1-ubyte", 100);
    
    std::vector<float> image  = train.data(0);
    EXPECT_EQ(image.size(), 28*28);
}