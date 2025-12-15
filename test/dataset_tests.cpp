#include "basicnn/data/dataset.hpp"
#include "basicnn/data/data_loader.hpp"
#include "basicnn/linalg/linalg.hpp"
#include <gtest/gtest.h>

using namespace basicnn::linalg;
using namespace basicnn::data;

TEST(DatasetTest, Init){

    auto trainloader = std::make_shared<MnistLoader>("dataset/train-images-idx3-ubyte",
                                                            "dataset/train-labels-idx1-ubyte", 100);

    Dataset<std::vector<float>, size_t> trainset(trainloader, 64, 42, {0});
    auto ids = trainloader->indices();
    auto [data, label] = trainset.batch(0);

    EXPECT_EQ(label[0], trainloader->label(1));
}

TEST(DatasetTest, Shuffle){

    auto trainloader = std::make_shared<MnistLoader>("dataset/train-images-idx3-ubyte",
                                                            "dataset/train-labels-idx1-ubyte", 100);

    Dataset<std::vector<float>, size_t> trainset(trainloader, 64, 42);

    auto [data, label] = trainset.batch(0);
    trainset.shuffle();
    auto [data1, label1] = trainset.batch(0);
    trainset.shuffle();
    auto [data2, label2] = trainset.batch(0);

    EXPECT_NE(label, label1);
    EXPECT_NE(label2, label1);
}