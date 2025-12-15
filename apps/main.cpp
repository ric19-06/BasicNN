#include "basicnn/data/data_loader.hpp"
#include "basicnn/data/dataset.hpp"
#include "basicnn/utils.hpp"
#include "basicnn/nn/training.hpp"
#include "basicnn/nn/layer.hpp"
#include "basicnn/nn/neural_network.hpp"
#include "basicnn/nn/optimization.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

// Add python path
namespace fs = std::filesystem;
namespace py = pybind11;

using namespace basicnn::data;
using namespace basicnn::utils;
using namespace basicnn::nn;
using namespace basicnn::nn::functional;
using namespace basicnn::nn::optimization;

using ftype = float;

enum class OptimizerType{

    SGD,
    ADAM
};

int main(int argc, char* argv[]){

    // Settings
    std::vector<size_t> classes = {0,1,2,3,4,5,6,7,8,9};
    size_t N_classes = classes.size();
    size_t batchsize = 256;
    float droprate = 0.2;
    size_t seed = 42;
    OptimizerType optimizerType = OptimizerType::ADAM;
    size_t patience = 10;
    float min_delta = 1e-3f;
    size_t epochs = 100;
    float lr = 1e-2f;

    // Fix paths
    fs::path exe_path = fs::canonical(argv[0]);
    fs::path exe_dir  = exe_path.parent_path();
    fs::path project_root = exe_dir / ".." / "..";
    fs::path python_dir = project_root / "python";
    fs::path venv_path = project_root / ".venv";
    fs::path site_packages = venv_path / "lib" / "python3.12" / "site-packages";

    py::scoped_interpreter guard{};
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(0, site_packages.string());
    sys.attr("path").attr("insert")(0, python_dir.string());

    // Dataset
    auto trainloader = std::make_shared<MnistLoader>("dataset/train-images-idx3-ubyte",
                                                     "dataset/train-labels-idx1-ubyte");
    auto [train_ids, val_ids] = split(trainloader, N_classes, 0.85, seed);
    Dataset<std::vector<ftype>, size_t> trainset(trainloader, batchsize, seed, val_ids);
    Dataset<std::vector<ftype>, size_t> valset(trainloader, batchsize, seed, train_ids);

    auto testloader = std::make_shared<MnistLoader>("dataset/t10k-images-idx3-ubyte",
                                                    "dataset/t10k-labels-idx1-ubyte");
    Dataset<std::vector<ftype>, size_t> testset(testloader, 1, seed);

    // Model
    std::vector<size_t> layers = {28*28, 512, 256, 10};
    std::vector<ActivationType> activations = {ActivationType::RELU, ActivationType::RELU, ActivationType::SOFTMAX};
    NeuralNetwork model(layers, activations, droprate, seed);
    model.weightInit();

    // Optimizer
    auto modelWeights = model.getWeights();
    auto modelGradients = model.getGradients();
    std::unique_ptr<Optimizer> optimizer;
    if (optimizerType == OptimizerType::SGD){
        optimizer = std::make_unique<SGD> (modelWeights, modelGradients, lr);
    } else if (optimizerType == OptimizerType::ADAM) {
        optimizer = std::make_unique<ADAM> (modelWeights, modelGradients, lr);
    }

    // LRScheduler
    std::unique_ptr<LRScheduler> lrschedule = std::make_unique<StepLR> (lr, 20, 0.5f);

    // Early stop
    EarlyStop earlystop(patience, min_delta);
    
    // Train the model
    model.setStatus(Status::TRAINING);
    auto [train_loss, val_loss] = train_model(
        model,
        trainset,
        *optimizer,
        *lrschedule,
        & crossEntropy,
        & crossEntropy_grad,
        valset,
        & crossEntropy,
        epochs,
        N_classes,
        earlystop
    );

    // Get the predictions
    auto [ytrue, yhat] = get_predictions(
        model,
        testset,
        N_classes
    );

    // Import the visualization lib
    py::module visualization = py::module::import("visualization");

    // Send train_loss and val_loss to python for loss plot
    visualization.attr("plot_loss")(train_loss, val_loss);

    // Send ytrue and yhat to python for performances
    visualization.attr("get_performances")(ytrue, yhat, classes, true);

    return 0;
}