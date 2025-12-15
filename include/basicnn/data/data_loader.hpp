#pragma once

#include <vector>
#include <string>

namespace basicnn::data{

    template <typename Data, typename Label>
    class DataLoader{

        public:

            virtual ~DataLoader() = default;

            virtual size_t size() const = 0;

            virtual std::vector<size_t> indices() = 0;
            virtual Data data(size_t id) const = 0;
            virtual Label label(size_t id) const = 0;
    };

    class MnistLoader : public DataLoader<std::vector<float>, size_t>{

        public:

            MnistLoader(std::string data_file, std::string label_file, int num);
            MnistLoader(std::string data_file, std::string label_file);
            ~MnistLoader() override = default;

            size_t size() const override { return size_; }

            std::vector<size_t> indices() override;
            std::vector<float> data(size_t id) const override { return images_[id]; }
            size_t label(size_t id) const override { return labels_[id]; }

        private:
            std::vector<std::vector<float>> images_;
            std::vector<size_t> labels_;

            size_t size_;
            size_t rows_;
            size_t cols_;

            void loadImages(std::string image_file, int num=0);
            void loadLabels(std::string label_file, int num=0);
            int to_int(char* p);
    };
}