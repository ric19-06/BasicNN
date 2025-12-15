#pragma once

#include <concepts>
#include <vector>
#include <string>

namespace basicnn::linalg{
    
    template <std::floating_point T>
    class Matrix{

        public:

            Matrix(const std::vector<T> & data, size_t rows, size_t cols);
            Matrix(const std::vector<std::vector<T>> & data);
            Matrix() { rows_ = 0; cols_ = 0; };

            Matrix(const Matrix&) = default;
            Matrix& operator=(const Matrix&) = default;
            Matrix(Matrix&&) noexcept = default;
            Matrix& operator=(Matrix&&) noexcept = default;
            ~Matrix() = default;

            size_t rows() const { return rows_; }
            size_t cols() const { return cols_; }
            size_t size() const { return rows_ * cols_; }
            std::vector<T> & data() { return data_; }
            const std::vector<T> & data() const { return data_; }
            void reserve(size_t rows, size_t cols);
            void clear();
            bool empty() const;
            void push_back(const std::vector<T> & v);
            void push_back(const std::vector<T> & v, size_t rows, size_t cols);
            T & operator()(const size_t idx);
            const T & operator()(const size_t idx) const;
            T & operator()(const size_t row, const size_t col);
            const T & operator()(const size_t row, const size_t col) const;
            std::vector<T> & vector();
            Matrix<T> transpose() const;

        private:

            size_t rows_, cols_;
            std::vector<T> data_;
    };
}
#include "matrix.tpp"