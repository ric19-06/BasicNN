#pragma once

#include "matrix.hpp"
#include <concepts>
#include <vector>
#include <string>
#include <iostream>
#include <cpptrace/cpptrace.hpp>

#define INVALID_ARGUMENT(msg) cpptrace::invalid_argument::exception_with_message(msg);
#define OUT_OF_RANGE(msg) cpptrace::out_of_range::exception_with_message(msg);

namespace basicnn::linalg{

    template <std::floating_point T>
    Matrix<T>::Matrix(const std::vector<T> & data, size_t rows, size_t cols) : 
        data_(data), rows_(rows), cols_(cols) {

            if (data_.size() != (rows_ * cols_))
                throw INVALID_ARGUMENT("Size of data should match rows * cols.")
    }

    template <std::floating_point T>
    Matrix<T>::Matrix(const std::vector<std::vector<T>> & data){

        rows_ = 0;
        cols_ = 0;
        for(size_t i=0; i<data.size(); ++i){

            push_back(data[i]);
        }
    }

    template <std::floating_point T>
    void Matrix<T>::reserve(size_t rows, size_t cols){

        data_.reserve(rows * cols);
    }

    template <std::floating_point T>
    void Matrix<T>::clear(){

        data_.clear();
        rows_ = 0;
        cols_ = 0;
    }

    template <std::floating_point T>
    bool Matrix<T>::empty() const {

        return (rows_ == 0 && cols_ == 0) && data_.empty();
    }

    template <std::floating_point T>
    void Matrix<T>::push_back(const std::vector<T> & v){

        if (cols_ != 0 && cols_ != v.size())
            throw INVALID_ARGUMENT("Input row size and column size must match.");
        
        if (cols_ == 0)
            cols_ = v.size();

        for(size_t i=0; i<v.size(); ++i){

            data_.push_back(v[i]);
        }
        rows_++;
    }

    template <std::floating_point T>
    void Matrix<T>::push_back(const std::vector<T> & v, size_t rows, size_t cols){

        if (v.size() != rows * cols)
            throw INVALID_ARGUMENT("Size of v should match rows * cols.");

        data_ = v;
        rows_ = rows;
        cols_ = cols;
    }

    template <std::floating_point T>
    T & Matrix<T>::operator()(const size_t idx){

        if (idx >= rows_ * cols_)
            throw OUT_OF_RANGE("Argument idx out of bounds.");

        return data_[idx];
    }

    template <std::floating_point T>
    const T & Matrix<T>::operator()(const size_t idx) const{

        if (idx >= rows_ * cols_)
            throw OUT_OF_RANGE("Argument idx out of bounds.");

        return data_[idx];
    }

    template <std::floating_point T>
    T & Matrix<T>::operator()(const size_t row, const size_t col){

        if (row >= rows_){
            throw OUT_OF_RANGE("Argument row out of bounds.");
        } else if (col >= cols_){
            throw OUT_OF_RANGE("Argument col out of bounds.");
        }

        return data_[row * cols_ + col];
    }

    template <std::floating_point T>
    const T & Matrix<T>::operator()(const size_t row, const size_t col) const{

        if (row >= rows_){

            throw OUT_OF_RANGE("Argument row out of bounds.");
        } else if (col >= cols_){
            throw OUT_OF_RANGE("Argument col out of bounds.");
        }

        return data_[row * cols_ + col];
    }

    template <std::floating_point T>
    std::vector<T> & Matrix<T>::vector(){

        if (!empty() && (rows_ != 1 && cols_ != 1))
            throw INVALID_ARGUMENT("Matrix is not a vector.");

        return data_;
    }

    template <std::floating_point T>
    Matrix<T> Matrix<T>::transpose() const{

        std::vector<T> data;
        data.reserve(rows_ * cols_);

        for(size_t j=0; j<cols_; ++j){
            for(size_t i=0; i<rows_; ++i){

                data.push_back(data_[i * cols_ + j]);
            }
        }
        
        return Matrix<T>(data, cols_, rows_);
    }
}