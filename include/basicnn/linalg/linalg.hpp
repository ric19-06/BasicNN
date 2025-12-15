#pragma once

#include "matrix.hpp"
#include <vector>
#include <concepts>
#include <ostream>

namespace basicnn::linalg{

    // Scalar-vector oveloads
    template <std::floating_point T>
    std::vector<T> operator*(const std::vector<T> & v, const T s);

    template <std::floating_point T>
    std::vector<T> operator*(const T s, const std::vector<T> & v);

    template <std::floating_point T>
    std::vector<T> operator/(const std::vector<T> & v, const T s);


    // Vector-vector overloads
    template <std::floating_point T>
    std::vector<T> operator+(const std::vector<T> & a, const std::vector<T> & b);

    template <std::floating_point T>
    std::vector<T> operator-(const std::vector<T> & a, const std::vector<T> & b);

    template <std::floating_point T>
    std::vector<T> operator*(const std::vector<T> & a, const std::vector<T> & b); // Element-wise

    template <std::floating_point T>
    std::vector<T> operator/(const std::vector<T> & a, const std::vector<T> & b); // Element-wise

    template <std::floating_point T>
    bool operator==(const std::vector<T> & a, const std::vector<T> & b);


    // Useful vector operations
    template <std::floating_point T>
    T dot(const std::vector<T> & a, const std::vector<T> & b);

    template <std::floating_point T>
    std::vector<T> pow(const std::vector<T> & v, T s); // Element-wise

    template <std::floating_point T>
    std::vector<T> sqrt(const std::vector<T> & v); // Element-wise


    // Scalar-matrix overloads
    template <std::floating_point T>
    Matrix<T> operator*(const Matrix<T> & M, const T s);

    template <std::floating_point T>
    Matrix<T> operator*(const T s, const Matrix<T> & M);

    template <std::floating_point T>
    Matrix<T> operator/(const Matrix<T> & M, const T s);


    // Vector-matrix overload
    template <std::floating_point T>
    std::vector<T> operator*(const Matrix<T> & M, const std::vector<T> & v);

    template <std::floating_point T>
    std::vector<T> operator*(const std::vector<T> & v, const Matrix<T> & M);


    // Matrix-matrix overloads
    template <std::floating_point T>
    Matrix<T> operator+(const Matrix<T> & A, const Matrix<T> & B);

    template <std::floating_point T>
    Matrix<T> operator-(const Matrix<T> & A, const Matrix<T> & B);

    template <std::floating_point T>
    Matrix<T> operator*(const Matrix<T> & A, const Matrix<T> & B);

    template <std::floating_point T>
    Matrix<T> operator/(const Matrix<T> & A, const Matrix<T> & B); // Element-wise

    template <std::floating_point T>
    bool operator==(const Matrix<T> & A, const Matrix<T> & B);

    // Useful matrix operations
    template <std::floating_point T>
    Matrix<T> hadamard(const Matrix<T> & A, const Matrix<T> & B);

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> pow(const basicnn::linalg::Matrix<T> & M, T s); // Element-wise

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> sqrt(const basicnn::linalg::Matrix<T> & M); // Element-wise

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> log(const basicnn::linalg::Matrix<T> & M); // Element-wise
}

#include "linalg.tpp"