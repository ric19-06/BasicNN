#pragma once

#include "matrix.hpp"
#include <vector>
#include <concepts>
#include <stdexcept>
#include <omp.h>
#include <cpptrace/cpptrace.hpp>
#include <cmath>

#define INVALID_ARGUMENT(msg) cpptrace::invalid_argument::exception_with_message(msg);

namespace basicnn::linalg{

    // Scalar-vector oveloads
    template <std::floating_point T>
    std::vector<T> mult_scalar_vector(const T s, const std::vector<T> & v){

        std::vector<T> u;
        u.reserve(v.size());

        for(size_t i=0; i<v.size(); ++i){

            u.push_back(s * v[i]);
        }

        return u;
    }

    template <std::floating_point T>
    std::vector<T> operator*(const std::vector<T> & v, const T s){

        return mult_scalar_vector(s, v);
    }

    template <std::floating_point T>
    std::vector<T> operator*(const T s, const std::vector<T> & v){

        return mult_scalar_vector(s, v);
    }

    template <std::floating_point T>
    std::vector<T> operator/(const std::vector<T> & v, const T s){

        if (s == T(0))
            throw INVALID_ARGUMENT("Division by zero.");

        return mult_scalar_vector(T(1)/s, v);
    }


    // Vector-vector overloads
    template <std::floating_point T>
    std::vector<T> operator+(const std::vector<T> & a, const std::vector<T> & b) {

        if (a.size() != b.size())
            throw INVALID_ARGUMENT("Vector sizes must match.");

        std::vector<T> c;
        c.reserve(a.size());

        for (size_t i = 0; i < a.size(); ++i)
            c.push_back(a[i] + b[i]);

        return c;
    }

    template <std::floating_point T>
    std::vector<T> operator-(const std::vector<T> & a, const std::vector<T> & b){

        if (a.size() != b.size())
            throw INVALID_ARGUMENT("Vector sizes must match.");

        std::vector<T> c;
        c.reserve(a.size());

        for (size_t i = 0; i < a.size(); ++i)
            c.push_back(a[i] - b[i]);

        return c;
    }

    template <std::floating_point T>
    std::vector<T> operator*(const std::vector<T> & a, const std::vector<T> & b) {

        // Element-wise
        if (a.size() != b.size())
            throw INVALID_ARGUMENT("Vector sizes must match.");

        std::vector<T> c;
        c.reserve(a.size());

        for (size_t i = 0; i < a.size(); ++i)
            c.push_back(a[i] * b[i]);

        return c;
    }

    template <std::floating_point T>
    std::vector<T> operator/(const std::vector<T> & a, const std::vector<T> & b){

        // Element-wise
        if (a.size() != b.size())
            throw INVALID_ARGUMENT("Vector sizes must match.");
        
        std::vector<T> c;
        c.reserve(a.size());

        for (size_t i = 0; i < a.size(); ++i)
            c.push_back(a[i] / b[i]);

        return c;
    }

    template <std::floating_point T>
    bool operator==(const std::vector<T> & a, const std::vector<T> & b){

        if (a.size() != b.size())
            return false;

        for(size_t i=0; i<a.size(); ++i){

            if (a[i] != b[i])
                return false;
        }

        return true;
    }


    // Useful vector operations
    template <std::floating_point T>
    T dot(const std::vector<T> & a, const std::vector<T> & b){

        std::vector<T> product = a * b;
        T c = 0;
        for(size_t i=0; i<product.size(); ++i){

            c += product[i];
        }

        return c;
    }

    template <std::floating_point T>
    std::vector<T> pow(const std::vector<T> & v, T s){

        // Element-wise
        std::vector<T> u;
        u.reserve(v.size());
        for(size_t i=0; i<v.size(); ++i){

            u.push_back(std::pow(v[i], s));
        }

        return u;
    }

    template <std::floating_point T>
    std::vector<T> sqrt(const std::vector<T> & v){

        std::vector<T> u;
        u.reserve(v.size());
        for(size_t i=0; i<v.size(); ++i){

            if (v[i] < 0.0)
                throw INVALID_ARGUMENT("Vector elements must be positive.");

            u.push_back(std::sqrt(v[i]));
        }

        return u;
    }


    // Scalar-matrix overloads
    template <std::floating_point T>
    Matrix<T> mult_scalar_matrix(const T s, const Matrix<T> & M){

        std::vector<T> data;
        data.reserve(M.size());
        for(size_t i=0; i<M.rows(); ++i) {
            for(size_t j=0; j< M.cols(); ++j) {

                data.push_back(s * M(i,j));
            }
        }

        Matrix<T> Q(data, M.rows(), M.cols());

        return Q;
    }

    template <std::floating_point T>
    Matrix<T> operator*(const Matrix<T> & M, const T s){

        return mult_scalar_matrix(s, M);
    }

    template <std::floating_point T>
    Matrix<T> operator*(const T s, const Matrix<T> & M){

        return mult_scalar_matrix(s, M);
    }

    template <std::floating_point T>
    Matrix<T> operator/(const Matrix<T> & M, const T s){

        if (s == T(0))
            throw INVALID_ARGUMENT("Division by zero.");

        return mult_scalar_matrix(T(1)/s, M);
    }

    
    // Vector-matrix overload
    template <std::floating_point T>
    std::vector<T> operator*(const Matrix<T> & M, const std::vector<T> & v){

        Matrix<T> V(v, v.size(), 1);
        Matrix<T> U = M * V;

        return U.vector();
    }

    template <std::floating_point T>
    std::vector<T> operator*(const std::vector<T> & v, const Matrix<T> & M){

        Matrix<T> V(v, 1, v.size());
        Matrix<T> U = V * M;

        return U.vector();
    }


    // Matrix-matrix overloads
    template <std::floating_point T>
    Matrix<T> operator+(const Matrix<T> & A, const Matrix<T> & B){

        if (A.rows() != B.rows() || A.cols() != B.cols())
            throw INVALID_ARGUMENT("Matrix dimensions must match.");

        std::vector<T> data;
        data.reserve(A.size());
        for(size_t i=0; i<A.rows(); ++i){
            for(size_t j=0; j<A.cols(); ++j){

                data.push_back(A(i,j) + B(i,j));
            }
        }
        Matrix<T> C(data, A.rows(), A.cols());

        return C;
    }

    template <std::floating_point T>
    Matrix<T> operator-(const Matrix<T> & A, const Matrix<T> & B){

        if (A.rows() != B.rows() || A.cols() != B.cols())
            throw INVALID_ARGUMENT("Matrix dimensions must match.");

        std::vector<T> data;
        data.reserve(A.size());
        for(size_t i=0; i<A.rows(); ++i){
            for(size_t j=0; j<A.cols(); ++j){

                data.push_back(A(i,j) - B(i,j));
            }
        }
        Matrix<T> C(data, A.rows(), A.cols());

        return C;
    }

    template <std::floating_point T>
    void multiply_unblocked(const Matrix<T> & A, const Matrix<T> & B, Matrix<T> & C){

        size_t i,k,j;

        const std::vector<T> & a = A.data();
        const std::vector<T> & b = B.data();
        std::vector<T> & c = C.data();

        size_t A_cols = A.cols();
        size_t B_cols = B.cols();

        // Cache aware loop (flip k and j)
        #pragma omp parallel for private(k,j) shared(A,B,C)
        for(i=0; i<A.rows(); ++i){
            for(k=0; k<A.cols(); ++k){
                for(j=0; j<B.cols(); ++j){

                    c[i*B_cols+j] += a[i*A_cols+k] * b[k*B_cols+j];
                }
            }
        }
    }

    template <std::floating_point T>
    void multiply_blocked(const Matrix<T> & A, const Matrix<T> & B, Matrix<T> & C, size_t bs){

        size_t ii,jj,kk,i,j,k;

        const std::vector<T> & a = A.data();
        const std::vector<T> & b = B.data();
        std::vector<T> & c = C.data();

        size_t A_cols = A.cols();
        size_t B_cols = B.cols();

        #pragma omp parallel for collapse(2) shared(A,B,C) schedule(static)
        for (ii=0; ii<A.rows(); ii+=bs){
            for (jj=0; jj<B.cols(); jj+=bs){
                for (kk=0; kk<A.cols(); kk+=bs){

                    size_t i_end = std::min(ii + bs, A.rows());
                    size_t j_end = std::min(jj + bs, B.cols());
                    size_t k_end = std::min(kk + bs, A.cols());

                    // Cache aware loop (flip k and j)
                    for (size_t i=ii; i<i_end; ++i){
                        for (size_t k=kk; k<k_end; ++k){
                            for (size_t j=jj; j<j_end; ++j){

                                c[i*B_cols+j] += a[i*A_cols+k] * b[k*B_cols+j];
                            }
                        }
                    }
                }
            }
        }
    }

    template <std::floating_point T>
    Matrix<T> operator*(const Matrix<T> & A, const Matrix<T> & B){

        if (A.cols() != B.rows())
            throw INVALID_ARGUMENT("Matrix A column size must match matrix B row size.");

        // Initialize C
        size_t C_rows = A.rows();
        size_t C_cols = B.cols();
        size_t C_size = C_rows * C_cols;
        Matrix<T> C(std::vector<T> (C_rows * C_cols, T(0)), C_rows, C_cols);
        
        // Set block size
        size_t bs = 64;

        // Number of threads
        size_t threads = omp_get_num_procs();

        // Compute the number of blocks
        size_t NB = ceil(C_size / bs);

        // If there are more blocks than threads parallelize over blocks
        if(NB > threads){
            multiply_blocked(A,B,C,bs);

        // Otherwise parallelize over elements
        } else {
            multiply_unblocked(A,B,C);
        }

        return C;
    }

    template <std::floating_point T>
    Matrix<T> operator/(const Matrix<T> & A, const Matrix<T> & B){

        if (A.rows() != B.rows() || A.cols() != B.cols())
            throw INVALID_ARGUMENT("Matrix dimensions must match.");

        std::vector<T> data;
        data.reserve(A.size());
        for(size_t i=0; i<A.rows(); ++i){
            for(size_t j=0; j<A.cols(); ++j){

                data.push_back(A(i,j) / B(i,j));
            }
        }
        Matrix<T> C(data, A.rows(), A.cols());

        return C;
    }

    template <std::floating_point T>
    bool operator==(const Matrix<T> & A, const Matrix<T> & B){

        if (A.rows() != B.rows() || A.cols() != B.cols())
            return false;

        for(size_t i=0; i<A.rows(); ++i){
            for(size_t j=0; j<A.cols(); ++j){

                if (A(i,j) != B(i,j))
                    return false;
            }
        }

        return true;
    }


    // Useful matrix operations
    template <std::floating_point T>
    Matrix<T> hadamard(const Matrix<T> & A, const Matrix<T> & B){

        if (A.rows() != B.rows() || A.cols() != B.cols())
            throw INVALID_ARGUMENT("Matrix dimensions must match.");

        std::vector<T> data;
        data.reserve(A.size());
        for(size_t i=0; i<A.rows(); ++i){
            for(size_t j=0; j<A.cols(); ++j){

                data.push_back(A(i,j) * B(i,j));
            }
        }
        Matrix<T> C(data, A.rows(), A.cols());

        return C;
    }

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> pow(const basicnn::linalg::Matrix<T> & M, T s){

        // Element-wise
        using namespace basicnn::linalg;

        std::vector<T> data;
        data.reserve(M.size());
        for(size_t i=0; i<M.size(); ++i){

            data.push_back(std::pow(M(i), s));
        }

        return Matrix<T>(data, M.rows(), M.cols());
    }

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> sqrt(const basicnn::linalg::Matrix<T> & M){

        using namespace basicnn::linalg;

        std::vector<T> data;
        data.reserve(M.size());
        for(size_t i=0; i<M.size(); ++i){

            if (M(i) < 0.0)
                throw INVALID_ARGUMENT("Matrix elements must be positive.");

            data.push_back(std::sqrt(M(i)));
        }

        return Matrix<T>(data, M.rows(), M.cols());
    }

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> log(const basicnn::linalg::Matrix<T> & M){

        using namespace basicnn::linalg;

        std::vector<T> data;
        data.reserve(M.size());
        for(size_t i=0; i<M.size(); ++i){

            if (M(i) <= 0.0)
                throw INVALID_ARGUMENT("Matrix elements must be positive.");

            data.push_back(std::log(M(i)));
        }

        return Matrix<T>(data, M.rows(), M.cols());
    }
}