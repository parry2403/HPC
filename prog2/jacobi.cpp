/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>
#include "utils.h"
// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    for (int i=0;i<n;i++)
	{
    	double sum = 0.0;
    	for(int j = 0;j<n;j++)
    		sum += A[i*n+j]*x[j];
    	y[i] = sum;
    }
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
     for (int i=0;i<n;i++)
	 {
    	double sum = 0.0;
    	for(int j = 0;j<m;j++)
    		sum +=A[i*n+j]*x[j];
    	y[i] = sum;
    }
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
	std::vector<double> D(n);
	std::vector<double> invD(n);
	std::vector<double> R(n*n);
	std::vector<double> y(n);
	std::vector<double> s(n);
	init(n,&x[0]);
	diagonal( n, &A[0], &D[0]);
	inverseDiagonal(n,&D[0],&invD[0]);
	nonDiagonal(n,&A[0],&R[0]);
	matrix_vector_mult(n,&A[0],&x[0],&y[0]);
	vectorSub(n,&y[0],&b[0],&s[0]);
	
	int iter = 0;
	while(norm(n,&s[0]) > l2_termination && iter < max_iter)
	{
		matrix_vector_mult(n,&R[0],&x[0],&y[0]);
		vectorSub(n,&b[0],&y[0],&s[0]);
		vectorMult(n,&invD[0],&s[0],&x[0]);

		matrix_vector_mult(n,&A[0],&x[0],&y[0]);
		vectorSub(n,&y[0],&b[0],&s[0]);	
	}
}