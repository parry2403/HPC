/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>

/*
 * TODO: Implement your solutions here
 */
//std::cerr<<"here start"<<std::endl;

void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
	int p, rank, coords[2], blockSize;
	MPI_Comm_rank(comm,&rank);

	MPI_Cart_coords(comm,rank,2,coords);
	MPI_Comm col_comm;
	MPI_Comm_split(comm,coords[1],coords[0],&col_comm);
	
	if (coords[1] == 0)
	{
		MPI_Comm_size(col_comm,&p);
		blockSize = block_decompose(n,p,coords[0]);
		*local_vector = new double[blockSize];
		
		MPI_Scatter(&input_vector[0],blockSize,MPI_DOUBLE,*local_vector,blockSize,MPI_DOUBLE,0,col_comm);
		
		std::cerr<<"Rank = "<<rank<<"values = ";
		for (int i=0;i<blockSize;i++)
			std::cerr<<(*local_vector)[i]<<",";
		std::cerr<<std::endl;


        // double* row_vector = new double[block_decompose_by_dim(n, comm, 0)];
        // transpose_bcast_vector(n, *local_vector, row_vector, comm);
	}
	
	MPI_Comm_free(&col_comm);
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    // TODO
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    // TODO
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    // TODO
    std::cerr<<"Dest = "<<std::endl;
     int p, rank, coords[2], blockSize;
     MPI_Comm_rank(comm,&rank);
     MPI_Cart_coords(comm,rank,2,coords);
    // MPI_Comm trans_comm;
     int row = coords[0];
     int col = coords[1];
     MPI_Comm col_comm;
    
    MPI_Comm_split(comm,col,row,&col_comm);
     
      
     MPI_Comm_size(comm,&p);
     p= sqrt (p);
     if(col==0){
    //     MPI_Comm_size(col_comm,&p);
         blockSize = block_decompose(n,p,coords[0]);
         int des_coords[2];
         des_coords[0]=row;
         des_coords[1]=row;
         int des;
         MPI_Cart_rank( comm, des_coords,  &des);
       //  std::cerr<<"Coordi = "<<des_coords[0]<<des_coords[1]<<std::endl;
         std::cerr<<"Src = "<<rank<<std::endl;
         std::cerr<<"Dest = "<<des<<std::endl;
         MPI_Send(&col_vector[0],blockSize,MPI_DOUBLE,des,1,comm);
     }
     if(row==col)
    {
         blockSize = block_decompose(n,p,coords[0]);
         std::vector<double> results(blockSize);
         int des_coords[2];
         des_coords[0]=row;
         des_coords[1]=0;
         int src;
         MPI_Cart_rank( comm, des_coords,  &src);
         std::cerr<<"Src = "<<src<<std::endl;
         MPI_Recv(&row_vector[0],blockSize,MPI_DOUBLE,src,1,comm,MPI_STATUS_IGNORE);
         

        
       
       //   MPI_Recv(row_vector,blockSize,MPI_DOUBLE,crank,1,col_comm,MPI_STATUS_IGNORE);
    }

     // int des_coords[2];
     // des_coords[0]=col;
     // des_coords[1]=col;
     // int src;
     // MPI_Cart_rank( comm, des_coords,  &src);
     // int crank ;
     // MPI_Comm_rank(col_comm,&crank);
   // std::cerr<<"Receive** = "<<col<<std::endl;
     MPI_Barrier(col_comm);
      std::cerr<<"BCastf ******* = "<<row<<col<<std::endl;
    MPI_Bcast(&row_vector[0],blockSize,MPI_DOUBLE,row,col_comm);
    //  MPI_Barrier(comm);
     //  int des_coords[2];
     //  des_coords[0]=col;
     //  des_coords[1]=col;
     //  MPI_Cart_rank( comm, des_coords,  &src);
     // // int crank ;
     // MPI_Comm_rank(col_comm,&crank);     
    // 
      std::cerr<<"BCast ******* = "<<row<<col<<std::endl;
      
      // for (int i=0;i<blockSize;i++)
      //       std::cerr<<row_vector[i]<<",";
      //   std::cerr<<std::endl;
     
    // MPI_Recv(row_vector,blockSize,MPI_DOUBLE,col,1,col_comm,NULL);
    // MPI_Comm_size(col_comm,&p);
      
        
        // std::cerr<<"Rank = "<<rank<<"values = ";
        // for (int i=0;i<blockSize;i++)
        //     std::cerr<<(*local_vector)[i]<<",";
        // std::cerr<<std::endl;
    
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // TODO
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    // TODO
}


// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
	// distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
	
	int rank, coords[2];
	MPI_Comm_rank(comm,&rank);
	MPI_Cart_coords(comm,rank,2,coords);
	
	if (coords[0] == 0 && coords[1] == 0)
	{
	std::cerr<<"original = ";
	for (int i=0;i<n;i++)
		std::cerr<<*(b+i)<<",";
	std::cerr<<std::endl;
	}
	
    distribute_matrix(n, &A[0], &local_A, comm);
	distribute_vector(n, &b[0], &local_b, comm);
    double* row_vector = new double[block_decompose_by_dim(n, comm, 0)];
    transpose_bcast_vector(n, local_b, row_vector, comm);
	
    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
