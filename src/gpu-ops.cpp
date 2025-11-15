#include <RcppBandicoot.h>

//' GPU Matrix Multiplication
//'
//' Multiply two matrices on the GPU using Bandicoot
//'
//' @param A First matrix
//' @param B Second matrix
//' @return Product of A and B computed on GPU
//' @export
// [[Rcpp::export]]
coot::fmat gpu_matrix_multiply(const coot::fmat& A, const coot::fmat& B) {
  return A * B;
}
 
//' GPU Matrix Transpose
//'
//' Transpose a matrix on the GPU using Bandicoot
//'
//' @param A Matrix to transpose
//' @return Transposed matrix computed on GPU
//' @export
// [[Rcpp::export]]
coot::fmat gpu_transpose(const coot::fmat& A) {
 return coot::trans(A);
}

//' GPU Matrix Addition
//'
//' Add two matrices on the GPU using Bandicoot
//'
//' @param A First matrix
//' @param B Second matrix
//' @return Sum of A and B computed on GPU
//' @export
// [[Rcpp::export]]
coot::fmat gpu_matrix_add(const coot::fmat& A, const coot::fmat& B) {
 return A + B;
}

//' GPU Element-wise Operations
//'
//' Apply element-wise square operation on GPU
//'
//' @param A Input matrix
//' @return Matrix with each element squared, computed on GPU
//' @export
// [[Rcpp::export]]
coot::mat gpu_element_square(const coot::mat& A) {
 return coot::square(A);
}

//' GPU Sum
//'
//' Calculate the sum of all elements in a matrix on GPU
//'
//' @param A Input matrix
//' @return Sum of all elements
//' @export
// [[Rcpp::export]]
double gpu_sum(const coot::fmat& A) {
 return coot::accu(A);
}

//' GPU Mean
//'
//' Calculate the mean of all elements in a matrix on GPU
//'
//' @param A Input matrix
//' @return Mean of all elements
//' @export
// [[Rcpp::export]]
double gpu_mean(const coot::fmat& A) {
 return coot::mean(coot::mean(A));
}

//' Create Identity Matrix on GPU
//'
//' Create an identity matrix on the GPU
//'
//' @param n Size of the identity matrix
//' @return n x n identity matrix on GPU
//' @export
// [[Rcpp::export]]
coot::fmat gpu_eye(int n) {
 return coot::eye<coot::fmat>(n, n);
}

//' Create Random Matrix on GPU
//'
//' Create a matrix with random values on the GPU
//'
//' @param n_rows Number of rows
//' @param n_cols Number of columns
//' @return Random matrix on GPU
//' @export
// [[Rcpp::export]]
coot::fmat gpu_randu(int n_rows, int n_cols) {
 return coot::randu<coot::fmat>(n_rows, n_cols);
}
