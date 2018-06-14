
#include <stdio.h>
#include <stdlib.h>


void Gen_matrix(double* matrix, int n);
void Gen_rand_matrix(double* matrix, int n);
void Gen_b_vector(double *A, int n, double *x, double *b);
void Multiply_transpose(double *mat, int n, double *ret_mat);
void Scal_vec_product(double *vec, int n, double scalar, double *ret_vec); 
void Mat_vec_prod(double *mat, double *vec, int col, int rows, double *ret_vec);
void Print_mat(double *mat, int n, const char label[]);
void Print_vec(double* vec, int n, const char label[]);
double* Conjugate_gradient(double* A, double* b, int n, int* k, double max_tol, int max_iter);
double Vec_norm_square(double *vec, int n);
double Dot_product(double *a, double *b, int n);
void Daxpy(double *y, double *x, double a, int n, double* ret_vec);


#define RAND_MIN 0.0
#define RAND_MAX_D 100.0

int main(int argc, char** argv) {
  int n, max_iter, k;
  double max_tol, *matrix, *b, *x, *solution;
  srandom(random());
  printf("Enter tolerance: ");
  scanf("%lf", &max_tol);

  printf("Enter max iterations: ");
  scanf("%d", &max_iter);

  printf("Enter num rows and cols: ");
  scanf("%d", &n);

  matrix = malloc(n * n * sizeof(double));
  b = malloc(n * sizeof(double));
  x = malloc(n * sizeof(double));
  Gen_matrix(matrix, n);
  Print_mat(matrix, n, "Matrix A");
  Gen_b_vector(matrix, n, x, b);
  Print_vec(x, n, "Vector x");
  Print_vec(b, n, "Vector b");
  solution = Conjugate_gradient(matrix, b, n, &k, max_tol, max_iter); 
  Print_vec(solution, n, "Solution");
  printf("Iterations: %d\n", k);

  free(matrix);
  free(b);
  free(x);
  return 0;
}

void Gen_matrix(double* matrix, int n) {
  double *rand_mat = malloc(n * n * sizeof(double));

  Gen_rand_matrix(rand_mat, n);

  Multiply_transpose(rand_mat, n, matrix);

  free(rand_mat);
}

void Gen_rand_matrix(double* matrix, int n) {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      matrix[i * n + j] = ((double)random() * (RAND_MAX_D - RAND_MIN)) / (double)RAND_MAX + RAND_MIN;
    }
  }
}

void Multiply_transpose(double *mat, int n, double* ret_mat) {
  int i, j, k;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      ret_mat[i * n + j] = 0.0;
      for (k = 0; k < n; k++) {
        ret_mat[i * n + j] += mat[i * n + k] * mat[j * n + k];
      }
    }
  }

}

void Gen_b_vector(double *A, int n, double *x, double *b) {
  int i;
  for (i = 0; i < n; i++) {
    x[i] = ((double)random() * (RAND_MAX_D - RAND_MIN)) / (double)RAND_MAX + RAND_MIN;
  }

  Mat_vec_prod(A, x, n, n, b); 
}

double* Conjugate_gradient(double* A, double* b, int n, int* k, double max_tol, int max_iter) {
  int i;
  double *x, *r, *p, beta, *r_min_1, *r_min_2, *s, alpha;
  *k = 0;
  x = malloc(max_iter * n * sizeof(double));
  r = malloc(max_iter * n * sizeof(double));
  p = malloc(max_iter * n * sizeof(double));
  s = malloc(n * sizeof(double));
  for (i = 0; i < n; i++) {
    r[i] = b[i];
  }

  x[0] = 0;

  while (Vec_norm_square(&r[(*k) * n], n) > max_tol && *k < max_iter) {
    (*k)++;
    r_min_1 = &r[(*k - 1) * n];
    if (*k == 1) {
      for (i = 0; i < n; i++) {
        p[1 * n + i] = r[0 * n + i];
      }
    } else {
      r_min_2 = &r[(*k - 2) * n];
      beta = Dot_product(r_min_1, r_min_1, n) / Dot_product(r_min_2, r_min_2, n); 
      Daxpy(r_min_1, &p[(*k - 1) * n], beta, n, &p[*k * n]);
    }
    
    Mat_vec_prod(A, &p[*k * n], n, n, s);
    alpha = Dot_product(r_min_1, r_min_1, n) / Dot_product(&p[*k * n], s, n);
    Daxpy(&x[(*k -1) * n], &p[*k * n], alpha, n, &x[*k * n]);
    Daxpy(r_min_1, s, -alpha, n, &r[*k * n]);
  }
  
  free(r);
  free(p);
  free(s);
  return &x[*k * n];
}

double Vec_norm_square(double *vec, int n) {
  int i;
  double res = 0.0;
  for (i = 0; i < n; i++) {
    res += vec[i] * vec[i];
  }

  return res;
}

double Dot_product(double *a, double *b, int n) {
  int i;
  double sum = 0.0;
  for (i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  
  return sum;
}

void Daxpy(double *y, double *x, double a, int n, double* ret_vec) {
  int i;
  for (i = 0; i < n; i++) {
    ret_vec[i] = y[i] + a * x[i];
  }
}

void Mat_vec_prod(double *mat, double *vec, int col, int rows, double *ret_vec) {
  int i, j;
  for (i = 0; i < rows; i++) {
     ret_vec[i] = 0.0;
     for (j = 0; j < col; j++) {
       ret_vec[i] += mat[i * col + j] * vec[j];
     }
  }
}

void Scal_vec_product(double *vec, int n, double scalar, double *ret_vec) {
  int i;
  ret_vec = malloc(n * sizeof(double));
  for (i = 0; i < n; i++) {
    ret_vec[i] = vec[i] * scalar;
  }
}

void Print_mat(double *mat, int n, const char label[]) {
  int i, j;
  printf("%s:\n", label);
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      printf("%lf,  ", mat[i * n + j]);
    }
    printf("\n");
  }
}

void Print_vec(double* vec, int n, const char label[]) {
  int i;
  printf("%s:\n[", label);
  for (i = 0; i < n; i++) {
    printf("%lf, ", vec[i]);
  }
  printf("]\n");
}
