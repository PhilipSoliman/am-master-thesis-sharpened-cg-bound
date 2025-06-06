#include <stdio.h>
#include <stdlib.h>
// #include <lapacke.h> #sadly clibs does not support lapacke :/
#include <algorithm>
#include <cmath>

extern "C"
{
   // CG: main function
   bool custom_cg(double *Af, double *b, double *x, double *x_m, double *alpha, double *beta, const int size, int &niters, const int max_iter, const double tol, const bool save_residuals, double *residuals, const bool exact_convergence, double *x_exact, double *errors);

   // CG: supporting functions
   void residual(double **A, double *b, double *x, double *r, const int n);
   void alpha_update(double r_dot_r, double *p, double *Ap, double &alpha, const int n);
   void solution_update(double *x, double *p, double alpha, const int n);
   void residual_update(double *r, double alpha, double *Ap, double *r_new, const int n);
   void beta_update(double r_dot_r, double *r_m, double &beta, const int n);
   void search_direction_update(double *p, double *r, double beta, const int n);

   // solve tridiagonal system
   void solve_tri(const int n, double *x, double *dl, double *d, double *du);
   void solve_tridiagonal_in_place_reusable(const int n, double *x, double *a, double *b, double *c);

   // basic linear algebra functions
   void dot_product(double *a, double *b, double &sum, int n);
   void matrix_vector_product(double **A, double *x, double *prod, const int n);
   void add(double *a, double *b, double *sum, const int n);
   void subtract(double *a, double *b, double *diff, const int n);
   void scalar_multiplication(double *a, double scalar, double *prod, const int n);

   // test function
   void TEST();
}

bool custom_cg(double *Af, double *b, double *x, double *x_m, double *alpha, double *beta, const int size, int &niters, const int max_iter, const double tol, const bool save_residuals, double *residuals, const bool exact_convergence, double *x_exact, double *errors)
{
   // convert Af to A
   double **A = (double **)malloc(size * sizeof(double *));
   for (int i = 0; i < size; i++)
   {
      A[i] = (double *)malloc(size * sizeof(double));
      for (int j = 0; j < size; j++)
      {
         A[i][j] = Af[i * size + j];
      }
   }

   // success flag
   bool success = false;

   // residual vectors
   double *r = (double *)malloc(size * sizeof(double));
   residual(A, b, x, r, size);
   double *r_m = (double *)malloc(size * sizeof(double));

   // search direction
   double *p = (double *)malloc(size * sizeof(double));
   std::copy(r, r + size, p);

   // initial residual
   double r0_dot_r0;
   dot_product(r, r, r0_dot_r0, size);
   double r0_norm = sqrt(r0_dot_r0);

   // initial error
   double e0[size];
   subtract(x_exact, x, e0, size);
   double e0_dot_e0;
   dot_product(e0, e0, e0_dot_e0, size);
   double e0_norm = sqrt(e0_dot_e0);

   int j;
   for (j = 0; j < max_iter; j++)
   {
      // precompute r_dot_r
      double r_dot_r;
      dot_product(r, r, r_dot_r, size);
      if (save_residuals)
      {
         residuals[j] = sqrt(r_dot_r);
      }

      // check for convergence
      if (exact_convergence)
      {
         double em[size];
         subtract(x_exact, x_m, em, size);
         double em_dot_em;
         dot_product(em, em, em_dot_em, size);
         errors[j] = sqrt(em_dot_em);
         double e_ratio = sqrt(em_dot_em) / e0_norm;
         if (e_ratio < tol)
         {
            success = true;
            break;
         }
      }
      else
      {
         double r_ratio = sqrt(r_dot_r) / r0_norm;
         if (r_ratio < tol)
         {
            success = true;
            break;
         }
      }

      double Ap[size];
      matrix_vector_product(A, p, Ap, size);

      double alpha_j;
      alpha_update(r_dot_r, p, Ap, alpha_j, size);
      alpha[j] = alpha_j;

      solution_update(x_m, p, alpha_j, size);

      residual_update(r, alpha_j, Ap, r_m, size);

      double beta_j;
      beta_update(r_dot_r, r_m, beta_j, size);
      beta[j] = beta_j;

      search_direction_update(p, r_m, beta_j, size);

      // update residual
      std::copy(r_m, r_m + size, r);
   }

   // free memory
   for (int i = 0; i < size; i++)
   {
      free(A[i]);
   }
   free(A);
   free(r);
   free(r_m);
   free(p);

   niters = j;
   return success;
}

void residual(double **A, double *b, double *x, double *r, const int n)
{
   matrix_vector_product(A, x, r, n);    // A * x
   scalar_multiplication(r, -1.0, r, n); // -1 * A * x
   add(b, r, r, n);                      // b - A * x
}

void alpha_update(double r_dot_r, double *p, double *Ap, double &alpha, const int n)
{
   double p_dot_Ap;
   dot_product(p, Ap, p_dot_Ap, n);
   alpha = r_dot_r / p_dot_Ap;
}

void solution_update(double *x, double *p, double alpha, const int n)
{
   double alpha_p[n];
   scalar_multiplication(p, alpha, alpha_p, n);
   add(x, alpha_p, x, n);
}

void residual_update(double *r, double alpha, double *Ap, double *r_new, const int n)
{
   double m_alpha_Ap[n];
   scalar_multiplication(Ap, -1.0 * alpha, m_alpha_Ap, n);
   add(r, m_alpha_Ap, r_new, n);
}

void beta_update(double r_dot_r, double *r_m, double &beta, const int n)
{
   double r_m_dot_r_m;
   dot_product(r_m, r_m, r_m_dot_r_m, n);
   beta = r_m_dot_r_m / r_dot_r;
}

void search_direction_update(double *p, double *r, double beta, const int n)
{
   double beta_p[n];
   scalar_multiplication(p, beta, beta_p, n);
   add(r, beta_p, p, n);
}

void solve_tri(const int n, double *x, double *dl, double *d, double *du)
{
   // fordward substitution
   for (int k = 1; k < n; k++)
   {
      double m = dl[k - 1] / d[k - 1];
      if (k < n - 1)
      {
         d[k] = d[k] - m * du[k - 1];
      }
      x[k] = x[k] - m * x[k - 1];
   }

   // backward substitution
   x[n - 1] = x[n - 1] / d[n - 1];
   for (int k = n - 2; k >= 0; k--)
   {
      x[k] = (x[k] - du[k] * x[k + 1]) / d[k];
   }
}

void solve_tridiagonal_in_place_reusable(const int n, double *x, double *a, double *b, double *c)
{
   double cprime[n];

   cprime[0] = c[0] / b[0];
   x[0] = x[0] / b[0];

   /* loop from 1 to X - 1 inclusive */
   for (int ix = 1; ix < n; ix++)
   {
      const float m = 1.0f / (b[ix] - a[ix] * cprime[ix - 1]);
      cprime[ix] = c[ix] * m;
      x[ix] = (x[ix] - a[ix] * x[ix - 1]) * m;
   }

   /* loop from X - 2 to 0 inclusive, safely testing loop end condition */
   for (int ix = n - 1; ix-- > 0;)
      x[ix] -= cprime[ix] * x[ix + 1];
   x[0] -= cprime[0] * x[1];
}

void dot_product(double *a, double *b, double &sum, const int n)
{
   sum = 0.0;
   for (int i = 0; i < n; i++)
   {
      sum += a[i] * b[i];
   }
}

void matrix_vector_product(double **A, double *x, double *prod, const int n)
{
   for (int i = 0; i < n; i++)
   {
      prod[i] = 0.0;
      for (int j = 0; j < n; j++)
      {
         prod[i] += A[i][j] * x[j];
      }
   }
}

void add(double *a, double *b, double *sum, const int n)
{
   for (int i = 0; i < n; i++)
   {
      sum[i] = a[i] + b[i];
   }
}

void subtract(double *a, double *b, double *diff, const int n)
{
   for (int i = 0; i < n; i++)
   {
      diff[i] = a[i] - b[i];
   }
}

void scalar_multiplication(double *a, double scalar, double *prod, const int n)
{
   for (int i = 0; i < n; i++)
   {
      prod[i] = a[i] * scalar;
   }
}

// test function
void TEST()
{
   printf("Hello from C++!\n");

   const int size = 4;
   double **A = (double **)malloc(size * sizeof(double *));
   for (int i = 0; i < size; i++)
   {
      A[i] = (double *)malloc(size * sizeof(double));
      for (int j = 0; j < size; j++)
      {
         if (i == j)
         {
            A[i][i] = 1.0;
         }
         else
         {
            A[i][j] = 0.0;
         }
         printf("A[%d][%d]: %f\n", i, j, A[i][j]);
      }
   }
   double x[size] = {0.0};
   double b[size] = {1.0, 1.0, 1.0, 1.0};
   for (int i = 0; i < size; i++)
   {
      printf("b[%d]: %f\n", i, b[i]);
   }

   printf("Testing reisdual function\n");
   double r[size];
   residual(A, b, x, r, size);
   for (int i = 0; i < size; i++)
   {
      printf("r[%d]: %f\n", i, r[i]);
   }

   printf("Testing alpha_update function\n");
   double p[size];
   std::copy(r, r + size, p);
   double r_dot_r;
   dot_product(r, r, r_dot_r, size);
   double Ap[size];
   matrix_vector_product(A, p, Ap, size);
   double alpha;
   alpha_update(r_dot_r, p, Ap, alpha, size);
   printf("alpha: %f\n", alpha);

   printf("Testing solution_update function\n");
   double x_m[size];
   solution_update(x_m, p, alpha, size);
   for (int i = 0; i < size; i++)
   {
      printf("x_m[%d]: %f\n", i, x_m[i]);
   }

   printf("Testing residual_update function\n");
   double r_new[size];
   residual_update(r, alpha, Ap, r_new, size);
   for (int i = 0; i < size; i++)
   {
      printf("r_new[%d]: %f\n", i, r_new[i]);
   }

   printf("Testing beta_update function\n");
   double r_m[size];
   std::copy(r_new, r_new + size, r_m);
   double beta;
   beta_update(r_dot_r, r_m, beta, size);
   printf("beta: %f\n", beta);

   printf("Testing search_direction_update function\n");
   search_direction_update(p, r_m, beta, size);
   for (int i = 0; i < size; i++)
   {
      printf("p[%d]: %f\n", i, p[i]);
   }
}
