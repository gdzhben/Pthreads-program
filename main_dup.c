#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <cblas.h>
#include <omp.h>

struct timeval tv1, tv2;
struct timezone tz;

double multiply_matrix_get_norm(double *A, double *B, double *result, int n, int no_col);

int main(int argc, char **argv)
{
    int n;
    printf("Matrix size n \t No.Of.Threads \t Time\n");
    for (n = 100; n < 1000; n += 50) {
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *result = (double *)calloc(n * n, sizeof(double));

    int i, j;
    for (i = 0; i < n * n; i++)
    {
        A[i] = (double)(rand() % 1000);
        B[i] = (double)(rand() % 1000);
    }


    int no_of_threads;
    double norm = 0.0;
    for (no_of_threads = 1; no_of_threads < n; no_of_threads++)
    {
        omp_set_num_threads(no_of_threads);

        gettimeofday(&tv1, &tz);

#pragma omp parallel for
        for (i = 0; i < no_of_threads; i++)
        {

            int partition = n / no_of_threads;
            int col_no = i * partition;

            if (i == no_of_threads - 1)
            {
                partition = n - (i * partition);
            }


            double sum = multiply_matrix_get_norm(A, &B[col_no * n],
                                                  &result[col_no * n], n, col_no);
#pragma omp critical
{
            if (norm < sum)
            {
                norm = sum;
            }
        }

      }
      gettimeofday(&tv2, &tz);
      double time = (double)(tv2.tv_sec - tv1.tv_sec) + (double)(tv2.tv_usec - tv1.tv_usec) * 1.e-6;
      printf("%d \t\t %d\t \t %f \n", n, no_of_threads, time);

    }



}
    return (EXIT_SUCCESS);
}

double multiply_matrix_get_norm(double *A, double *B, double *result, int n, int no_col)
{

    double first = 1.0, second = 0.0;
    ATL_dgemm(CblasNoTrans, CblasNoTrans,
              n, no_col, n, first,
              A, n,
              B, n,
              second,
              result, n);

    int i, j;
    double max_sum = 0;
    for (i = 0; i < no_col; i++)
    {
        double sum = 0;
        for (j = 0; j < n; j++)
        {
            sum += fabs(result[n * i + j]);
        }
        if (max_sum < sum)
        {
            max_sum = sum;
        }
    }

    return max_sum;
}
