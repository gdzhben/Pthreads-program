/*
 * File:   main.c
 * Author: Ben
 *
 * Created on 06 November 2017, 21:20
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <cblas.h>

struct timeval tv1, tv2;
struct timezone tz;

typedef struct {
    double *A;
    double *B;
    double *result;
    double *max_value;
    pthread_mutex_t *mutex;
    int m;
    int n;
    int k;
} multiply_matrix_blas;

void *multiply_matrix(void *arg);
double calculate_norm(int no_of_threads, int n, double *A, double *B, double *result);

int main(int argc, char **argv) {
    int n = 0;
    int no_of_thread = 0;
    
    printf("\nsize of matrix\tnumber of threads\ttime\n");
    
    for (n = 100; n < 2000; n += 100) {
        for (no_of_thread = 1; no_of_thread <= n; no_of_thread += no_of_thread) {

            //allocating space for array A B and result.
            double *A = (double *) malloc(n * n * sizeof (double));
            double *B = (double *) malloc(n * n * sizeof (double));
            double *result = (double *) malloc(n * n * sizeof (double));

            //fill in random values for array A and B
            int i;
            for (i = 0; i < n * n; i++) {
                A[i] = (double) (rand() % 1000);
                B[i] = (double) (rand() % 1000);
                result[i] = 0.0;
            }
            //timer start
            gettimeofday(&tv1, &tz);
            double norm = calculate_norm(no_of_thread, n, A, B, result);
            //stop timer
            gettimeofday(&tv2, &tz);
            double time = (double) (tv2.tv_sec - tv1.tv_sec) + (double) (tv2.tv_usec - tv1.tv_usec) * 1.e-6;
            printf("%d\t\t\t%d\t\t\t%f\t%f\n", n, no_of_thread, time, norm);
            free(A);
            free(B);
            free(result);
        }
    }
    return (EXIT_SUCCESS);
}

double calculate_norm(int no_of_threads, int n, double *A, double *B, double *result) {
    int i;
    void *status;
    pthread_t *pthreads;
    multiply_matrix_blas *multiply_matrix_arr;
    double norm_value = 0.0;
    pthread_mutex_t *mutex;

    if (no_of_threads > n) {
        no_of_threads = n;
    } else if (no_of_threads < 1) {
        no_of_threads = 1;
    }

    mutex = malloc(sizeof (pthread_mutex_t));
    pthreads = malloc(no_of_threads * sizeof (pthread_t));
    multiply_matrix_arr = malloc(no_of_threads * sizeof (multiply_matrix_blas));
    pthread_mutex_init(mutex, NULL);
    for (i = 0; i < no_of_threads; i++) {
        int no_of_cols = n / no_of_threads;
        int col_no = i * no_of_cols;
        if (i == no_of_threads - 1) {
            no_of_cols = n - (i * no_of_cols);
        }

        multiply_matrix_arr[i].A = A;
        multiply_matrix_arr[i].B = &B[col_no];
        multiply_matrix_arr[i].result = &result[col_no];
        multiply_matrix_arr[i].m = n;
        multiply_matrix_arr[i].n = no_of_cols;
        multiply_matrix_arr[i].k = n;
        multiply_matrix_arr[i].max_value = &norm_value;
        multiply_matrix_arr[i].mutex = mutex;

        int success = pthread_create(&pthreads[i], NULL, multiply_matrix,
                (void *) &multiply_matrix_arr[i]);
        if (success) {
            perror("error when creating thread");
            exit(EXIT_FAILURE);
        }
    }
    for (i = 0; i < no_of_threads; i++) {
        int success = pthread_join(pthreads[i], &status);
        if (success) {
            perror("error when joining the thread");
            exit(EXIT_FAILURE);
        }
    }

    return norm_value;
}

void *multiply_matrix(void *arg) {
    multiply_matrix_blas *data = arg;
    double first = 1.0, second = 0.0;

    ATL_dgemm(CblasNoTrans,
            CblasNoTrans,
            data->m, data->n, data->k,
            first,
            data->A, data->k,
            data->B, data->k,
            second,
            data->result,
            data->n);

    int i, j;
    for (i = 0; i < data->n; i++) {
        double sum = 0.0;
        for (j = 0; j < data->k; j++) {
            double va = data->result[data->n * j + i];
            sum += fabs(va);
        }

        pthread_mutex_lock(data->mutex);
        if (*data->max_value < sum) {
            *data->max_value = sum;
        }
        pthread_mutex_unlock(data->mutex);
        pthread_exit(NULL);
    }
}
