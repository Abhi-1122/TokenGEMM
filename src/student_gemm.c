#include "student_gemm.h"
#include <stdio.h>
void student_gemm(const float *A, const float *B, float *C, int M, int K, int N)
{
    (void)A;
    (void)B;
    (void)K;

    for (int i = 0; i < M; i++)
    {
        int j_end = 0;
        int j_start = 0;
        while (j_end != N)
        {
            float temp0[8] = {0};
            float temp1[8] = {0};
            float temp2[8] = {0};
            float temp3[8] = {0};
            j_start = j_end;
            j_end = (j_end + 8) < N ? j_end + 8 : N;
            for (int k = 0; k < K; k+=4)
            {
                float a0 = A[i * K + k];
                float a1 = A[i * K + k + 1];
                float a2 = A[i * K + k + 2];
                float a3 = A[i * K + k + 3];
                for (int j = j_start; j < j_end; j++)
                {
                    temp0[j- j_start] += a0 * B[k * N + j];
                    temp1[j- j_start] += a1 * B[(k + 1) * N + j];
                    temp2[j- j_start] += a2 * B[(k + 2) * N + j];
                    temp3[j- j_start] += a3 * B[(k + 3) * N + j];
                }
            }
            for (int j = j_start; j < j_end; j++)
                C[i * N + j] = temp0[j - j_start] + temp1[j - j_start] + temp2[j - j_start] + temp3[j - j_start];
        }
    }
}
