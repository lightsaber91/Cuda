#include <stdio.h>
#include <stdlib.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void vec_add(int* A, int* B, int* C, int size) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < size) {
        C[index] = A[index] + B[index];
    }
}

int main(int argc, char *argv[]) {
    if(argc != 3) {
        //Prendo il numero di elementi dell'array e il numero di thread che voglio per ogni blocco
        fprintf(stderr,"Usage: %s <array size> <threads per block>\n", argv[0]);
        return EXIT_FAILURE;
    }
    //Variabili necessarie al calcolo
    int array_size, thread, *a, *b, *c, *d;
    int *gpu_a, *gpu_b, *gpu_c;

    array_size = atoi(argv[1]);
    thread = atoi(argv[2]);
    //Alloco memoria per gli array sulla CPU
    a = (int *) malloc(array_size*sizeof(int));
    b = (int *) malloc(array_size*sizeof(int));
    c = (int *) malloc(array_size*sizeof(int));
    d = (int *) malloc(array_size*sizeof(int)); //questo mi serve per controllo
    //Alloco la memoria sulla GPU
    gpuErrchk(cudaMalloc(&gpu_a, array_size*sizeof(int)));
    gpuErrchk(cudaMalloc(&gpu_b, array_size*sizeof(int)));
    gpuErrchk(cudaMalloc(&gpu_c, array_size*sizeof(int)));
    //Riempio i vettori
    for(int i=0; i<array_size; i++) {
        a[i] = rand();
        b[i] = rand();
        c[i] = 0;
    }
    //Copio i vettori sulla GPU
    gpuErrchk(cudaMemcpy(gpu_a, a, array_size*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_b, b, array_size*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_c, c, array_size*sizeof(int), cudaMemcpyHostToDevice));
    //Faccio in modo che block_size sia sempre intero
    int block_size;
    if(array_size % thread == 0) {
        block_size = array_size/thread;
    }
    else {
        block_size = (array_size/thread) + 1;
    }
    //Eseguo sulla GPU la somma
    vec_add<<<block_size, thread>>>(gpu_a, gpu_b, gpu_c, array_size);
    //Copio il risultato sulla CPU
    gpuErrchk(cudaMemcpy(c, gpu_c, array_size*sizeof(int), cudaMemcpyDeviceToHost));
    //Sommo gli array sulla CPU
    for(int i=0; i<array_size; i++) {
        d[i] = a[i] + b[i];
    }
    //Stampo i vettori
    for(int i=0; i<array_size; i++) {
        if(c[i] == d[i]) {
            printf("Index: %d CPU / GPU = %d / %d\n", i, d[i], c[i]);
        }
        else
            printf("DIFFERENT --> Index: %d CPU / GPU = %d / %d\n", i, d[i], c[i]);
    }
    //Libero la memoria
    free(a); free(b); free(c); free(d);
    cudaFree(gpu_a); cudaFree(gpu_b); cudaFree(gpu_c);
    return EXIT_SUCCESS;
}
