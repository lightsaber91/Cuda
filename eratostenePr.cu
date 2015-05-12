#include <stdio.h>
#include <math.h>

#define THREADS 256
#define MINARG 2
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        exit( EXIT_FAILURE );
    }
}

void startTimer(cudaEvent_t *start, cudaEvent_t *stop) {
	HANDLE_ERROR( cudaEventCreate(start));
	HANDLE_ERROR( cudaEventCreate(stop));
	HANDLE_ERROR( cudaEventRecord(*start, 0));
}

void stopAndPrint(cudaEvent_t *start, cudaEvent_t *stop) {
	HANDLE_ERROR( cudaEventRecord(*stop, 0));
	HANDLE_ERROR( cudaEventSynchronize(*stop));
	float time=0;
	HANDLE_ERROR( cudaEventElapsedTime(&time, *start, *stop));
	printf("Elapsed Time: %f milliseconds\n", time);
	HANDLE_ERROR( cudaEventDestroy(*start));
	HANDLE_ERROR( cudaEventDestroy(*stop));
}

void print(int *array, int size){
    int i = 0;
    int c = 0;
    for (i=0;i<size;i++){
        if (array[i]) {
            printf("%d\n", array[i]);
            c++;
        }
    }
    printf("Total number of primes: %d\n", c);
}

__global__ void eliminateMultiples(int *list, int end, int *next, int fine) {
    __shared__ unsigned int block_next;
    block_next = *next;
    unsigned long start, i;
    do {
        start = (unsigned long) block_next*(threadIdx.x + 2 + blockIdx.x * blockDim.x) - 1;
        for(i = start; i < end; i += (unsigned long) block_next*blockDim.x*gridDim.x) {
            //elimino i multipli
            list[i] = 0;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            unsigned int j;
            bool found = false;
            //cambio il next
            if(block_next == 2) {
                j = block_next;
            }
            else
                j = block_next + 1;
            for(; j < end && found == false; j+=2) {
                if(list[j] > block_next) {
                    block_next = list[j];
                    found = true;
                }
            }
        }
        __syncthreads();
    }while(block_next < fine);
}

void findAllPrimeNumbers(int N){
	//Definisco il numero di blocchi
    if(N%2) {
        N+=1;
    }
    int blocks = (((N-2)/2)+(THREADS-1))/THREADS;
    printf("Number of threads: %d, Number of blocks: %d\n",THREADS,blocks);
    //Variabili GPU
    int *dev_list, *dev_next;
    //Variabili CPU
    int *list = new int[N];
    int next = 2;
    for(int i=0; i<N; i++) {
        list[i]=i+1;
    }
    int fine = (int) (sqrt(N)+0.5);
    //Timer
    cudaEvent_t start,stop;
    //Allocazione su GPU
    //cudaMalloc((void**)&dev_end, sizeof(int));
    cudaMalloc((void**)&dev_next,sizeof(int));
    cudaMalloc((void**)&dev_list,sizeof(int)*N);
    //Copia dati sulla GPU
    cudaMemcpy(dev_list,list, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_next, &next, sizeof(int), cudaMemcpyHostToDevice);
    //Inizializzazione del Timer
    startTimer(&start,&stop);
    //Lancio del Kernel
    eliminateMultiples<<<blocks,THREADS>>>(dev_list, N, dev_next, fine);
    cudaDeviceSynchronize();
    //Fine del timer
    stopAndPrint(&start,&stop);
    //Ricopio il risultato sulla GPU
    cudaMemcpy(list, dev_list, sizeof(int)*N, cudaMemcpyDeviceToHost);
    //Libero Memoria
    cudaFree(dev_list);
    cudaFree(dev_next);
    //Stampo informazioni
    print(list, N);
    delete[] list;
}

int main(int argc, char *argv[]) {
	if(argc<MINARG) {
		fprintf(stderr,"Usage: %s N\n",argv[0]);
		exit(-1);
	}
    int N = atoi(argv[1]);
    if(N<0) {
		fprintf(stderr,"Invalid number: %d must be > 0\n",N);
		exit(-1);
	}
    printf("Prime numbers from 0 to %d:\n",N);
    findAllPrimeNumbers(N);
    return 0;
}
