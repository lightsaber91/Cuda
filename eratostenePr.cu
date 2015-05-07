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
	printf("Elapsed Time: %f\n", time);
	HANDLE_ERROR( cudaEventDestroy(*start));
	HANDLE_ERROR( cudaEventDestroy(*stop));
}

void print(int *array, int size){
    int i =0;
    int c = 0;
    for (i=0;i<size;i++){
        if (array[i]) {
            printf("%d\n", i+1);
            c++;
        }
    }
    printf("Total number of primes: %d\n", c);
}

__global__ void eliminateMultiples(int *list, int end, int next, int fine) {
}

void findAllPrimeNumbers(int N){
	if(N%2) {
        N+=1;
    }
    int blocks = (((N-2)/2)+(THREADS-1))/THREADS;
    int *dev_list, *dev_next;
    int *list = new int[N];
    int next;
	int i;
    cudaEvent_t start,stop;
	printf("Number of threads: %d, Number of blocks: %d\n",THREADS,blocks);
    //cudaMalloc((void**)&dev_end, sizeof(int));
    cudaMalloc((void**)&dev_next,sizeof(int));
    cudaMalloc((void**)&dev_list,sizeof(int)*N);
	for(i=0; i<N; i++) {
		list[i]=1;
	}
    cudaMemcpy(dev_list,list, sizeof(int)*N, cudaMemcpyHostToDevice);
    next=2;
    cudaMemcpy(dev_next, &next, sizeof(int), cudaMemcpyHostToDevice);
    int fine = (int) (sqrt(N)+0.5);
    startTimer(&start,&stop);
    eliminateMultiples<<<blocks,THREADS>>>(dev_list, N, next, fine);
    cudaDeviceSynchronize();
    stopAndPrint(&start,&stop);
    cudaMemcpy(list, dev_list, sizeof(int)*N, cudaMemcpyDeviceToHost);
    cudaFree(dev_list);
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
