#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

__global__ void deci_oct ( int *ip, int *op,int len){
	int tid=threadIdx.x;
	if(tid <len){
		int decNum = ip[tid];
        	int octNum = 0;
        	int factor = 1;

        	while (decNum > 0) {
            		int remainder = decNum % 8;
            		octNum += remainder * factor;
            		factor *= 10;
            		decNum /= 8;
        	}
        	op[tid] = octNum;
	}
}

int main(){
	int N = 5;  // Size of arrays
    	int size = N * sizeof(int);

    	int ip[N]={9,8,33,40,54};
    	int op[N];

    	int *d_ip, *d_op;
	
	cudaMalloc((void **)&d_ip,size);
	cudaMalloc((void **)&d_op,size);
	
	cudaMemcpy(d_ip, ip, size, cudaMemcpyHostToDevice);
	
	
	deci_oct<<<1,N>>>(d_ip,d_op,N);
	
	cudaMemcpy(op,d_op,size,cudaMemcpyDeviceToHost);
	
	for(int i=0;i<N;i++){
		printf("octal(%d) = %d \n",ip[i],op[i]);
	}
	cudaFree(d_ip);
	cudaFree(d_op);
	return 0;
}
