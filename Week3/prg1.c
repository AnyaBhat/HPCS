#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
int fact(int n){
	int fac=1;
	if (n==0 || n==1) return fac;
	if(n==2)return 2;
	for(int i=2; i<=n;i++){
		fac*=i;
	}
	return fac;
}
int main(int argc, char *argv[]){
	int rank,size,a,b,res;
	int num;
	
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	int *array=(int*)malloc(size*sizeof(int));
	if(rank==0){
		printf("Enter %d  elements:",size);
		for(int i=0;i<size;i++){
			scanf("%d",&array[i]);
		}
	}
	MPI_Scatter(&array[0],1,MPI_INT,&num,1,MPI_INT,0,MPI_COMM_WORLD);
	int result=fact(num);
	printf("Factorial of %d is %d,rank %d \n",num,result,rank);
	MPI_Finalize();
	return 0;
}
	
