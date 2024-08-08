#include <stdio.h>
#include <mpi.h>
void main(int argc,char * argv[]){
	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	if(rank==0){
		int N,sum2=0,sum1,sum;
		printf("Enter the length of the array:");
		scanf("%d",&N);
		int arr[N];
		printf("Enter the array elements:");
		for(int i=0;i<N;i++){
			scanf("%d",&arr[i]);
		 }
		 MPI_Send(&N,1,MPI_INT,1,0,MPI_COMM_WORLD);
		 MPI_Send(&arr[N/2],N/2,MPI_INT,1,1,MPI_COMM_WORLD);
		 for(int i=0;i<N/2;i++) sum2+=arr[i];
		 MPI_Recv(&sum1,1,MPI_INT,1,2,MPI_COMM_WORLD,&status);
		 printf("Sum of second half array received by rank %d\n",rank);
		 sum=sum1+sum2;
		 printf("Sum=%d\n",sum);
		 
	}
	else if(rank==1){
		int N,sum1=0;
		MPI_Recv(&N,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		int B[N];
		MPI_Recv(&B[0],N/2,MPI_INT,0,1,MPI_COMM_WORLD,&status);
		printf("Array received by rank %d\n",rank);
		for(int i=0;i<N/2;i++) sum1+=B[i];
		MPI_Send(&sum1,1,MPI_INT,0,2,MPI_COMM_WORLD);
	}
	MPI_Finalize();
 
}
