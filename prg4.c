#include <stdio.h>
#include <mpi.h>
void main(int argc,char * argv[]){
	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	if (size != 2) {
        	if (rank == 0) {
            		printf("This program requires exactly 2 processes.\n");
        	}
        	MPI_Finalize();
        	return ;
   	 }
	if(rank==0){
		int N;
		int found=0,found1=0;
		printf("Enter the length of the array:");
		scanf("%d",&N);
		int arr[N];
		printf("Enter the array elements:");
		for(int i=0;i<N;i++){
			scanf("%d",&arr[i]);
		}
		int search;
		printf("Enter element to be searched :");
		scanf("%d",&search);
		MPI_Send(&N,1,MPI_INT,1,0,MPI_COMM_WORLD);
		MPI_Send(&arr[N/2],N/2,MPI_INT,1,1,MPI_COMM_WORLD);
		MPI_Send(&search,1,MPI_INT,1,2,MPI_COMM_WORLD);
		for(int i=0;i<N/2;i++) {
			if(arr[i]==search){
				found=1;
				break;
			}
		}
		MPI_Recv(&found1,1,MPI_INT,1,3,MPI_COMM_WORLD,&status);
		if(found==1||found1==1){
			printf("Element Found\n");
		}
		else{
			printf("Element Not found\n");
		}
		 
	}
	else if(rank==1){
		int N,search,found1=0;
		MPI_Recv(&N,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		int B[N/2];
		MPI_Recv(&B,N/2,MPI_INT,0,1,MPI_COMM_WORLD,&status);
		MPI_Recv(&search,1,MPI_INT,0,2,MPI_COMM_WORLD,&status);
		printf("Array received by rank %d\n",rank);
		for(int i=0;i<N/2;i++) {
			if(B[i]==search){
				found1=1;
				break;
			}
		}
		MPI_Send(&found1,1,MPI_INT,0,3,MPI_COMM_WORLD);
	}
	MPI_Finalize();
 
}
