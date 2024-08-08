#include <stdio.h>
#include <mpi.h>
void main(int argc, char *argv[]){
	int rank,size,a,b,res;
	
	
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	if(size<4){
		if(rank ==0){
			printf("Error: At least 4 processes are required.\n");
		}
		MPI_Finalize();
		return ;
	}
	
	if(rank ==0){
		printf("Enter 2 numbers:");
		scanf("%d %d", &a, &b);
	}
	
	MPI_Bcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD);
   	MPI_Bcast(&b, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(rank ==0){
		res=a+b;
		printf("Rank %d, %d+%d=%d\n",rank,a,b,res);
	}
	else if(rank ==1){
		res=a-b;
		printf("Rank %d, %d-%d=%d\n",rank,a,b,res);
	}
	else if(rank ==2){
		res=a*b;
		printf("Rank %d, %d*%d=%d\n",rank,a,b,res);
	}
	else if(rank ==3){
		if (b != 0) {
            		res = a / b;
            		printf("Rank %d: %d / %d = %d\n", rank, a, b, res);
        	} else {
            		printf("Rank %d: Division by zero error.\n", rank);
       		}
	}
	
	MPI_Finalize();
}
