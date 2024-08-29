#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main (int argc, char *argv[]){
	int size,rank;
	int sendbuf, recvbuf,sum=0;
	double t1,t2;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	t1=MPI_Wtime();
	sendbuf=rank+1;
	MPI_Scan(&sendbuf,&recvbuf,1,MPI_INT,MPI_PROD,MPI_COMM_WORLD);
	t2=MPI_Wtime();
	printf("Process %d number=%d factorial=%d, time taken=%1.2f\n",rank,sendbuf,recvbuf,t2-t1);
	MPI_Reduce(&recvbuf,&sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	if (rank==0){
		printf("result=%d",sum);
	}
	MPI_Finalize();
	return 0;
}
/*
Process 0 number=1 factorial=1, time taken=0.00
result=4037913Process 1 number=2 factorial=2, time taken=0.00
Process 2 number=3 factorial=6, time taken=0.00
Process 3 number=4 factorial=24, time taken=0.00
Process 4 number=5 factorial=120, time taken=0.00
Process 5 number=6 factorial=720, time taken=0.00
Process 7 number=8 factorial=40320, time taken=0.00
Process 8 number=9 factorial=362880, time taken=0.00
Process 9 number=10 factorial=3628800, time taken=0.00
Process 6 number=7 factorial=5040, time taken=0.00
*/
