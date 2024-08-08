#include <stdio.h>
#include<string.h>
#include <mpi.h>
#include <ctype.h>

void toggle_case(char *str){
	while(*str){
		if(islower(*str)){
			*str=toupper(*str);
		}else if(isupper(*str)){
			*str=tolower(*str);
		}
		str++;
	}
}
void main(int argc,char *argv[]){
	int rank,size;
	char msg[1000];
	MPI_Init(&argc,&argv);
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	if(rank==0){
		printf("Enter a message:");
		scanf("%s",msg);
		int len=strlen(msg);
		len=len++;
		MPI_Ssend(&len,1,MPI_INT,1,0,MPI_COMM_WORLD);
		MPI_Ssend(msg,len,MPI_CHAR,1,1,MPI_COMM_WORLD);
		MPI_Recv(&len,1,MPI_INT,1,2,MPI_COMM_WORLD,&status);
		printf("Length received by rank %d\n",rank);
		MPI_Recv(msg,len,MPI_CHAR,1,3,MPI_COMM_WORLD,&status);
		printf("Message received by rank %d\n",rank);
		printf("Message =%s\n",msg);
	}
	else if(rank==1){
		int len1;
		char msg1[len1];
		MPI_Recv(&len1,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		printf("Length received by rank %d\n",rank);
		MPI_Recv(msg1,len1,MPI_CHAR,0,1,MPI_COMM_WORLD,&status);
		printf("Message received by rank %d\n",rank);
		printf("Message =%s\n",msg1);
		toggle_case(msg1);
		MPI_Ssend(&len1,1,MPI_INT,0,2,MPI_COMM_WORLD);
		MPI_Ssend(msg1,len1,MPI_CHAR,0,3,MPI_COMM_WORLD);
	}
	MPI_Finalize();
}
