#include <stdio.h>
#include <omp.h>
void main()
{
int a[24]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
int b[24]={31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54};
int c[24];
int tid;
#pragma omp parallel 
{
tid=omp_get_thread_num();
c[tid]=a[tid]+b[tid];
printf("c[%d]=%d for thread %d\n",tid,c[tid],tid);
}
}
