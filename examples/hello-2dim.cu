#include <stdio.h>
#include <unistd.h>

//DEMO program that shows the identifying variables of threads in a 2-dim grid and thread block: 
__global__ void hello( )
{
   printf("grid coord: (%d,%d), thread coord: (%d,%d), grid dim:   (%d,%d), block dim:    (%d,%d)\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
}

int main()
{
   dim3 gridShape  = dim3( 3, 2 );
   dim3 blockShape = dim3( 2, 3 );

   hello<<< gridShape, blockShape>>>( );

   printf("I am the CPU: Hello World ! \n");
   cudaDeviceSynchronize();
} 