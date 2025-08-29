//DEMO program that shows how to compute unique ID for threads in a 2-dim grid and thread block
#include <stdio.h>
#include <unistd.h>

__global__ void hello( )
{
   printf("blockIdx:(%d,%d), threadIdx:(%d,%d) -> Row,Col=(%d,%d)\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
           blockIdx.x * blockDim.x + threadIdx.x,   // rowID    
           blockIdx.y * blockDim.y + threadIdx.y);  // columnID 
}

int main()
{
   dim3 blockShape = dim3( 2, 3 );
   dim3 gridShape  = dim3( 3, 2 );

   hello<<< gridShape, blockShape>>>( );

   printf("I am the CPU: Hello World ! \n");
   cudaDeviceSynchronize();
} 