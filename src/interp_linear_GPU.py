__global__ void interpolate_linear_GPU(float *pO, float *pF,
  float *pZ, const int M, const int N, const int O) {


  int blockId = blockIdx.x*gridDim.y + blockIdx.y;
  //int threadId = threadIdx.x + blockDim.z*(threadIdx.y+blockDim.z*threadIdx.z);
  int threadId = threadIdx.z*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;

  //printf("blockID=(%i %i) threadIdx=(%i %i %i) threadId=%i\\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z, threadId);

  //const float x = pX[blockId];
  //const float y = pY[blockId];
  const float x = blockIdx.x;
  const float y = blockIdx.y;
  const float z = pZ[blockId];

  const float x_floor = floor(x);
  const float y_floor = floor(y);
  const float z_floor = floor(z);

  const float dx = x-x_floor;
  const float dy = y-y_floor;
  const float dz = z-z_floor;

    const float wx0 = 1.0f-dx;
    const float wx1 = dx;

    const float wy0 = 1.0f-dy;
    const float wy1 = dy;

    const float wz0 = 1.0f-dz;
    const float wz1 = dz;

    __shared__ int f_i[8];

    const int f000_i, f100_i, f010_i, f110_i;
    const int f001_i, f101_i, f011_i, f111_i;


    if (x<=1 || y<=1 || z<=1 || x>=M-2 || y>=N-2 || z>=O-2) {
      f000_i = access_(M,N,O, x,   y  , z);
      f100_i = access_(M,N,O, x+1, y  , z);

      f010_i = access_(M,N,O, x,   y+1, z);
      f110_i = access_(M,N,O, x+1, y+1, z);

      f001_i = access_(M,N,O, x,   y  , z+1);
      f101_i = access_(M,N,O, x+1, y  , z+1);

      f011_i = access_(M,N,O, x,   y+1, z+1);
      f111_i = access_(M,N,O, x+1, y+1, z+1);
    } else {
      f000_i = access_unchecked_(M,N,O, x,   y  , z);
      f100_i = access_unchecked_(M,N,O, x+1, y  , z);

      f010_i = access_unchecked_(M,N,O, x,   y+1, z);
      f110_i = access_unchecked_(M,N,O, x+1, y+1, z);

      f001_i = access_unchecked_(M,N,O, x,   y  , z+1);
      f101_i = access_unchecked_(M,N,O, x+1, y  , z+1);

      f011_i = access_unchecked_(M,N,O, x,   y+1, z+1);
      f111_i = access_unchecked_(M,N,O, x+1, y+1, z+1);
    }
    if (threadId == 0) {
      pO[blockId] =
        wz0*(
            wy0*(wx0 * pF[f000_i] + wx1 * pF[f100_i]) +
            wy1*(wx0 * pF[f010_i] + wx1 * pF[f110_i])
            )+
        wz1*(
            wy0*(wx0 * pF[f001_i] + wx1 * pF[f101_i]) +
            wy1*(wx0 * pF[f011_i] + wx1 * pF[f111_i])
            );
    }
  }
}
