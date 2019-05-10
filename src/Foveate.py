#install PyCUDA
#https://pypi.python.org/pypi/pycuda
#follow the instructions, make sure that CUDA_ROOT/bin/nvcc is on the path (export PATH in ~/.bashrc)

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
cuda.init()
from pycuda.compiler import SourceModule
# import pycuda.autoinit
# from pycuda.autoinit import context

import time
import scipy.io as sio
import numpy as np
import math
import cv2

class Foveate:
    def __init__(self, dotPitch, viewDist, rodsAndCones):
        self.CTO = 1/64 #constant from Geisler & Perry
        self.alpha = 0.106  #constant from Geisler & Perry
        self.epsilon2 = 2.3 #constant from Geisler & Perry

        self.dotPitch = dotPitch
        self.viewDist = viewDist
        self.rodsAndCones = rodsAndCones

        self.origW = -1
        self.origH = -1

        self.img = None
        self.imgFov = None

        self.pyramid_d = None
        self.pyrlevel_d = None
        self.fov_d = None

        self.device = cuda.Device(0)
        self.context = self.device.make_context()
        self.loadKernel()
        self.context.pop()

    def setImage(self, img):
        self.img = img.copy()

    def allocate_GPU_mem(self):
        self.pyramid_d = cuda.mem_alloc(self.pyramid.nbytes)
        self.pyrlevel_d = cuda.mem_alloc(self.pyrlevelCones.nbytes)
        self.fov_d = cuda.mem_alloc(self.pyrlevelCones.nbytes)


    def init(self):
        self.origH = self.img.shape[0]
        self.origW = self.img.shape[1]

        self.numLevels = min(7, math.floor(math.log2(max(self.origH, self.origW))))

        self.imgFov = np.empty_like(self.img)

        self.pyrlevelCones = None
        self.pyrlevelRods = None
        self.fovCones = None
        self.fovRods = None

        self.pyramid = np.zeros((self.numLevels, self.origH, self.origW), dtype=np.float32)

        x, step = np.linspace(0, self.origH-1, num=self.origH, retstep=True, dtype=np.float32)
        y, step = np.linspace(0, self.origW-1, num=self.origW, retstep=True, dtype=np.float32)

        self.ix, self.iy = np.meshgrid(y, x, sparse=False, indexing='xy')

        self.ec = None


    def loadKernel(self):

        self.kernel = SourceModule("""

        #include <stdio.h>
        __device__ inline int access_(int M, int N, int O, int x, int y, int z) {
          if (x<0) x=0; else if (x>=M) x=M-1;
          if (y<0) y=0; else if (y>=N) y=N-1;
          if (z<0) z=0; else if (z>=O) z=O-1;
          //return y + M*(x + N*z);
          return z*M*N + x*N + y;
        }

        __device__ int access_unchecked_(int M, int N, int O, int x, int y, int z) {
          //return y + M*(x + N*z);
          return z*M*N + x*N + y;
        }


        __global__ void interpolate_bicubic_GPU(float *pO, float *pF,
        	float *pZ, const int M, const int N,  const int O) {

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

            const float dxx = dx*dx;
            const float dxxx = dxx*dx;

            const float dyy = dy*dy;
            const float dyyy = dyy*dy;

            const float dzz = dz*dz;
            const float dzzz = dzz*dz;


            const float wx0 = 0.5f * (    - dx + 2.0f*dxx -       dxxx);
            const float wx1 = 0.5f * (2.0f      - 5.0f*dxx + 3.0f * dxxx);
            const float wx2 = 0.5f * (      dx + 4.0f*dxx - 3.0f * dxxx);
            const float wx3 = 0.5f * (         -     dxx +       dxxx);

            const float wy0 = 0.5f * (    - dy + 2.0f*dyy -       dyyy);
            const float wy1 = 0.5f * (2.0f      - 5.0f*dyy + 3.0f * dyyy);
            const float wy2 = 0.5f * (      dy + 4.0f*dyy - 3.0f * dyyy);
            const float wy3 = 0.5f * (         -     dyy +       dyyy);

            const float wz0 = 0.5f * (    - dz + 2.0f*dzz -       dzzz);
            const float wz1 = 0.5f * (2.0f      - 5.0f*dzz + 3.0f * dzzz);
            const float wz2 = 0.5f * (      dz + 4.0f*dzz - 3.0f * dzzz);
            const float wz3 = 0.5f * (         -     dzz +       dzzz);

            __shared__ int f_i[64];

            //indices_cubic_(f_i, int(x_floor-1), int(y_floor-1), int(z_floor-1), M, N, O);

        	int x_ = x_floor-1;
        	int y_ = y_floor-1;
        	int z_ = z_floor-1;

        	  if (x_<=2 || y_<=2 || z_<=2 || x_>=N-3 || y_>=M-3 || z_>=O-3) {
        	    //for (int i=0; i<4; ++i)
        	    //  for (int j=0; j<4; ++j)
        	    //    for (int k=0; k<4; ++k)

        	    //f_i[i+4*(j+4*k)] = access_(M,N,O, x+i-1, y+j-1, z+k-1);

        	  	f_i[threadId] = access_(M, N, O, x_+threadIdx.x-1, y_+threadIdx.y-1, z_+threadIdx.z-1);

        	  } else {
        	    //for (int i=0; i<4; ++i)
        	    //  for (int j=0; j<4; ++j)
        	    //    for (int k=0; k<4; ++k)
        	    //      f_i[i+4*(j+4*k)] = access_unchecked_(M,N,O, x+i-1, y+j-1, z+k-1);
        	  	f_i[threadId] = access_unchecked_(M, N, O, x_+threadIdx.x-1, y_+threadIdx.y-1, z_+threadIdx.z-1);
        	  }

        	__syncthreads();

            //if (threadId == 0 && blockIdx.x == 300 && blockIdx.y == 15) {
            //    printf("blockID=(%i %i) %i %i %i M=%i N=%i O=%i pF[%i]=%f\\n", blockIdx.x, blockIdx.y, x_, y_, z_, M, N, O, blockId, pF[blockId]);
            //    for (int k = 0; k < 64; k++)
            //        printf("%0.03f ", pF[k]);
            //    printf("\\n");
            //}

        	if (threadId == 0) {

        	pO[blockId] =
        	wz0*(
        		wy0*(wx0 * pF[f_i[0]] + wx1 * pF[f_i[1]] +  wx2 * pF[f_i[2]] + wx3 * pF[f_i[3]]) +
        		wy1*(wx0 * pF[f_i[4]] + wx1 * pF[f_i[5]] +  wx2 * pF[f_i[6]] + wx3 * pF[f_i[7]]) +
        		wy2*(wx0 * pF[f_i[8]] + wx1 * pF[f_i[9]] +  wx2 * pF[f_i[10]] + wx3 * pF[f_i[11]]) +
        		wy3*(wx0 * pF[f_i[12]] + wx1 * pF[f_i[13]] +  wx2 * pF[f_i[14]] + wx3 * pF[f_i[15]])
        		) +
        	wz1*(
        		wy0*(wx0 * pF[f_i[16]] + wx1 * pF[f_i[17]] +  wx2 * pF[f_i[18]] + wx3 * pF[f_i[19]]) +
        		wy1*(wx0 * pF[f_i[20]] + wx1 * pF[f_i[21]] +  wx2 * pF[f_i[22]] + wx3 * pF[f_i[23]]) +
        		wy2*(wx0 * pF[f_i[24]] + wx1 * pF[f_i[25]] +  wx2 * pF[f_i[26]] + wx3 * pF[f_i[27]]) +
        		wy3*(wx0 * pF[f_i[28]] + wx1 * pF[f_i[29]] +  wx2 * pF[f_i[30]] + wx3 * pF[f_i[31]])
        		) +
        	wz2*(
        		wy0*(wx0 * pF[f_i[32]] + wx1 * pF[f_i[33]] +  wx2 * pF[f_i[34]] + wx3 * pF[f_i[35]]) +
        		wy1*(wx0 * pF[f_i[36]] + wx1 * pF[f_i[37]] +  wx2 * pF[f_i[38]] + wx3 * pF[f_i[39]]) +
        		wy2*(wx0 * pF[f_i[40]] + wx1 * pF[f_i[41]] +  wx2 * pF[f_i[42]] + wx3 * pF[f_i[43]]) +
        		wy3*(wx0 * pF[f_i[44]] + wx1 * pF[f_i[45]] +  wx2 * pF[f_i[46]] + wx3 * pF[f_i[47]])
        		) +
        	wz3*(
        		wy0*(wx0 * pF[f_i[48]] + wx1 * pF[f_i[49]] +  wx2 * pF[f_i[50]] + wx3 * pF[f_i[51]]) +
        		wy1*(wx0 * pF[f_i[52]] + wx1 * pF[f_i[53]] +  wx2 * pF[f_i[54]] + wx3 * pF[f_i[55]]) +
        		wy2*(wx0 * pF[f_i[56]] + wx1 * pF[f_i[57]] +  wx2 * pF[f_i[58]] + wx3 * pF[f_i[59]]) +
        		wy3*(wx0 * pF[f_i[60]] + wx1 * pF[f_i[61]] +  wx2 * pF[f_i[62]] + wx3 * pF[f_i[63]])
        		);
                //if (blockIdx.x == 300 && blockIdx.y == 15)
                //    printf("blockID=(%i %i) %f\\n", blockIdx.x, blockIdx.y, pO[blockId]);
            }
        }


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

            //__shared__ int f_i[8];

            int f000_i, f100_i, f010_i, f110_i;
            int f001_i, f101_i, f011_i, f111_i;


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

        """)


    def preprocess(self, gazePos):

        #eradius is the radial distance between each point and the point of gaze in meters.
        distPx = np.sqrt(np.power(self.ix-gazePos[1], 2) + np.power(self.iy-gazePos[0], 2))
        eradius = distPx*self.dotPitch

        #ec - eccentricity from the fovea center for each pixel in degrees
        ec = 180*np.arctan(eradius/self.viewDist)/math.pi

        eyefreqCones = self.epsilon2/(self.alpha*(ec + self.epsilon2))*math.log(1/self.CTO)
        eyefreqCones = np.power(eyefreqCones, 0.3)

        maxVal = np.amax(eyefreqCones)
        minVal = np.amin(eyefreqCones)
        eyefreqCones = (eyefreqCones-minVal)/(maxVal-minVal)

        #pyrlevel is a fractional level of the pyramid which must be used at each pixel
        #in order to match the foveal resolution function defined above
        eyefreqCones = 1 - eyefreqCones
        self.pyrlevelCones = (self.numLevels-1)*eyefreqCones

        #constrain pyrlevel to conform to the levels of the pyramid which have been computed
        self.pyrlevelCones = np.maximum(0, np.minimum(self.numLevels, self.pyrlevelCones))

        if self.rodsAndCones:
            #this is a bit ad hoc
            #the function was fitted manually in Matlab to resemble distribution of rods in the retina
            #likely we will need something better for the next version of STAR
            p = np.array([8.8814e-11,  -1.6852e-07,   1.1048e-04,  -3.1856e-02,   3.7501e+00,  -3.0283e+00])
            eyefreqRods = self.evalpoly(distPx, p, 5)

            maxVal = np.amax(eyefreqRods)
            eyefreqRods = eyefreqRods/maxVal
            eyefreqRods = 1 - eyefreqRods
            self.pyrlevelRods = np.maximum(0, np.minimum(self.numLevels, self.numLevels*eyefreqRods+2))

    #for color images compute transform for each channel separately
    #and combine them for the final result
    def interpolate(self):
        #since rods affect only intensity, convert image to YCrCb colorspace
        if self.rodsAndCones:
            #this cv2 colorspace conversion only works on uint8 images
            img = cv2.cvtColor(self.img.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
        else:
            img = self.img;

        for c in range(img.shape[2]):
            self.computeImagePyramid(img[:, :, c].astype(np.float32))

            #compute rods only for the first channel (intensity)
            if self.rodsAndCones and c == 0:
                fovRods = self.interp3(self.pyrlevelRods)
                fovCones = self.interp3(self.pyrlevelCones)
                self.imgFov[:, :, c] = fovCones*0.7 + fovRods*0.3 #mixing proportions are approximate, see JEMR paper for justification
            else:
                fovCones = self.interp3(self.pyrlevelCones)
                self.imgFov[:, :, c] = fovCones

        #convert back to BGR
        if self.rodsAndCones:
            self.imgFov = cv2.cvtColor(self.imgFov.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
            self.imgFov = self.imgFov.astype(np.float32)

        cv2.normalize(self.imgFov, self.imgFov, 0, 1, cv2.NORM_MINMAX)


    def foveate(self, img, gazePos):

        self.setImage(img)
        if self.origH != img.shape[0] or self.origW != img.shape[1]:
            self.init()
            self.preprocess(gazePos)
            self.context.push()
            self.allocate_GPU_mem()
            self.context.pop()

        self.context.push()
        self.interpolate()
        self.context.pop()


    def computeImagePyramid(self, img):
        self.pyramid[0, :, :] = img.copy()
        tmp = img.copy()

        for i in range(1, self.numLevels):
            tmp1 = cv2.pyrDown(tmp, dstsize=(round(tmp.shape[1]/2), round(tmp.shape[0]/2)), borderType=cv2.BORDER_DEFAULT )
            self.pyramid[i, :, :] = cv2.resize(tmp1, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
            tmp = tmp1.copy()

    def interp3(self, pyrlevel, method='bicubic'):
        #free, total = cuda.mem_get_info()
        #print(str((free/total)*100) + '% of device memory is free ' + str(free) + ' ' + str(total))

        fov = np.empty_like(pyrlevel)

        cuda.memcpy_htod(self.pyramid_d, self.pyramid)
        cuda.memcpy_htod(self.pyrlevel_d, pyrlevel)

        func = self.kernel.get_function('interpolate_'+method+'_GPU')


        block = (4, 4, 4)
        grid = (self.img.shape[0], self.img.shape[1])

        func(self.fov_d, self.pyramid_d, self.pyrlevel_d, np.int32(self.img.shape[0]), np.int32(self.img.shape[1]), np.int32(self.numLevels), block=block, grid=grid)
        cuda.memcpy_dtoh(fov, self.fov_d)

        return fov

    # using the numpy built-in polyval instead
    #use Horner's method to evaluate polynomial
    def evalpoly(self, x, p, nc):
        y = np.zeros(x.shape) + p[0]
        for i in range(1, nc+1):
            np.multiply(x, y, out=y)
            y += p[i]
        return y.astype(np.float32)
