/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication with multi GPU support.
 * It has been written for clarity of exposition to illustrate various OpenCL
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11. 
 *
 */

// standard utilities and system includes
//#include <oclUtils.h>
//#include <shrQATest.h>
#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>

// project include
//#include "matrixMul.h"

#define BLOCK_SIZE 16
// max GPU's to manage for multi-GPU parallel compute
const unsigned int MAX_GPU_COUNT = 1;

// Globals for size of matrices
unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
int iSizeMultiple = 1;

// global variables
cl_context cxGPUContext;
cl_kernel multiplicationKernel[MAX_GPU_COUNT];
cl_command_queue commandQueue[MAX_GPU_COUNT];

inline bool
sdkCompareL2fe(const float *reference, const float *data,
               const unsigned int len, const float epsilon)
{
    assert(epsilon >= 0);

    float error = 0;
    float ref = 0;

    for (unsigned int i = 0; i < len; ++i)
    {

        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);

    if (fabs(ref) < 1e-7)
    {
#ifdef _DEBUG
        std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
        return false;
    }

    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
#ifdef _DEBUG

    if (! result)
    {
        std::cerr << "ERROR, l2-norm error "
                  << error << " is greater than epsilon " << epsilon << "\n";
    }

#endif

    return result;
}
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(int argc, const char** argv);
void printDiff(float*, float*, int, int, int, float);
void matrixMulGPU(cl_uint ciDeviceCount, cl_mem h_A, float* h_B_data, unsigned int mem_size_B, float* h_C );
// Helper function to return precision delta time for 3 counters since last call based upon host high performance counter
// *********************************************************************
double shrDeltaT(int iCounterID = 0)
{
    // local var for computation of microseconds since last call
    double DeltaT;
        static struct timeval _NewTime;  // new wall clock time (struct representation in seconds and microseconds)
        static struct timeval _OldTime[3]; // old wall clock timers 0, 1, 2 (struct representation in seconds and microseconds)

        // Get new counter reading
        gettimeofday(&_NewTime, NULL);

		if (iCounterID >= 0 && iCounterID <= 2) 
		{
		    // Calculate time difference for timer (iCounterID).  (zero when called the first time) 
		    DeltaT =  ((double)_NewTime.tv_sec + 1.0e-6 * (double)_NewTime.tv_usec) - ((double)_OldTime[iCounterID].tv_sec + 1.0e-6 * (double)_OldTime[iCounterID].tv_usec);
		    // Reset old timer (iCounterID) to new timer
		    _OldTime[iCounterID].tv_sec  = _NewTime.tv_sec;
		    _OldTime[iCounterID].tv_usec = _NewTime.tv_usec;
		}
		else 
		{
	        // Requested counterID is out of rangewith respect to available counters
	        DeltaT = -9999.0;
		}

	    // Returns time difference in seconds sunce the last call
	    return DeltaT;
} 
//////////////////////////////////////////////////////////////////////////////
//! Gets the platform ID for NVIDIA if available, otherwise default
//!
//! @return the id 
//! @param clSelectedPlatformID         OpenCL platoform ID
//////////////////////////////////////////////////////////////////////////////
cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID)
{
    char chBuffer[1024];
    cl_uint num_platforms; 
    cl_platform_id* clPlatformIDs;
    cl_int ciErrNum;
    *clSelectedPlatformID = NULL;

    // Get OpenCL platform count
    ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS)
    {
        printf(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
        return -1000;
    }
    else 
    {
        if(num_platforms == 0)
        {
            printf("No OpenCL platform found!\n\n");
            return -2000;
        }
        else 
        {
            // if there's a platform or more, make space for ID's
            if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
            {
                printf("Failed to allocate memory for cl_platform ID's!\n\n");
                return -3000;
            }

            // get platform info for each platform and trap the NVIDIA platform if found
            ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
            for(cl_uint i = 0; i < num_platforms; ++i)
            {
                ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                if(ciErrNum == CL_SUCCESS)
                {
                    if(strstr(chBuffer, "NVIDIA") != NULL)
                    {
                        *clSelectedPlatformID = clPlatformIDs[i];
                        break;
                    }
                }
            }

            // default to zeroeth platform if NVIDIA not found
            if(*clSelectedPlatformID == NULL)
            {
                printf("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
                *clSelectedPlatformID = clPlatformIDs[0];
            }

            free(clPlatformIDs);
        }
    }

    return CL_SUCCESS;
}
//////////////////////////////////////////////////////////////////////////////
//! Gets the id of the nth device from the context
//!
//! @return the id or -1 when out of range
//! @param cxGPUContext         OpenCL context
//! @param device_idx            index of the device of interest
//////////////////////////////////////////////////////////////////////////////
cl_device_id oclGetDev(cl_context cxGPUContext, unsigned int nr)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    
    if( szParmDataBytes / sizeof(cl_device_id) <= nr ) {
      return (cl_device_id)-1;
    }
    
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
    
    cl_device_id device = cdDevices[nr];
    free(cdDevices);

    return device;
}
//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename        program filename
//! @param cPreamble        code that is prepended to the loaded file, typically a set of #defines or a header
//! @param szFinalLength    returned length of the code string
//////////////////////////////////////////////////////////////////////////////
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
    // locals 
    FILE* pFileStream = NULL;
    size_t szSourceLength;

    // open the OpenCL source code file
    // Linux version
    pFileStream = fopen(cFilename, "rb");
    if(pFileStream == 0) 
    {       
        return NULL;
    }

    size_t szPreambleLength = strlen(cPreamble);

    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END); 
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET); 

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1); 
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    if(szFinalLength != 0)
    {
        *szFinalLength = szSourceLength + szPreambleLength;
    }
    cSourceString[szSourceLength + szPreambleLength] = '\0';

    return cSourceString;
}
//////////////////////////////////////////////////////////////////////////////
//! Find the path for a file assuming that
//! files are found in the searchPath.
//!
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
char* shrFindFilePath(const char* filename, const char* executable_path) 
{
    // <executable_name> defines a variable that is replaced with the name of the executable

    // Typical relative search paths to locate needed companion files (e.g. sample input data, or JIT source files)
    // The origin for the relative search may be the .exe file, a .bat file launching an .exe, a browser .exe launching the .exe or .bat, etc
    const char* searchPath[] = 
    {
        "./",                                       // same dir 
        "./data/",                                  // "/data/" subdir 
        "./src/",                                   // "/src/" subdir
        "./inc/",                                   // "/inc/" subdir
        "../",                                      // up 1 in tree 
        "../data/",                                 // up 1 in tree, "/data/" subdir 
        "../src/",                                  // up 1 in tree, "/src/" subdir 
        "../inc/"                                   // up 1 in tree, "/inc/" subdir 
    };
    
    // Extract the executable name
    std::string executable_name;
    if (executable_path != 0) 
    {
        executable_name = std::string(executable_path);

       // Linux & OSX path delimiter
        size_t delimiter_pos = executable_name.find_last_of('/');        
        executable_name.erase(0,delimiter_pos+1);        
    }
    
    // Loop over all search paths and return the first hit
    for( unsigned int i = 0; i < sizeof(searchPath)/sizeof(char*); ++i )
    {
        std::string path(searchPath[i]);        
        size_t executable_name_pos = path.find("<executable_name>");

        // If there is executable_name variable in the searchPath 
        // replace it with the value
        if(executable_name_pos != std::string::npos)
        {
            if(executable_path != 0) 
            {
                path.replace(executable_name_pos, strlen("<executable_name>"), executable_name);

            } 
            else 
            {
                // Skip this path entry if no executable argument is given
                continue;
            }
        }
        
        // Test if the file exists
        path.append(filename);
        std::fstream fh(path.c_str(), std::fstream::in);
        if (fh.good())
        {
            // File found
            // returning an allocated array here for backwards compatibility reasons
            char* file_path = (char*) malloc(path.length() + 1);
            strcpy(file_path, path.c_str());                
            return file_path;
        }
    }    

    // File not found
    return 0;
}

// Helper function to init data arrays 
// *********************************************************************
void FillArray(float* pfData, int iSize)
{
    int i; 
    const float fScale = 1.0f / (float)RAND_MAX;
    for (i = 0; i < iSize; ++i) 
    {
        pfData[i] = fScale * rand();
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}

// Round Up Division function
size_t shrRoundUp(int group_size, int global_size) 
{
    int r = global_size % group_size;
    if(r == 0) 
    {
        return global_size;
    } else 
    {
        return global_size + group_size - r;
    }
}
////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

double executionTime(cl_event &event)
{
    cl_ulong start, end;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    
    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    
    printf("%s Starting...\n\n", argv[0]); 

    // run the code
    bool bOK = (runTest(argc, (const char **)argv) == CL_SUCCESS);
    printf("%s\n\n", (bOK ? "PASSED" : "FAILED"));

    return 0;
    
}

void matrixMulGPU(cl_uint ciDeviceCount, cl_mem h_A, float* h_B_data, unsigned int mem_size_B, float* h_C )
{
    cl_mem d_A[MAX_GPU_COUNT];
    cl_mem d_C[MAX_GPU_COUNT];
    cl_mem d_B[MAX_GPU_COUNT];

    cl_event GPUDone[MAX_GPU_COUNT];
    cl_event GPUExecution[MAX_GPU_COUNT];

    // Start the computation on each available GPU
    
    // Create buffers for each GPU
    // Each GPU will compute sizePerGPU rows of the result
    int sizePerGPU = uiHA / ciDeviceCount;

    int workOffset[MAX_GPU_COUNT];
    int workSize[MAX_GPU_COUNT];

    workOffset[0] = 0;
    for(unsigned int i=0; i < ciDeviceCount; ++i) 
    {
        // Input buffer
        workSize[i] = (i != (ciDeviceCount - 1)) ? sizePerGPU : (uiHA - workOffset[i]);        

        d_A[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, workSize[i] * sizeof(float) * uiWA, NULL,NULL);

        // Copy only assigned rows from host to device
        clEnqueueCopyBuffer(commandQueue[i], h_A, d_A[i], workOffset[i] * sizeof(float) * uiWA, 
                            0, workSize[i] * sizeof(float) * uiWA, 0, NULL, NULL);        
        
        // create OpenCL buffer on device that will be initiatlize from the host memory on first use
        // on device
        d_B[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                mem_size_B, h_B_data, NULL);

        // Output buffer
        d_C[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,  workSize[i] * uiWC * sizeof(float), NULL,NULL);
              
        // set the args values
        clSetKernelArg(multiplicationKernel[i], 0, sizeof(cl_mem), (void *) &d_C[i]);
        clSetKernelArg(multiplicationKernel[i], 1, sizeof(cl_mem), (void *) &d_A[i]);
        clSetKernelArg(multiplicationKernel[i], 2, sizeof(cl_mem), (void *) &d_B[i]);
        clSetKernelArg(multiplicationKernel[i], 3, sizeof(float) * BLOCK_SIZE *BLOCK_SIZE, 0 );
        clSetKernelArg(multiplicationKernel[i], 4, sizeof(float) * BLOCK_SIZE *BLOCK_SIZE, 0 );
        clSetKernelArg(multiplicationKernel[i], 5, sizeof(cl_int), (void *) &uiWA);
        clSetKernelArg(multiplicationKernel[i], 6, sizeof(cl_int), (void *) &uiWB);
        clSetKernelArg(multiplicationKernel[i], 7, sizeof(cl_int), (void *) &workSize[i]);

        if(i+1 < ciDeviceCount)
            workOffset[i + 1] = workOffset[i] + workSize[i];
    }
    
    // Execute Multiplication on all GPUs in parallel
    size_t localWorkSize[] = {BLOCK_SIZE, BLOCK_SIZE};
    size_t globalWorkSize[] = {shrRoundUp(BLOCK_SIZE, uiWC), shrRoundUp(BLOCK_SIZE, workSize[0])};
    
    // Launch kernels on devices
#if 1
	
	int nIter = 30;

    for (int j = -1; j < nIter; j++) 
    {
        // Sync all queues to host and start timer first time through loop
        if(j == 0){
            for(unsigned int i = 0; i < ciDeviceCount; i++) 
            {
                clFinish(commandQueue[i]);
            }

            shrDeltaT(0);
        }
#endif
        for(unsigned int i = 0; i < ciDeviceCount; i++) 
        {
			// Multiplication - non-blocking execution:  launch and push to device(s)
			globalWorkSize[1] = shrRoundUp(BLOCK_SIZE, workSize[i]);
			clEnqueueNDRangeKernel(commandQueue[i], multiplicationKernel[i], 2, 0, globalWorkSize, localWorkSize,
				                   0, NULL, &GPUExecution[i]);
            clFlush(commandQueue[i]);
		}

#if 1
    }
#endif

    // sync all queues to host
	for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
		clFinish(commandQueue[i]);
	}

#if 1

    // stop and log timer 
    double dSeconds = shrDeltaT(0)/(double)nIter;
    double dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
    double gflops = 1.0e-9 * dNumOps/dSeconds;
    printf("oclMatrixMul, Throughput = %.4f GFlops/s, Time = %.5f s, Size = %.0f, NumDevsUsed = %d, Workgroup = %lu\n", 
            gflops, dSeconds, dNumOps, ciDeviceCount, localWorkSize[0] * localWorkSize[1]);

    // Print kernel timing per GPU
    printf("\n");
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {    
        printf("  Kernel execution time on GPU %d \t: %.5f s\n", i, executionTime(GPUExecution[i]));
    }
    printf("\n");
#endif

    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {    
        // Non-blocking copy of result from device to host
        clEnqueueReadBuffer(commandQueue[i], d_C[i], CL_FALSE, 0, uiWC * sizeof(float) * workSize[i], 
                            h_C + workOffset[i] * uiWC, 0, NULL, &GPUDone[i]);
    }

	// CPU sync with GPU
    clWaitForEvents(ciDeviceCount, GPUDone);


    // Release mem and event objects    
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
        clReleaseMemObject(d_A[i]);
        clReleaseMemObject(d_C[i]);
        clReleaseMemObject(d_B[i]);
	    clReleaseEvent(GPUExecution[i]);
	    clReleaseEvent(GPUDone[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for 
////////////////////////////////////////////////////////////////////////////////
int runTest(int argc, const char** argv)
{
    cl_platform_id cpPlatform = NULL;
    cl_uint ciDeviceCount = 0;
    cl_device_id *cdDevices = NULL;
    cl_int ciErrNum = CL_SUCCESS;

    //Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: Failed to create OpenCL context!\n");
        return ciErrNum;
    }

    //Get the devices
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ciDeviceCount);
    printf("ciDeviceCount is %d\n", ciDeviceCount);
    cdDevices = (cl_device_id *)malloc(ciDeviceCount * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, ciDeviceCount, cdDevices, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: Failed to create OpenCL context 111!\n");
        return ciErrNum;
    }

    //Create the context
    cxGPUContext = clCreateContext(0, ciDeviceCount, cdDevices, NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: Failed to create OpenCL context 222!\n");
        return ciErrNum;
    }

    // if(shrCheckCmdLineFlag(argc, (const char**)argv, "device"))
    // {
    //     // User specified GPUs
    //     char* deviceList;
    //     char* deviceStr;
    //     char* next_token;
    //     shrGetCmdLineArgumentstr(argc, (const char**)argv, "device", &deviceList);

    //     #ifdef WIN32
    //         deviceStr = strtok_s (deviceList," ,.-", &next_token);
    //     #else
    //         deviceStr = strtok (deviceList," ,.-");
    //     #endif   
    //     ciDeviceCount = 0;
    //     while(deviceStr != NULL) 
    //     {
    //         // get and print the device for this queue
    //         cl_device_id device = oclGetDev(cxGPUContext, atoi(deviceStr));
	// 		if( device == (cl_device_id) -1  ) {
	// 			printf(" Device %s does not exist!\n", deviceStr);
	// 			return -1;
	// 		}
			
	// 		printf("Device %s: ", deviceStr);
    //         oclPrintDevName(LOGBOTH, device);            
    //         printf("\n");
           
    //         // create command queue
    //         commandQueue[ciDeviceCount] = clCreateCommandQueue(cxGPUContext, device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
    //         if (ciErrNum != CL_SUCCESS)
    //         {
    //             printf(" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
    //             return ciErrNum;
    //         }
                
    //         ++ciDeviceCount;

    //         #ifdef WIN32
    //             deviceStr = strtok_s (NULL," ,.-", &next_token);
    //         #else            
    //             deviceStr = strtok (NULL," ,.-");
    //         #endif
    //     }

    //     free(deviceList);
    // } 
    // else 
    // {
        // Find out how many GPU's to compute on all available GPUs
	    size_t nDeviceBytes;
	    ciErrNum |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
	    ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);

        if (ciErrNum != CL_SUCCESS)
        {
            printf(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
            return ciErrNum;
        }
        else if (ciDeviceCount == 0)
        {
            printf(" There are no devices supporting OpenCL (return code %i)\n\n", ciErrNum);
            return -1;
        } 

        // create command-queues
        for(unsigned int i = 0; i < ciDeviceCount; ++i) 
        {
            // get and print the device for this queue
            cl_device_id device = oclGetDev(cxGPUContext, i);
            printf("Device %d: \n", i);
            //oclPrintDevName(LOGBOTH, device);            
            //printf("\n");

            // create command queue
            commandQueue[i] = clCreateCommandQueue(cxGPUContext, device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                printf(" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
                return ciErrNum;
            }
        }
    // }

    // Optional Command-line multiplier for matrix sizes
    //shrGetCmdLineArgumenti(argc, (const char**)argv, "sizemult", &iSizeMultiple); 
    //iSizeMultiple = CLAMP(iSizeMultiple, 1, 10);
    //uiWA = WA * iSizeMultiple;
    //uiHA = HA * iSizeMultiple;
    //uiWB = WB * iSizeMultiple;
    //uiHB = HB * iSizeMultiple;
    //uiWC = WC * iSizeMultiple;
    //uiHC = HC * iSizeMultiple;
    uiWA = 4096;
    uiHA = 4096;
    uiWB = 4096;
    uiHB = 4096;
    uiWC = 4096;
    uiHC = 4096;

    printf("\nUsing Matrix Sizes: A(%u x %u), B(%u x %u), C(%u x %u)\n", 
            uiWA, uiHA, uiWB, uiHB, uiWC, uiHC);

    // allocate host memory for matrices A and B
    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A_data = (float*)malloc(mem_size_A);
    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B_data = (float*)malloc(mem_size_B);

    // initialize host memory
    srand(2006);
    FillArray(h_A_data, size_A);
    FillArray(h_B_data, size_B);

    // allocate host memory for result
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);

    // create OpenCL buffer pointing to the host memory
    cl_mem h_A = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				    mem_size_A, h_A_data, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clCreateBuffer\n");
        return ciErrNum;
    }

    // Program Setup
    size_t program_length;
    const char* header_path = shrFindFilePath("matrixMul.h", argv[0]);
    //oclCheckError(header_path != NULL, shrTRUE);
    char* header = oclLoadProgSource(header_path, "", &program_length);
    if(!header)
    {
        printf("Error: Failed to load the header %s!\n", header_path);
        return -1000;
    }
    const char* source_path = shrFindFilePath("matrixMul.cl", argv[0]);
    //oclCheckError(source_path != NULL, shrTRUE);
    char *source = oclLoadProgSource(source_path, header, &program_length);
    if(!source)
    {
        printf("Error: Failed to load compute program %s!\n", source_path);
        return -2000;
    }

    // create the program
    cl_program cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&source, 
                                                    &program_length, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: Failed to create program\n");
        return ciErrNum;
    }
    free(header);
    free(source);
    
    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then return error
        //shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        //oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        //oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMatrixMul.ptx");
        //return ciErrNum;
        printf("clBuildProgram failed!\n");
        exit(-1);
    }

    // // write out PTX if requested on the command line
    // if(shrCheckCmdLineFlag(argc, argv, "dump-ptx") )
    // {
    //     oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMatrixMul.ptx");
    // }

    // Create Kernel
    for(unsigned int i = 0; i < ciDeviceCount; ++i) {
        multiplicationKernel[i] = clCreateKernel(cpProgram, "matrixMul", &ciErrNum);
        if (ciErrNum != CL_SUCCESS)
        {
            printf("Error: Failed to create kernel\n");
            return ciErrNum;
        }
    }
        
    // Run multiplication on 1..deviceCount GPUs to compare improvement
    printf("\nRunning Computations on 1 - %d GPU's...\n\n", ciDeviceCount);
    for(unsigned int k = 1; k <= ciDeviceCount; ++k) 
    {
        matrixMulGPU(k, h_A, h_B_data, mem_size_B, h_C);
    }

    // compute reference solution
    printf("Comparing results with CPU computation... \n\n");
    float* reference = (float*) malloc(mem_size_C);
    computeGold(reference, h_A_data, h_B_data, uiHA, uiWA, uiWB);

    // check result
    bool res = sdkCompareL2fe(reference, h_C, size_C, 1.0e-6f);
    if (res != true) 
    {
         printDiff(reference, h_C, uiWC, uiHC, 100, 1.0e-5f);
    }

    // clean up OCL resources
    ciErrNum = clReleaseMemObject(h_A);
    for(unsigned int k = 0; k < ciDeviceCount; ++k) 
    {
        ciErrNum |= clReleaseKernel( multiplicationKernel[k] );
        ciErrNum |= clReleaseCommandQueue( commandQueue[k] );
    }
    ciErrNum |= clReleaseProgram(cpProgram);
    ciErrNum |= clReleaseContext(cxGPUContext);
    if(ciErrNum != CL_SUCCESS)
    {
        printf("Error: Failure releasing OpenCL resources: %d\n", ciErrNum);
        return ciErrNum;
    }

    // clean up memory
    free(h_A_data);
    free(h_B_data);
    free(h_C);
    free(reference);
    
    return 0; //return ((shrTRUE == res) ? CL_SUCCESS : -3000);
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;
    for (j = 0; j < height; j++) 
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }
        for (i = 0; i < width; i++) 
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);
            if (fDiff > fListTol) 
            {                
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    printf(" \n  Total Errors = %d\n\n", error_count);
}
