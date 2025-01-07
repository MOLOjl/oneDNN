/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <hip/hip_runtime.h>
#include "hip_customs.h"
#include <hip/hip_fp16.h>

namespace hip_custom {

void print_device_array(void* dev_a, size_t length, int dtype)
{
  int element_bytes = 1;
  FILE* fp;
  fp = fopen("./array_log.txt", "a");
  fprintf(fp, "======================================================================\n");

  if(dtype == 0){
    fprintf(fp, "---------------------float, lenth:%ld-------------------\n", length);
    int element_bytes = sizeof(float);
    float* host_a = (float*)malloc(element_bytes*length);

    hipMemcpy(host_a, dev_a, element_bytes*length, hipMemcpyDeviceToHost);

    for(size_t i=0; i<length; i++){
      if(i % 8 == 0)
        fprintf(fp, "\n");
      fprintf(fp, "%.3f, ", host_a[i]);
    }

    free(host_a);
  }
  else if(dtype == 1){
    fprintf(fp, "---------------------half, lenth:%ld-------------------\n", length);
    int element_bytes = sizeof(half);
    half* host_a = (half*)malloc(element_bytes*length);

    hipMemcpy(host_a, dev_a, element_bytes*length, hipMemcpyDeviceToHost);

    for(size_t i=0; i<length; i++){
      if(i % 8 == 0)
        fprintf(fp, "\n");
      fprintf(fp, "%.3f, ", __half2float(host_a[i]));
    }

    free(host_a);
  }
  else if(dtype == 2){
    fprintf(fp, "---------------------int32, lenth:%ld-------------------\n", length);
    int element_bytes = sizeof(int32_t);
    int32_t* host_a = (int32_t*)malloc(element_bytes*length);

    hipMemcpy(host_a, dev_a, element_bytes*length, hipMemcpyDeviceToHost);

    for(size_t i=0; i<length; i++){
      if(i % 8 == 0)
        fprintf(fp, "\n");
      fprintf(fp, "%d, ", host_a[i]);
    }

    free(host_a);
  }
  else if(dtype == 3){
    fprintf(fp, "---------------------int8, lenth:%ld-------------------\n", length);
    int element_bytes = sizeof(int8_t);
    int8_t* host_a = (int8_t*)malloc(element_bytes*length);

    hipMemcpy(host_a, dev_a, element_bytes*length, hipMemcpyDeviceToHost);

    for(size_t i=0; i<length; i++){
      if(i % 8 == 0)
        fprintf(fp, "\n");
      fprintf(fp, "%d, ", host_a[i]);
    }

    free(host_a);
  }

  fclose(fp);
}

} // hip_custom