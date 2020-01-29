#ifndef AIE_ALGORITHMS_H
#define AIE_ALGORITHMS_H

#include "aie_data_struct.h"


/////shared_data_adaptive without rearrangement
template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_dense_shared_data_adaptive(int tree, float* sdata, int cols, vec<NITEMS>& out, 
								algo_t algo_, int num_trees, float missing, int num_nodes_per_tree, float* bits_values_d, char* bits_char_d) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = bits_values_d[tree*num_nodes_per_tree + curr[j]];
      char n_bits = bits_char_d[tree*num_nodes_per_tree + curr[j]];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 5) - 1);
	  bool n_def_left = n_bits & (1 << 5);
	  bool n_is_leaf = n_bits & (1 << 6);
	  bool n_is_exchange = n_bits & (1 << 7);
	  //printf("%d\n", curr[j]);
	  //printf("n_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d, n_is_exchange: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf, n_is_exchange);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      const float eps=1.0e-6;
      bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;

	  if(n_is_exchange) cond = !cond;

	  curr[j] = (curr[j]<<1)+1+cond;

    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += bits_values_d[tree*num_nodes_per_tree + curr[j]];
  //printf("\n");
}


template <int NITEMS>
__global__ void infer_k_shared_data_adaptive(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_d, char* bits_char_d) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;

  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = threadIdx.x; j < num_trees_; j += blockDim.x) {
    infer_one_tree_dense_shared_data_adaptive<NITEMS>(j, sdata, params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_d, bits_char_d);
  }
  __syncthreads();

  typedef cub::BlockReduce<vec<NITEMS>, FIL_TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);

  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int idx = blockIdx.x * NITEMS + i;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[i];
	  }
    }
  }
}







/////shared_data_adaptive with rearrangement for int
template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_dense_shared_data_adaptive_reorg_int(int tree, float* sdata, int cols, vec<NITEMS>& out, 
								algo_t algo_, int num_trees, float missing, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = bits_values_reorg_d[tree + curr[j]*num_trees];
      int n_bits = bits_int_reorg_d[tree + curr[j]*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 29) - 1);
	  bool n_def_left = n_bits & (1 << 29);
	  bool n_is_leaf = n_bits & (1 << 30);
	  bool n_is_exchange = n_bits & (1 << 31);
	  //printf("%d\n", curr[j]);
	  //printf("n_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d, n_is_exchange: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf, n_is_exchange);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      const float eps=1.0e-6;
      bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;

	  if(n_is_exchange) cond = !cond;

	  curr[j] = (curr[j]<<1)+1+cond;

    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += bits_values_reorg_d[tree + curr[j]*num_trees];
  //printf("\n");
}


template <int NITEMS>
__global__ void infer_k_shared_data_adaptive_reorg_int(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;

  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = threadIdx.x; j < num_trees_; j += blockDim.x) {
    infer_one_tree_dense_shared_data_adaptive_reorg_int<NITEMS>(j, sdata, params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_int_reorg_d);
  }
  __syncthreads();

  typedef cub::BlockReduce<vec<NITEMS>, FIL_TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);

  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int idx = blockIdx.x * NITEMS + i;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[i];
	  }
    }
  }
}





/////shared_data_adaptive with rearrangement for short
template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_dense_shared_data_adaptive_reorg_short(int tree, float* sdata, int cols, vec<NITEMS>& out, 
								algo_t algo_, int num_trees, float missing, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = bits_values_reorg_d[tree + curr[j]*num_trees];
      short n_bits = bits_short_reorg_d[tree + curr[j]*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 13) - 1);
	  bool n_def_left = n_bits & (1 << 13);
	  bool n_is_leaf = n_bits & (1 << 14);
	  bool n_is_exchange = n_bits & (1 << 15);
	  //printf("%d\n", curr[j]);
	  //printf("n_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d, n_is_exchange: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf, n_is_exchange);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      const float eps=1.0e-6;
      bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;

	  if(n_is_exchange) cond = !cond;

	  curr[j] = (curr[j]<<1)+1+cond;

    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += bits_values_reorg_d[tree + curr[j]*num_trees];
  //printf("\n");
}


template <int NITEMS>
__global__ void infer_k_shared_data_adaptive_reorg_short(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;

  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = threadIdx.x; j < num_trees_; j += blockDim.x) {
    infer_one_tree_dense_shared_data_adaptive_reorg_short<NITEMS>(j, sdata, params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_short_reorg_d);
  }
  __syncthreads();

  typedef cub::BlockReduce<vec<NITEMS>, FIL_TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);

  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int idx = blockIdx.x * NITEMS + i;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[i];
	  }
    }
  }
}




/////shared_data_adaptive with rearrangement for char
template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_dense_shared_data_adaptive_reorg_char(int tree, float* sdata, int cols, vec<NITEMS>& out, 
								algo_t algo_, int num_trees, float missing, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = bits_values_reorg_d[tree + curr[j]*num_trees];
      char n_bits = bits_char_reorg_d[tree + curr[j]*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 5) - 1);
	  bool n_def_left = n_bits & (1 << 5);
	  bool n_is_leaf = n_bits & (1 << 6);
	  bool n_is_exchange = n_bits & (1 << 7);
	  //printf("%d\n", curr[j]);
	  //printf("n_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d, n_is_exchange: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf, n_is_exchange);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      const float eps=1.0e-6;
      bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;

	  if(n_is_exchange) cond = !cond;

	  curr[j] = (curr[j]<<1)+1+cond;

    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += bits_values_reorg_d[tree + curr[j]*num_trees];
  //printf("\n");
}


template <int NITEMS>
__global__ void infer_k_shared_data_adaptive_reorg_char(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;

  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = threadIdx.x; j < num_trees_; j += blockDim.x) {
    infer_one_tree_dense_shared_data_adaptive_reorg_char<NITEMS>(j, sdata, params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_char_reorg_d);
  }
  __syncthreads();

  typedef cub::BlockReduce<vec<NITEMS>, FIL_TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);

  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int idx = blockIdx.x * NITEMS + i;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[i];
	  }
    }
  }
}




template <int NITEMS>
__global__ void infer_adaptive_reorg_char(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d) {
  int index = blockIdx.x * FIL_TPB + threadIdx.x;
  if(index < params.num_rows)
  {
  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = 0; j < num_trees_; j++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_char<NITEMS>(j, const_cast<float*>(&params.data[index * params.num_cols]), params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_char_reorg_d);
  }
  params.preds[index] = out[0];
  }

}
template <int NITEMS>
__global__ void infer_adaptive_reorg_short(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d) {
  int index = blockIdx.x * FIL_TPB + threadIdx.x;
  if(index < params.num_rows)
  {
  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = 0; j < num_trees_; j++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_short<NITEMS>(j, const_cast<float*>(&params.data[index * params.num_cols]), params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_short_reorg_d);
  }
  params.preds[index] = out[0];
  }

}
template <int NITEMS>
__global__ void infer_adaptive_reorg_int(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d) {
  int index = blockIdx.x * FIL_TPB + threadIdx.x;
  if(index < params.num_rows)
  {
  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = 0; j < num_trees_; j++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_int<NITEMS>(j, const_cast<float*>(&params.data[index * params.num_cols]), params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_int_reorg_d);
  }
  params.preds[index] = out[0];
  }

}




__device__ __forceinline__ void infer_one_tree_dense_shared_forest_adaptive_reorg_char(char* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  do {
      float n_val = stree_vals[num_tree_curr + curr*num_trees];
      char n_bits = stree_bits[num_tree_curr + curr*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 5) - 1);
	  bool n_def_left = n_bits & (1 << 5);
	  bool n_is_leaf = n_bits & (1 << 6);
	  bool n_is_exchange = n_bits & (1 << 7);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}

__global__ void infer_k_shared_forest_adaptive_reorg_char(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  char* stree_bits = smem + sizeof(float)*num_trees_*num_nodes_per_tree;
  for (int i = 0; i < num_trees_; i++) {
	for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
	{
		stree_bits[i*num_nodes_per_tree+j] = bits_char_reorg_d[i*num_nodes_per_tree+j];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[i*num_nodes_per_tree+j];
	}
  }
  __syncthreads();

  float out = 0.0;

  int idx = blockIdx.x * FIL_TPB + threadIdx.x;
  if (idx < params.num_rows)
  {
    infer_one_tree_dense_shared_forest_adaptive_reorg_char(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, num_trees_, num_nodes_per_tree, params.missing);
	params.preds[idx] = out;
  }

}


__device__ __forceinline__ void infer_one_tree_dense_shared_forest_adaptive_reorg_int(int* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  do {
      float n_val = stree_vals[num_tree_curr + curr*num_trees];
      int n_bits = stree_bits[num_tree_curr + curr*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 29) - 1);
	  bool n_def_left = n_bits & (1 << 29);
	  bool n_is_leaf = n_bits & (1 << 30);
	  bool n_is_exchange = n_bits & (1 << 31);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_shared_forest_adaptive_reorg_int(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  int index = sizeof(float)*num_trees_*num_nodes_per_tree;
  while(index%4 != 0) index++;
  int* stree_bits = (int*) (smem + index);
  for (int i = 0; i < num_trees_; i++) {
	for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
	{
		stree_bits[i*num_nodes_per_tree+j] = bits_int_reorg_d[i*num_nodes_per_tree+j];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[i*num_nodes_per_tree+j];
	}
  }
  __syncthreads();

  float out = 0.0;

  int idx = blockIdx.x * FIL_TPB + threadIdx.x;
  if (idx < params.num_rows)
  {
    infer_one_tree_dense_shared_forest_adaptive_reorg_int(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, num_trees_, num_nodes_per_tree, params.missing);
	params.preds[idx] = out;
  }

}




__device__ __forceinline__ void infer_one_tree_dense_shared_forest_adaptive_reorg_short(short* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  do {
      float n_val = stree_vals[num_tree_curr + curr*num_trees];
      short n_bits = stree_bits[num_tree_curr + curr*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 13) - 1);
	  bool n_def_left = n_bits & (1 << 13);
	  bool n_is_leaf = n_bits & (1 << 14);
	  bool n_is_exchange = n_bits & (1 << 15);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_shared_forest_adaptive_reorg_short(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  int index = sizeof(float)*num_trees_*num_nodes_per_tree;
  while(index%2 != 0) index++;
  short* stree_bits = (short*) (smem + index);
  for (int i = 0; i < num_trees_; i++) {
	for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
	{
		stree_bits[i*num_nodes_per_tree+j] = bits_short_reorg_d[i*num_nodes_per_tree+j];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[i*num_nodes_per_tree+j];
	}
  }
  __syncthreads();

  float out = 0.0;

  int idx = blockIdx.x * FIL_TPB + threadIdx.x;
  if (idx < params.num_rows)
  {
    infer_one_tree_dense_shared_forest_adaptive_reorg_short(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, num_trees_, num_nodes_per_tree, params.missing);
	params.preds[idx] = out;
  }

}




__device__ __forceinline__ void infer_one_tree_dense_split_forest_adaptive_reorg_int(int* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  int* curr_tree_bits = stree_bits;
  float* curr_tree_vals = stree_vals;
  do {
      float n_val = curr_tree_vals[curr];
      int n_bits = curr_tree_bits[curr];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 29) - 1);
	  bool n_def_left = n_bits & (1 << 29);
	  bool n_is_leaf = n_bits & (1 << 30);
	  bool n_is_exchange = n_bits & (1 << 31);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr_tree_bits+=num_nodes;
		  curr_tree_vals+=num_nodes;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_split_forest_adaptive_reorg_int(predict_params params, int trees_per_sm, float* temp_out, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  int* stree_bits = (int*)(smem + sizeof(float)*trees_per_sm*num_nodes_per_tree);
  int trees = 0;
  for (int i = 0; i < trees_per_sm; i++) {
	if((trees_per_sm * (blockIdx.x) + i) < num_trees_)
	{
		for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
		{
		int tree_index = i+trees_per_sm*(blockIdx.x);
		stree_bits[i*num_nodes_per_tree+j] = bits_int_reorg_d[tree_index + j*num_trees_];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[tree_index + j*num_trees_];
		}
		trees++;
	}
  }
  __syncthreads();

  float out=0.0;

  for(int idx= threadIdx.x; idx<params.num_rows; idx+=blockDim.x)
  {
	out = 0.0;
    infer_one_tree_dense_split_forest_adaptive_reorg_int(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, trees, num_nodes_per_tree, params.missing);
	temp_out[idx*gridDim.x + blockIdx.x] = out;
  }

}




__device__ __forceinline__ void infer_one_tree_dense_split_forest_adaptive_reorg_short(short* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  short* curr_tree_bits = stree_bits;
  float* curr_tree_vals = stree_vals;
  do {
      float n_val = curr_tree_vals[curr];
      short n_bits = curr_tree_bits[curr];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 13) - 1);
	  bool n_def_left = n_bits & (1 << 13);
	  bool n_is_leaf = n_bits & (1 << 14);
	  bool n_is_exchange = n_bits & (1 << 15);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr_tree_bits+=num_nodes;
		  curr_tree_vals+=num_nodes;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_split_forest_adaptive_reorg_short(predict_params params, int trees_per_sm, float* temp_out, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  short* stree_bits = (short*)(smem + sizeof(float)*trees_per_sm*num_nodes_per_tree);
  int trees = 0;
  for (int i = 0; i < trees_per_sm; i++) {
	if((trees_per_sm * (blockIdx.x) + i) < num_trees_)
	{
		for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
		{
		int tree_index = i+trees_per_sm*(blockIdx.x);
		stree_bits[i*num_nodes_per_tree+j] = bits_short_reorg_d[tree_index + j*num_trees_];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[tree_index + j*num_trees_];
		}
		trees++;
	}
  }
  __syncthreads();

  float out=0.0;

  for(int idx= threadIdx.x; idx<params.num_rows; idx+=blockDim.x)
  {
	out = 0.0;
    infer_one_tree_dense_split_forest_adaptive_reorg_short(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, trees, num_nodes_per_tree, params.missing);
	temp_out[idx*gridDim.x + blockIdx.x] = out;
  }

}



__device__ __forceinline__ void infer_one_tree_dense_split_forest_adaptive_reorg_char(char* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  char* curr_tree_bits = stree_bits;
  float* curr_tree_vals = stree_vals;
  do {
      float n_val = curr_tree_vals[curr];
      char n_bits = curr_tree_bits[curr];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 5) - 1);
	  bool n_def_left = n_bits & (1 << 5);
	  bool n_is_leaf = n_bits & (1 << 6);
	  bool n_is_exchange = n_bits & (1 << 7);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr_tree_bits+=num_nodes;
		  curr_tree_vals+=num_nodes;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_split_forest_adaptive_reorg_char(predict_params params, int trees_per_sm, float* temp_out, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  char* stree_bits = smem + sizeof(float)*trees_per_sm*num_nodes_per_tree;
  int trees = 0;
  for (int i = 0; i < trees_per_sm; i++) {
	if((trees_per_sm * (blockIdx.x) + i) < num_trees_)
	{
		for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
		{
		int tree_index = i+trees_per_sm*(blockIdx.x);
		stree_bits[i*num_nodes_per_tree+j] = bits_char_reorg_d[tree_index + j*num_trees_];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[tree_index + j*num_trees_];
		}
		trees++;
	}
  }
  __syncthreads();

  float out=0.0;

  for(int idx= threadIdx.x; idx<params.num_rows; idx+=blockDim.x)
  {
	out = 0.0;
    infer_one_tree_dense_split_forest_adaptive_reorg_char(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, trees, num_nodes_per_tree, params.missing);
	temp_out[idx*gridDim.x + blockIdx.x] = out;
  }

}



template <int NITEMS>
__global__ void infer_k_shared_data_wo_adaptive_reorg_int(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d, int num) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * num;

  for (int j = threadIdx.x; (j < num) && ((j+rid) < params.num_rows); j += blockDim.x) {
    for (int i = 0; i < params.num_cols; i ++) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = params.data[row * params.num_cols + i] ;
    }

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int k = 0; k < num_trees_; k ++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_int<NITEMS>(k, &sdata[j * params.num_cols], params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_int_reorg_d);
  }

      int idx = blockIdx.x * num + j;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[0];
	  }

}

}




template <int NITEMS>
__global__ void infer_k_shared_data_wo_adaptive_reorg_short(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d, int num) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * num;

  for (int j = threadIdx.x; (j < num) && ((j+rid) < params.num_rows); j += blockDim.x) {
    for (int i = 0; i < params.num_cols; i ++) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = params.data[row * params.num_cols + i] ;
    }

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int k = 0; k < num_trees_; k ++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_short<NITEMS>(k, &sdata[j * params.num_cols], params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_short_reorg_d);
  }

      int idx = blockIdx.x * num + j;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[0];
	  }

}

}




template <int NITEMS>
__global__ void infer_k_shared_data_wo_adaptive_reorg_char(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d, int num) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * num;

  for (int j = threadIdx.x; (j < num) && ((j+rid) < params.num_rows); j += blockDim.x) {
    for (int i = 0; i < params.num_cols; i ++) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = params.data[row * params.num_cols + i] ;
    }

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int k = 0; k < num_trees_; k ++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_char<NITEMS>(k, &sdata[j * params.num_cols], params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_char_reorg_d);
  }

      int idx = blockIdx.x * num + j;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[0];
	  }

}

}







#endif

