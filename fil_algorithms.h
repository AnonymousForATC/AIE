#ifndef FIL_ALGORITHMS_H
#define FIL_ALGORITHMS_H
#include "cuda_base.h"
#include "fil_data_struct.h"
#include <time.h>
#include <string>
#include <math.h> 



void encode_node_adaptive(std::vector<float> &values_reorder_h, std::vector<int> &fids_reorder_h, bool* defaults_reorder_h, bool* is_leafs_reorder_h, bool* exchanges_reorder_h, float* bits_values_h, char* bits_char_h, short int* bits_short_h, int* bits_int_h, int bits_length, int length)
{
	for(int i=0; i<length; i++)
	{
		bits_values_h[i] = values_reorder_h[i];
		if(bits_length == 1)
		{
			bits_char_h[i] = (fids_reorder_h[i] & FID_MASK_CHAR) | (defaults_reorder_h[i] ? DEF_LEFT_MASK_CHAR : 0) | (is_leafs_reorder_h[i] ? IS_LEAF_MASK_CHAR : 0)
							| (exchanges_reorder_h[i] ? EXCHANGE_MASK_CHAR : 0);
		}
		if(bits_length == 2)
		{
			bits_short_h[i] = (fids_reorder_h[i] & FID_MASK_SHORT) | (defaults_reorder_h[i] ? DEF_LEFT_MASK_SHORT : 0) 
			                | (is_leafs_reorder_h[i] ? IS_LEAF_MASK_SHORT : 0) | (exchanges_reorder_h[i] ? EXCHANGE_MASK_SHORT : 0);
		}
		if(bits_length == 4)
		{
			bits_int_h[i] = (fids_reorder_h[i] & FID_MASK_INT) | (defaults_reorder_h[i] ? DEF_LEFT_MASK_INT : 0) | (is_leafs_reorder_h[i] ? IS_LEAF_MASK_INT : 0)
							| (exchanges_reorder_h[i] ? EXCHANGE_MASK_INT : 0);
		}
	}
}


void encode_node(dense_node_t* node, int fid, float value, bool def_left, float weight, bool is_leaf)
{
	node->weight = weight;
	node->val = value;
	node->bits = (fid & FID_MASK) | (def_left ? DEF_LEFT_MASK : 0) | (is_leaf ? IS_LEAF_MASK : 0);
}

void dense_node_decode(const dense_node_t* n, float* value, float* weight,
                       int* fid, bool* def_left, bool* is_leaf) {
  *value = n->val;
  *weight = n->weight;
  *fid = n->bits & FID_MASK;
  *def_left = n->bits & DEF_LEFT_MASK;
  *is_leaf = n->bits & IS_LEAF_MASK;
}





template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_dense_shared_data(dense_tree tree, float* sdata,
                                               int cols, vec<NITEMS>& out, algo_t algo_, int num_trees, float missing) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = tree.nodes_[curr[j]].val;
      int n_bits = tree.nodes_[curr[j]].bits;
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 30) - 1);
	  bool n_def_left = n_bits & (1 << 30);
	  bool n_is_leaf = n_bits & (1 << 31);
	  //printf("%d\n", &tree.nodes_[curr[j]]);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      const float eps=1.0e-6;
      bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;

	  if(algo_ == algo_t::NAIVE)
	  {
		  curr[j] = (curr[j]<<1)+1+cond;
	  }
	  if(algo_ == algo_t::TREE_REORG)
	  {
		  unsigned int temp = ((curr[j]/num_trees)*2+1)*num_trees + (cond ? num_trees : 0);////////////////////////////////////////////////////
		  curr[j] = temp ;/////////////////////////////////////////////////////////////////
		  //curr[j] = (((curr[j]/num_trees)*2)+1)*num_trees + curr[j]%2 + cond;
	  }
	  if(algo_ == algo_t::BATCH_TREE_REORG)
	  {
		  unsigned int temp = ((curr[j]/num_trees)*2+1)*num_trees + (cond ? num_trees : 0);////////////////////////////////////////////////////
		  curr[j] = temp ;/////////////////////////////////////////////////////////////////
		  //curr[j] = (((curr[j]/num_trees)*2)+1)*num_trees + curr[j]%2 + cond;
	  }

    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += tree.nodes_[curr[j]].val;
  //printf("\n");
}


template <int NITEMS>
__global__ void infer_k_shared_data(dense_storage forest, predict_params params) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;
  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] =
        row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  // one block works on a single row and the whole forest
  for (int j = threadIdx.x; j < forest.num_trees(); j += blockDim.x) {
    //for(int loop=0;loop<50;loop++)
    infer_one_tree_dense_shared_data<NITEMS>(forest[j], sdata, params.num_cols, out, params.algo, forest.num_trees(), params.missing);
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


void infer_dense_shared_data(dense_storage forest, predict_params params, cudaStream_t stream) {
  //printf("shared memory is %d\n", params.max_shm);
  int num_items = params.max_shm / (sizeof(float) * params.num_cols);
  //printf("shared memory is %d\n", params.max_shm);
  if (num_items == 0) {
    //int max_cols = params.max_shm / sizeof(float);
    assert(false && "too many features");
  }
  num_items = params.algo == algo_t::BATCH_TREE_REORG ? 1 : 1;
  //printf("MAX_BATCH_ITEMS: %d\n", num_items);
  int num_blocks = ceildiv(int(params.num_rows), num_items);
  int shm_sz = num_items * sizeof(float) * params.num_cols;
  //printf("start infer_k: %d, num_blocks: %d, FIL_TPB: %d, shm_sz: %d, stream: %d\n", num_items, num_blocks, FIL_TPB, shm_sz, stream);

  switch (num_items) {
    case 1:
      //infer_k_shared_data<1><<<1, 1, shm_sz, stream>>>(forest, params);
      infer_k_shared_data<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 2:
      infer_k_shared_data<2><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 3:
      infer_k_shared_data<3><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 4:
      infer_k_shared_data<4><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 16:
      infer_k_shared_data<16><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 24:
      infer_k_shared_data<24><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    default:
      assert(false && "internal error: nitems > 4");
  }
  //printf("end infer_k\n");
  CUDA_CHECK(cudaPeekAtLastError());
}





__device__ __forceinline__ void infer_one_tree_dense_shared_forest(dense_node_t* tree, const float* sdata, int cols, float& out, int num_trees, int num_nodes) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  dense_node_t* curr_tree = tree;
  do {
      float n_val = curr_tree[curr].val;
      int n_bits = curr_tree[curr].bits;
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 30) - 1);
	  bool n_def_left = n_bits & (1 << 30);
	  bool n_is_leaf = n_bits & (1 << 31);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr_tree+=num_nodes;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
      bool cond = isnan(val) ? !n_def_left : val >= n_thresh;
	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_shared_forest(dense_storage forest, predict_params params) {
  extern __shared__ char smem[];
  dense_node_t* stree = (dense_node_t*)smem;
  for (int i = 0; i < forest.num_trees(); i++) {
	for(int j= threadIdx.x; j<forest.tree_stride_; j+=blockDim.x)
	{
		stree[i*forest.tree_stride_+j].val = forest[i].nodes_[j].val;
		stree[i*forest.tree_stride_+j].bits = forest[i].nodes_[j].bits;
	}
  }
  __syncthreads();

  float out = 0.0;

  int idx = blockIdx.x * FIL_TPB + threadIdx.x;
  if (idx < params.num_rows)
  {
    infer_one_tree_dense_shared_forest(stree, &params.data[idx*params.num_cols], params.num_cols, out, forest.num_trees(), forest.tree_stride_);
	params.preds[idx] = out;
  }

}

void infer_dense_shared_forest(dense_storage forest, predict_params params, cudaStream_t stream) {
  int shm_sz = forest.num_trees() * sizeof(struct dense_node_t) * forest.tree_stride_;
  //printf("shared memory is %d\n", params.max_shm);
  if (shm_sz > params.max_shm) {
    assert(false && "forest is too large to save in shared memory");
  }

  int num_blocks = ceildiv(int(params.num_rows), FIL_TPB);
  //printf("%d - %d\n", num_blocks, FIL_TPB);
  infer_k_shared_forest<<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
  //printf("end infer_k\n");
  CUDA_CHECK(cudaPeekAtLastError());
}









__device__ __forceinline__ void infer_one_tree_dense_split_forest(dense_node_t* tree, const float* sdata, int cols, float& out, int num_trees, int num_nodes) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  dense_node_t* curr_tree = tree;
  do {
      float n_val = curr_tree[curr].val;
      int n_bits = curr_tree[curr].bits;
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 30) - 1);
	  bool n_def_left = n_bits & (1 << 30);
	  bool n_is_leaf = n_bits & (1 << 31);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr_tree+=num_nodes;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
      bool cond = isnan(val) ? !n_def_left : val >= n_thresh;
	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}



__global__ void infer_k_split_forest(dense_storage forest, predict_params params, int trees_per_sm, float* temp_out) {
  extern __shared__ char smem[];
  dense_node_t* stree = (dense_node_t*)smem;
  int trees = 0;
  for (int i = 0; i < trees_per_sm; i++) {
  //for (int i = trees_per_sm*(blockIdx.x); i < trees_per_sm*(blockIdx.x+1) && i < forest.num_trees(); i++) {
	if((trees_per_sm * (blockIdx.x) + i) < forest.num_trees())
	{
		for(int j= threadIdx.x; j<forest.tree_stride_; j+=blockDim.x)
		{
		stree[i*forest.tree_stride_+j].val = forest[trees_per_sm * (blockIdx.x) + i].nodes_[j].val;
		stree[i*forest.tree_stride_+j].bits = forest[trees_per_sm * (blockIdx.x) + i].nodes_[j].bits;
		}
		trees++;
	}
  }
  __syncthreads();

  float out=0.0;

  for(int idx= threadIdx.x; idx<params.num_rows; idx+=blockDim.x)
  {
	out = 0.0;
    infer_one_tree_dense_split_forest(stree, &params.data[idx*params.num_cols], params.num_cols, out, trees, forest.tree_stride_);
	temp_out[idx*gridDim.x + blockIdx.x] = out;
  }

}

void infer_dense_split_forest(dense_storage forest, predict_params params, cudaStream_t stream) {
  int trees_per_sm = params.max_shm / (sizeof(struct dense_node_t) * forest.tree_stride_);
  //printf("shared memory is %d\n", params.max_shm);
  if (trees_per_sm == 0) {
    assert(false && "single tree is too big to save in shared memory");
  }
  int num_blocks = ceildiv(forest.num_trees(), trees_per_sm);
  int shm_sz = trees_per_sm * (sizeof(struct dense_node_t)) * forest.tree_stride_;////////////////////////////DO NOT KNOW WHY!!!!!!!!!!!!!
  //printf("shared memory is %d\n", shm_sz);
  //printf("num trees is %d, tree_stride_ is %d\n", forest.num_trees(), forest.tree_stride_);
  //printf("start infer_k: %d, num_blocks: %d, FIL_TPB: %d, shm_sz: %d, stream: %d\n", num_items, num_blocks, FIL_TPB, shm_sz, stream);
  //printf("trees_per_sm: %d, num_blocks: %d\n", trees_per_sm, num_blocks);

  float* temp_out = NULL;
  allocate(temp_out, num_blocks*params.num_rows);

  infer_k_split_forest<<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params, trees_per_sm, temp_out);

  int* h_offsets=(int*)malloc(sizeof(int)*(params.num_rows+1));
  for(int i=0;i<=params.num_rows;i++)
	  h_offsets[i]=i*num_blocks;

  int* d_offsets;
  allocate(d_offsets, (params.num_rows+1));
  updateDevice(d_offsets, h_offsets, (params.num_rows+1), stream);

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, temp_out, params.preds, params.num_rows, d_offsets, d_offsets + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sum-reduction
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, temp_out, params.preds, params.num_rows, d_offsets, d_offsets + 1);

  CUDA_CHECK(cudaFree(d_temp_storage));

  free(h_offsets);
  CUDA_CHECK(cudaFree(d_offsets));

  CUDA_CHECK(cudaFree(temp_out));

  //printf("end infer_k\n");
  CUDA_CHECK(cudaPeekAtLastError());
}












__global__ void infer_k_split_forest_shared_data(dense_storage forest, predict_params params, int trees_per_sm, int data_per_sm, float* temp_out) {
  extern __shared__ char smem[];
  dense_node_t* stree = (dense_node_t*)smem;
  int trees = 0;
  for (int i = 0; i < trees_per_sm; i++) {
	if((trees_per_sm * (blockIdx.x) + i) < forest.num_trees())
	{
		for(int j= threadIdx.x; j<forest.tree_stride_; j+=blockDim.x)
		{
		stree[i*forest.tree_stride_+j].val = forest[trees_per_sm * (blockIdx.x) + i].nodes_[j].val;
		stree[i*forest.tree_stride_+j].bits = forest[trees_per_sm * (blockIdx.x) + i].nodes_[j].bits;
		}
		trees++;
	}
  }

  __syncthreads();


  int loops = ceildiv_dev(params.num_rows, data_per_sm);
  float* sdata = (float*)&(smem[trees_per_sm * (sizeof(struct dense_node_t)) * forest.tree_stride_]);
  for(int loop=0; loop<loops; loop++)
  {

   for (int j = 0; j < data_per_sm && (loop*data_per_sm+j < params.num_rows); ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      sdata[j * params.num_cols + i] = params.data[(loop*data_per_sm+j) * params.num_cols + i];
    }
   __syncthreads();

  for(int idx= threadIdx.x; idx<data_per_sm; idx+=blockDim.x)
  {
    float out = 0.0;
    infer_one_tree_dense_split_forest(stree, &sdata[idx*params.num_cols], params.num_cols, out, trees, forest.tree_stride_);
    temp_out[(idx+loop*data_per_sm)*gridDim.x + blockIdx.x] = out;
  }

  }

  }


  /*
  float out=0.0;

  for(int idx= threadIdx.x; idx<params.num_rows; idx+=blockDim.x)
  {
	out = 0.0;
    infer_one_tree_dense_split_forest(stree, &params.data[idx*params.num_cols], params.num_cols, out, trees, forest.tree_stride_);
	temp_out[idx*gridDim.x + blockIdx.x] = out;
  }
  */

}



void infer_dense_split_forest_shared_data(dense_storage forest, predict_params params, cudaStream_t stream) {
  int trees_per_sm = (params.max_shm / 2.0) / (sizeof(struct dense_node_t) * forest.tree_stride_);
  //printf("shared memory is %d\n", params.max_shm);
  if (trees_per_sm == 0) {
    assert(false && "single tree is too big to save in shared memory");
  }
  int num_blocks = ceildiv(forest.num_trees(), trees_per_sm);
  //printf("shared memory is %d\n", shm_sz);
  //printf("num trees is %d, tree_stride_ is %d\n", forest.num_trees(), forest.tree_stride_);
  //printf("start infer_k: %d, num_blocks: %d, FIL_TPB: %d, shm_sz: %d, stream: %d\n", num_items, num_blocks, FIL_TPB, shm_sz, stream);
  //printf("trees_per_sm: %d, num_blocks: %d\n", trees_per_sm, num_blocks);


  //printf("shared memory is %d\n", params.max_shm);
  int data_per_sm = (params.max_shm / 2.0) / (sizeof(float) * params.num_cols);
  //printf("shared memory is %d\n", params.max_shm);
  if (data_per_sm == 0) {
    //int max_cols = params.max_shm / sizeof(float);
    assert(false && "too many features");
  }

  int shm_sz = trees_per_sm * (sizeof(struct dense_node_t)) * forest.tree_stride_ + data_per_sm * sizeof(float) * params.num_cols;////////////////////////////DO NOT KNOW WHY!!!!!!!!!!!!!

  //printf("trees_per_sm: %d, data_per_sm: %d\n", trees_per_sm, data_per_sm);

  float* temp_out = NULL;
  allocate(temp_out, num_blocks*params.num_rows);

  infer_k_split_forest_shared_data<<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params, trees_per_sm, data_per_sm, temp_out);

  int* h_offsets=(int*)malloc(sizeof(int)*(params.num_rows+1));
  for(int i=0;i<=params.num_rows;i++)
	  h_offsets[i]=i*num_blocks;

  int* d_offsets;
  allocate(d_offsets, (params.num_rows+1));
  updateDevice(d_offsets, h_offsets, (params.num_rows+1), stream);

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, temp_out, params.preds, params.num_rows, d_offsets, d_offsets + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sum-reduction
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, temp_out, params.preds, params.num_rows, d_offsets, d_offsets + 1);

  CUDA_CHECK(cudaFree(d_temp_storage));

  free(h_offsets);
  CUDA_CHECK(cudaFree(d_offsets));

  CUDA_CHECK(cudaFree(temp_out));

  //printf("end infer_k\n");
  CUDA_CHECK(cudaPeekAtLastError());
}







#endif

