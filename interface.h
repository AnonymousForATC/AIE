#ifndef STRUCT_H
#define STRUCT_H
#include "fil_algorithms.h"
#include "aie_algorithms.h"
#include "simhash.h"

int adaptive_format_number = 0;


struct dense_forest : forest {

  void transform_trees(const dense_node_t* nodes) {
    // populate node information
    for (int i = 0, gid = 0; i < num_trees_; ++i) {
      for (int j = 0, nid = 0; j <= depth_; ++j) {
        for (int k = 0; k < 1 << j; ++k, ++nid, ++gid) {
          h_nodes_[nid * num_trees_ + i] = nodes[gid];
        }
      }
    }
  } 

  void init(cudaStream_t stream, const dense_node_t* nodes,
            const forest_params_t* params) {
    init_common(params);
    int num_nodes = forest_num_nodes(num_trees_, depth_);
    allocate(nodes_, num_nodes);
    h_nodes_.resize(num_nodes);
    if (algo_ == algo_t::NAIVE) {
      std::copy(nodes, nodes + num_nodes, h_nodes_.begin());
    } else {
      transform_trees(nodes);
    }

    CUDA_CHECK(cudaMemcpyAsync(nodes_, h_nodes_.data(), num_nodes * sizeof(dense_node_t), cudaMemcpyHostToDevice, stream));
    // copy must be finished before freeing the host data
    CUDA_CHECK(cudaStreamSynchronize(stream));
    h_nodes_.clear();
    h_nodes_.shrink_to_fit();

  }

  virtual void infer(predict_params params, cudaStream_t stream) override {
    dense_storage forest(nodes_, num_trees_,
                         algo_ == algo_t::NAIVE ? tree_num_nodes(depth_) : 1,
                         algo_ == algo_t::NAIVE ? 1 : num_trees_);
	if(params.strategy == strategy_t::SHARED_DATA)
	{
		infer_dense_shared_data(forest, params, stream);
	}
	else if(params.strategy == strategy_t::SHARED_FOREST)
	{
		infer_dense_shared_forest(forest, params, stream);
	}
	else if(params.strategy == strategy_t::SPLIT_FOREST)
	{
		infer_dense_split_forest(forest, params, stream);
	}
	else if(params.strategy == strategy_t::SPLIT_FOREST_SHARED_DATA)
	{
		infer_dense_split_forest_shared_data(forest, params, stream);
	}

  } 

  dense_node_t* nodes_ = nullptr;
  thrust::host_vector<dense_node_t> h_nodes_;

};





struct dense_adaptive_forest : forest {

void swap_child(int k, int depth, std::vector<float> &values_h, std::vector<float> &weights_h, std::vector<int> &fids_h, bool* defaults_h, bool* is_leafs_h, bool* exchanges_h, int tree_offset)
{

	int offset = pow(2,depth);
	float temp_value, temp_weight;
	int temp_fid;
	bool temp_default, temp_is_leaf, temp_exchange;

	temp_value = values_h[k + offset+tree_offset];
	values_h[k + offset+tree_offset] = values_h[k+tree_offset];
	values_h[k+tree_offset] = temp_value;

	temp_weight = weights_h[k + offset+tree_offset];
	weights_h[k + offset+tree_offset] = weights_h[k+tree_offset];
	weights_h[k+tree_offset] = temp_weight;

	temp_fid = fids_h[k + offset+tree_offset];
	fids_h[k + offset+tree_offset] = fids_h[k+tree_offset];
	fids_h[k+tree_offset] = temp_fid;

	temp_default = defaults_h[k + offset+tree_offset];
	defaults_h[k + offset+tree_offset] = defaults_h[k+tree_offset];
	defaults_h[k+tree_offset] = temp_default;

	temp_is_leaf = is_leafs_h[k + offset+tree_offset];
	is_leafs_h[k + offset+tree_offset] = is_leafs_h[k+tree_offset];
	is_leafs_h[k+tree_offset] = temp_is_leaf;
	
	temp_exchange = exchanges_h[k + offset+tree_offset];
	exchanges_h[k + offset+tree_offset] = exchanges_h[k+tree_offset];
	exchanges_h[k+tree_offset] = temp_exchange;

	if((k*2+1)<tree_num_nodes(depth_))
	{
	swap_child((k*2+1), depth+1, values_h, weights_h, fids_h, defaults_h, is_leafs_h, exchanges_h, tree_offset);
	swap_child((k*2+2), depth+1, values_h, weights_h, fids_h, defaults_h, is_leafs_h, exchanges_h, tree_offset);
	}

}





  void init(cudaStream_t stream, const dense_node_t* nodes,
            const forest_params_t* params) {
    init_common(params);
    int num_nodes = forest_num_nodes(num_trees_, depth_);
    int num_nodes_per_tree = tree_num_nodes(depth_);
    std::vector<float> values_h(num_nodes), weights_h(num_nodes);
    std::vector<int> fids_h(num_nodes);
    bool* defaults_h = new bool[num_nodes];
    bool* is_leafs_h = new bool[num_nodes];
    bool* exchanges_h = new bool[num_nodes];

    for (size_t i = 0; i < num_trees_; ++i) {
        for (size_t j = 0; j < num_nodes_per_tree; ++j) {
		dense_node_decode(&nodes[i*num_nodes_per_tree + j], &values_h[i*num_nodes_per_tree + j], &weights_h[i*num_nodes_per_tree + j],
				&fids_h[i*num_nodes_per_tree + j], &defaults_h[i*num_nodes_per_tree + j], &is_leafs_h[i*num_nodes_per_tree + j]);
		exchanges_h[i*num_nodes_per_tree + j] = 0;
	}
    }

    for (size_t i = 0; i < num_trees_; ++i) {
		int tree_offset = i*num_nodes_per_tree;
        for (int j = (depth_-1); j >= 0; --j) {
	    for(int k = (pow(2,j)-1); k < (pow(2,j+1)-1); k++) {
			if(is_leafs_h[k+tree_offset]==0)
			{
               float left_weight = weights_h[k*2+1+tree_offset];
               float right_weight = weights_h[k*2+2+tree_offset];
			   if(left_weight<right_weight)
			   {
				    exchanges_h[k+tree_offset] = 1;

					float temp_value, temp_weight;
					int temp_fid;
					bool temp_default, temp_is_leaf, temp_exchange;

					temp_value = values_h[k*2+1+tree_offset];
					values_h[k*2+1+tree_offset] = values_h[k*2+2+tree_offset];
					values_h[k*2+2+tree_offset] = temp_value;

					temp_weight = weights_h[k*2+1+tree_offset];
					weights_h[k*2+1+tree_offset] = weights_h[k*2+2+tree_offset];
					weights_h[k*2+2+tree_offset] = temp_weight;

					temp_fid = fids_h[k*2+1+tree_offset];
					fids_h[k*2+1+tree_offset] = fids_h[k*2+2+tree_offset];
					fids_h[k*2+2+tree_offset] = temp_fid;
	
					temp_default = defaults_h[k*2+1+tree_offset];
					defaults_h[k*2+1+tree_offset] = defaults_h[k*2+2+tree_offset];
					defaults_h[k*2+2+tree_offset] = temp_default;
	
					temp_is_leaf = is_leafs_h[k*2+1+tree_offset];
					is_leafs_h[k*2+1+tree_offset] = is_leafs_h[k*2+2+tree_offset];
					is_leafs_h[k*2+2+tree_offset] = temp_is_leaf;

					temp_exchange = exchanges_h[k*2+1+tree_offset];
					exchanges_h[k*2+1+tree_offset] = exchanges_h[k*2+2+tree_offset];
					exchanges_h[k*2+2+tree_offset] = temp_exchange;


					if(((k*2+1)*2+1)<tree_num_nodes(depth_))
					{
						swap_child((k*2+1)*2+1, 1, values_h, weights_h, fids_h, defaults_h, is_leafs_h, exchanges_h, tree_offset);
						swap_child((k*2+1)*2+2, 1, values_h, weights_h, fids_h, defaults_h, is_leafs_h, exchanges_h, tree_offset);
					}
			   }
			}
	}
    }
    }

	int max_fid = 0;
	for(int i=0; i<num_nodes; i++)
		max_fid = (max_fid < fids_h[i])?fids_h[i]:max_fid;

	float fid_len = (log(max_fid)/log(2) + 3 )/8; //+3 is for other bits

	if(fid_len < 1.0 && fid_len > 0.0)
	{
		bits_length = 1;
		adaptive_format_number = 1;
	}
	else if(fid_len < 2.0 && fid_len >= 1.0)
	{
		bits_length = 2;
		adaptive_format_number = 2;
	}
	else if(fid_len < 4.0 && fid_len >= 2.0)
	{
		bits_length = 4;
		adaptive_format_number = 4;
	}
	else
	{
		bits_length = -1;
		adaptive_format_number = -1;
	}

	char ***test = new char**[num_trees_];

	for(int i=0; i<num_trees_; i++)
	{
		test[i] = new char*[num_nodes_per_tree-2];
		for(int j=0; j<num_nodes_per_tree-2; j++)
		{
			test[i][j] = new char[10];
		}
	}


    std::vector<std::pair<ul_int,int>> hash(num_trees_);
	for(int i=0; i<num_trees_; i++)
	{
		hash[i] = std::make_pair(sh_simhash(test[i], num_nodes_per_tree-2), i);
	}

	std::sort(hash.begin(),hash.end());

    std::vector<float> values_reorder_h(num_nodes), weights_reorder_h(num_nodes);
    std::vector<int> fids_reorder_h(num_nodes);
    bool* defaults_reorder_h = new bool[num_nodes];
    bool* is_leafs_reorder_h = new bool[num_nodes];
    bool* exchanges_reorder_h = new bool[num_nodes];

	for(int i=0; i<num_trees_; i++)
	{
		for(int j=0; j<num_nodes_per_tree; j++)
		{
			values_reorder_h[i*num_nodes_per_tree+j] = values_h[(hash[i].second)*num_nodes_per_tree+j];
			weights_reorder_h[i*num_nodes_per_tree+j] = weights_h[(hash[i].second)*num_nodes_per_tree+j];
			fids_reorder_h[i*num_nodes_per_tree+j] = fids_h[(hash[i].second)*num_nodes_per_tree+j];
			defaults_reorder_h[i*num_nodes_per_tree+j] = defaults_h[(hash[i].second)*num_nodes_per_tree+j];
			is_leafs_reorder_h[i*num_nodes_per_tree+j] = is_leafs_h[(hash[i].second)*num_nodes_per_tree+j];
			exchanges_reorder_h[i*num_nodes_per_tree+j] = exchanges_h[(hash[i].second)*num_nodes_per_tree+j];
		}
	}
	
	bits_values_h = new float[num_nodes];
	if(bits_length == 1)
		bits_char_h = new char[num_nodes];
	if(bits_length == 2)
		bits_short_h = new short int[num_nodes];
	if(bits_length == 4)
		bits_int_h = new int[num_nodes];

	encode_node_adaptive(values_reorder_h, fids_reorder_h, defaults_reorder_h, is_leafs_reorder_h, exchanges_reorder_h, bits_values_h, bits_char_h, bits_short_h, bits_int_h, bits_length, num_nodes);

	bits_values_reorg_h = new float[num_nodes];
	if(bits_length == 1)
		bits_char_reorg_h = new char[num_nodes];
	if(bits_length == 2)
		bits_short_reorg_h = new short int[num_nodes];
	if(bits_length == 4)
		bits_int_reorg_h = new int[num_nodes];

	for(int j=0; j<num_nodes_per_tree; j++)
	{
		for(int i=0; i<num_trees_; i++)
		{
			bits_values_reorg_h[j*num_trees_ + i] = bits_values_h[i*num_nodes_per_tree + j];
			if(bits_length == 1)
				bits_char_reorg_h[j*num_trees_ + i] = bits_char_h[i*num_nodes_per_tree + j];
			if(bits_length == 2)
				bits_short_reorg_h[j*num_trees_ + i] = bits_short_h[i*num_nodes_per_tree + j];
			if(bits_length == 4)
				bits_int_reorg_h[j*num_trees_ + i] = bits_int_h[i*num_nodes_per_tree + j];
		}
	}

	//printf_float_CPU(bits_values_h, num_nodes);
	//printf_float_CPU(bits_values_org_h, num_nodes);

	allocate(bits_values_d, num_nodes);
	if(bits_length == 1)
		allocate(bits_char_d, num_nodes);
	if(bits_length == 2)
		allocate(bits_short_d, num_nodes);
	if(bits_length == 4)
		allocate(bits_int_d, num_nodes);

	updateDevice(bits_values_d, bits_values_h, num_nodes, stream);	
	if(bits_length == 1)
		updateDevice(bits_char_d, bits_char_h, num_nodes, stream);	
	if(bits_length == 2)
		updateDevice(bits_short_d, bits_short_h, num_nodes, stream);	
	if(bits_length == 4)
		updateDevice(bits_int_d, bits_int_h, num_nodes, stream);	



	allocate(bits_values_reorg_d, num_nodes);
	if(bits_length == 1)
		allocate(bits_char_reorg_d, num_nodes);
	if(bits_length == 2)
		allocate(bits_short_reorg_d, num_nodes);
	if(bits_length == 4)
		allocate(bits_int_reorg_d, num_nodes);

	updateDevice(bits_values_reorg_d, bits_values_reorg_h, num_nodes, stream);	
	if(bits_length == 1)
		updateDevice(bits_char_reorg_d, bits_char_reorg_h, num_nodes, stream);	
	if(bits_length == 2)
		updateDevice(bits_short_reorg_d, bits_short_reorg_h, num_nodes, stream);	
	if(bits_length == 4)
		updateDevice(bits_int_reorg_d, bits_int_reorg_h, num_nodes, stream);	


    values_h.clear();
    weights_h.clear();
    fids_h.clear();
    delete[] defaults_h;
    delete[] is_leafs_h;
    delete[] exchanges_h;

    values_reorder_h.clear();
    weights_reorder_h.clear();
    fids_reorder_h.clear();
    delete[] defaults_reorder_h;
    delete[] is_leafs_reorder_h;
    delete[] exchanges_reorder_h;

	for(int i=0; i<num_trees_; i++)
	{
		for(int j=0; j<num_nodes_per_tree-2; j++)
			delete test[i][j];
		delete test[i];
	}
	delete test;


  }





  void infer_adaptive(predict_params params, cudaStream_t stream) {
  int num_blocks = ceildiv(int(params.num_rows), FIL_TPB);

  if(adaptive_format_number == 1)
	infer_adaptive_reorg_char<1><<<num_blocks, FIL_TPB, 0, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_char_reorg_d);
  if(adaptive_format_number == 2)
	infer_adaptive_reorg_short<1><<<num_blocks, FIL_TPB, 0, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_short_reorg_d);
  if(adaptive_format_number == 4)
	infer_adaptive_reorg_int<1><<<num_blocks, FIL_TPB, 0, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_int_reorg_d);
  CUDA_CHECK(cudaPeekAtLastError());
}



  void infer_dense_shared_data_adaptive(predict_params params, cudaStream_t stream) {
  int num_items = params.max_shm / (sizeof(float) * params.num_cols) ;
  if (num_items == 0) {
    assert(false && "too many features");
  }
  num_items = 1;
  int num_blocks = ceildiv(int(params.num_rows), num_items);
  int shm_sz = num_items * sizeof(float) * params.num_cols;

  if(adaptive_format_number == 1)
	infer_k_shared_data_adaptive_reorg_char<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_char_reorg_d);
  if(adaptive_format_number == 2)
	infer_k_shared_data_adaptive_reorg_short<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_short_reorg_d);
  if(adaptive_format_number == 4)
	infer_k_shared_data_adaptive_reorg_int<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_int_reorg_d);

  CUDA_CHECK(cudaPeekAtLastError());
}



  void infer_dense_shared_data_wo_adaptive(predict_params params, cudaStream_t stream) {
  int num_items = params.max_shm / (sizeof(float) * params.num_cols) ;
  if (num_items == 0) {
    assert(false && "too many features");
  }
  int num_blocks = ceildiv(int(params.num_rows), num_items);
  int shm_sz = num_items * sizeof(float) * params.num_cols;
  //printf("num_items %d, num_blocks %d\n", num_items, num_blocks);

  if(adaptive_format_number == 1)
	infer_k_shared_data_wo_adaptive_reorg_char<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_char_reorg_d, num_items);
  if(adaptive_format_number == 2)
	infer_k_shared_data_wo_adaptive_reorg_short<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_short_reorg_d, num_items);
  if(adaptive_format_number == 4)
	infer_k_shared_data_wo_adaptive_reorg_int<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_int_reorg_d, num_items);

  CUDA_CHECK(cudaPeekAtLastError());
}



  void infer_dense_shared_forest_adaptive(predict_params params, cudaStream_t stream) {
  if(adaptive_format_number == 1)
  {
	  int shm_sz = num_trees_ * (sizeof(char)+sizeof(float)) * tree_num_nodes(depth_);
	  //printf("shared memory is %d\n", params.max_shm);
	  if (shm_sz > params.max_shm) {
	    assert(false && "forest is too large to save in shared memory");
	  }
	
	  int num_blocks = ceildiv(int(params.num_rows), FIL_TPB);
	  //printf("%d - %d\n", num_blocks, FIL_TPB);
	  infer_k_shared_forest_adaptive_reorg_char<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_char_reorg_d);
	  //printf("end infer_k\n");
	  CUDA_CHECK(cudaPeekAtLastError());
  }
  if(adaptive_format_number == 2)
  {
	  int shm_sz = num_trees_ * (sizeof(short)+sizeof(float)) * tree_num_nodes(depth_);
	  //printf("shared memory is %d\n", params.max_shm);
	  if (shm_sz > params.max_shm) {
	    assert(false && "forest is too large to save in shared memory");
	  }
	
	  int num_blocks = ceildiv(int(params.num_rows), FIL_TPB);
	  //printf("%d - %d\n", num_blocks, FIL_TPB);
	  infer_k_shared_forest_adaptive_reorg_short<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_short_reorg_d);
	  //printf("end infer_k\n");
	  CUDA_CHECK(cudaPeekAtLastError());
  }
  if(adaptive_format_number == 4)
  {
	  int shm_sz = num_trees_ * (sizeof(int)+sizeof(float)) * tree_num_nodes(depth_);
	  //printf("shared memory is %d\n", params.max_shm);
	  if (shm_sz > params.max_shm) {
	    assert(false && "forest is too large to save in shared memory");
	  }
	
	  int num_blocks = ceildiv(int(params.num_rows), FIL_TPB);
	  //printf("%d - %d\n", num_blocks, FIL_TPB);
	  infer_k_shared_forest_adaptive_reorg_int<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_int_reorg_d);
	  //printf("end infer_k\n");
	  CUDA_CHECK(cudaPeekAtLastError());
  }


}


void infer_dense_split_forest_adaptive(predict_params params, cudaStream_t stream) {
  int trees_per_sm = 0;
  if(adaptive_format_number == 1)
	trees_per_sm = params.max_shm / ( (sizeof(char)+sizeof(float)) * tree_num_nodes(depth_) );
  if(adaptive_format_number == 2)
	trees_per_sm = params.max_shm / ( (sizeof(short)+sizeof(float)) * tree_num_nodes(depth_) );
  if(adaptive_format_number == 4)
	trees_per_sm = params.max_shm / ( (sizeof(int)+sizeof(float)) * tree_num_nodes(depth_) );
  //printf("shared memory is %d\n", params.max_shm);
  if (trees_per_sm == 0) {
    assert(false && "single tree is too big to save in shared memory");
  }
  int num_blocks = ceildiv(num_trees_, trees_per_sm);
  int shm_sz = 0;
  if(adaptive_format_number == 1)
	shm_sz = trees_per_sm * ( (sizeof(char)+sizeof(float)) * tree_num_nodes(depth_) );////////////////////////////DO NOT KNOW WHY!!!!!!!!!!!!!
  if(adaptive_format_number == 2)
	shm_sz = trees_per_sm * ( (sizeof(short)+sizeof(float)) * tree_num_nodes(depth_) );////////////////////////////DO NOT KNOW WHY!!!!!!!!!!!!!
  if(adaptive_format_number == 4)
	shm_sz = trees_per_sm * ( (sizeof(int)+sizeof(float)) * tree_num_nodes(depth_) );////////////////////////////DO NOT KNOW WHY!!!!!!!!!!!!!

  float* temp_out = NULL;
  allocate(temp_out, num_blocks*params.num_rows);

  if(adaptive_format_number == 1)
  infer_k_split_forest_adaptive_reorg_char<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, trees_per_sm, temp_out, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_char_reorg_d);
  if(adaptive_format_number == 2)
  infer_k_split_forest_adaptive_reorg_short<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, trees_per_sm, temp_out, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_short_reorg_d);
  if(adaptive_format_number == 4)
  infer_k_split_forest_adaptive_reorg_int<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, trees_per_sm, temp_out, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_int_reorg_d);


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



  virtual void infer(predict_params params, cudaStream_t stream) override {
		//infer_adaptive(params, stream);
		//infer_dense_shared_data_wo_adaptive(params, stream);
		//infer_dense_shared_data_adaptive(params, stream);
		//infer_dense_shared_forest_adaptive(params, stream);
		infer_dense_split_forest_adaptive(params, stream);
  } 

  float* bits_values_d = nullptr;
  int* bits_int_d = nullptr;
  short int* bits_short_d = nullptr;
  char* bits_char_d = nullptr;

  float* bits_values_h = nullptr;
  int* bits_int_h = nullptr;
  short int* bits_short_h = nullptr;
  char* bits_char_h = nullptr;


  float* bits_values_reorg_d = nullptr;
  int* bits_int_reorg_d = nullptr;
  short int* bits_short_reorg_d = nullptr;
  char* bits_char_reorg_d = nullptr;

  float* bits_values_reorg_h = nullptr;
  int* bits_int_reorg_h = nullptr;
  short int* bits_short_reorg_h = nullptr;
  char* bits_char_reorg_h = nullptr;


  int bits_length = 0;

};






template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_sparse(sparse_tree tree, float* sdata,
                                               int cols, vec<NITEMS>& out, algo_t algo_, int num_trees) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = tree.nodes_[curr[j]].val;
      int n_bits = tree.nodes_[curr[j]].bits;
	  //float n_output = n_val;
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 30) - 1);
	  bool n_def_left = n_bits & (1 << 30);
	  bool n_is_leaf = n_bits & (1 << 31);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      bool cond = isnan(val) ? !n_def_left : val >= n_thresh;
	  if(algo_ == algo_t::NAIVE)
	  {
		  //curr[j] = curr[j] + 1 + cond;
		  curr[j] = tree.nodes_[curr[j]].left_idx + cond;
	  }
    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += tree.nodes_[curr[j]].val;
}


template <int NITEMS>
__global__ void infer_k(sparse_storage forest, predict_params params) {
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
    infer_one_tree_sparse<NITEMS>(forest[j], sdata, params.num_cols, out, params.algo, forest.num_trees());
  }

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


void infer_sparse(sparse_storage forest, predict_params params, cudaStream_t stream) {
  const int MAX_BATCH_ITEMS = 4;
  params.max_items =
    params.algo == algo_t::BATCH_TREE_REORG ? MAX_BATCH_ITEMS : 1;
  int num_items = params.max_shm / (sizeof(float) * params.num_cols);
  //printf("shared memory is %d\n", params.max_shm);
  if (num_items == 0) {
    //int max_cols = params.max_shm / sizeof(float);
    assert(false && "too many features");
  }
  num_items = std::min(num_items, params.max_items);
  int num_blocks = ceildiv(int(params.num_rows), num_items);
  int shm_sz = num_items * sizeof(float) * params.num_cols;
  //printf("start infer_k: %d, num_blocks: %d, FIL_TPB: %d, shm_sz: %d, stream: %d\n", num_items, num_blocks, FIL_TPB, shm_sz, stream);
  switch (num_items) {
    case 1:
      infer_k<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 2:
      infer_k<2><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 3:
      infer_k<3><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 4:
      infer_k<4><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    default:
      assert(false && "internal error: nitems > 4");
  }
  //printf("end infer_k\n");
  CUDA_CHECK(cudaPeekAtLastError());
}


struct sparse_forest : forest {

  void init(const cudaStream_t stream, const int* trees, const sparse_node_t* nodes,
            const forest_params_t* params) {
    init_common(params);
    depth_ = 0;  // a placeholder value
    num_nodes_ = params->num_nodes;

    // trees
    allocate(trees_, num_trees_);
    CUDA_CHECK(cudaMemcpyAsync(trees_, trees, sizeof(int) * num_trees_, cudaMemcpyHostToDevice, stream));

    // nodes
	allocate(nodes_, num_nodes_);
    CUDA_CHECK(cudaMemcpyAsync(nodes_, nodes, sizeof(sparse_node_t) * num_nodes_, cudaMemcpyHostToDevice, stream));
  } 

  virtual void infer(predict_params params, cudaStream_t stream) override {
    sparse_storage forest(trees_, nodes_, num_trees_);
    infer_sparse(forest, params, stream);
  } 
 
  int num_nodes_ = 0;
  int* trees_ = nullptr;
  sparse_node_t* nodes_ = nullptr;

};


#endif
