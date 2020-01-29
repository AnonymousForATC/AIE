#ifndef FIL_DATA_STRUCT_H
#define FIL_DATA_STRUCT_H

enum algo_t {
  NAIVE,
  TREE_REORG,
  BATCH_TREE_REORG
};

enum strategy_t {
  SHARED_DATA,
  SHARED_FOREST,
  SPLIT_FOREST,
  SPLIT_FOREST_SHARED_DATA
};


enum output_t {
  RAW = 0x0,
  AVG = 0x1,
  SIGMOID = 0x10,
  THRESHOLD = 0x100,
};

struct dense_node_t {
  float weight;
  float val;
  int bits;
};

struct sparse_node_t {
  float val;
  int bits;
  int left_idx;
};


static const int FID_MASK = (1 << 30) - 1;
static const int DEF_LEFT_MASK = 1 << 30;
static const int IS_LEAF_MASK = 1 << 31;

static const int FID_MASK_INT = (1 << 29) - 1;
static const int DEF_LEFT_MASK_INT = 1 << 29;
static const int IS_LEAF_MASK_INT = 1 << 30;
static const int EXCHANGE_MASK_INT = 1 << 31;

static const int FID_MASK_SHORT = (1 << 13) - 1;
static const int DEF_LEFT_MASK_SHORT = 1 << 13;
static const int IS_LEAF_MASK_SHORT = 1 << 14;
static const int EXCHANGE_MASK_SHORT = 1 << 15;

static const int FID_MASK_CHAR = (1 << 5) - 1;
static const int DEF_LEFT_MASK_CHAR = 1 << 5;
static const int IS_LEAF_MASK_CHAR = 1 << 6;
static const int EXCHANGE_MASK_CHAR = 1 << 7;


__host__ __device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__host__ __device__ __forceinline__ int tree_num_nodes(int depth) {
	  return (1 << (depth + 1)) - 1;
}

__host__ __device__ __forceinline__ int forest_num_nodes(int num_trees, int depth) {
  return num_trees * tree_num_nodes(depth);
}


struct FilTestParams {
  // input data parameters
  int num_rows;
  int num_cols;
  float nan_prob;
  // forest parameters
  int depth;
  int num_trees;
  float leaf_prob;
  // output parameters
  output_t output;
  float threshold;
  float global_bias;
  // runtime parameters
  algo_t algo;
  int seed;
  float tolerance;
  strategy_t strategy;

  char input_model_file[1024];
  char input_data_file[1024];
  float missing;
};


// predict_params are parameters for prediction
struct predict_params {
  // Model parameters.
  int num_cols;
  algo_t algo;
  strategy_t strategy;
  int max_items;  // only set and used by infer()

  // Data parameters.
  float* preds;
  const float* data;
  size_t num_rows;

  // Other parameters.
  int max_shm;

  float missing;
};


/** forest_params_t are the trees to initialize the predictor */
struct forest_params_t {
  // total number of nodes; ignored for dense forests
  int num_nodes;
  // maximum depth; ignored for sparse forests
  int depth;
  // ntrees is the number of trees
  int num_trees;
  // num_cols is the number of columns in the data
  int num_cols;
  // algo is the inference algorithm;
  // sparse forests do not distinguish between NAIVE and TREE_REORG
  algo_t algo;
  // output is the desired output type
  output_t output;
  // threshold is used to for classification if output == OUTPUT_CLASS,
  // and is ignored otherwise
  float threshold;
  // global_bias is added to the sum of tree predictions
  // (after averaging, if it is used, but before any further transformations)
  float global_bias;
  strategy_t strategy;

  float missing;
};


/** performs additional transformations on the array of forest predictions
    (preds) of size n; the transformations are defined by output, and include
    averaging (multiplying by inv_num_trees), adding global_bias (always done),
    sigmoid and applying threshold */
__global__ void transform_k(float* preds, size_t n, output_t output,
                            float inv_num_trees, float threshold,
                            float global_bias) {
  size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
  if (i >= n) return;
  float result = preds[i];
  if ((output & output_t::AVG) != 0) result *= inv_num_trees;
  result += global_bias;
  if ((output & output_t::SIGMOID) != 0) result = sigmoid(result);
  if ((output & output_t::THRESHOLD) != 0) {
    result = result > threshold ? 1.0f : 0.0f;
  }
  preds[i] = result;
}


struct forest {


   void init_max_shm() {
    int device = 0;
	cudaDeviceProp dev;
	cudaGetDeviceProperties(&dev, device);
	//printf( "Shared mem per mp: %d\n", (int)dev.sharedMemPerBlock );
	max_shm_ = (int)dev.sharedMemPerBlock * 0.8;
	/*
    // TODO(canonizer): use cumlHandle for this
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shm_, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    // TODO(canonizer): use >48KiB shared memory if available
	*/
  } 

  void init_common(const forest_params_t* params) {
    depth_ = params->depth;
    num_trees_ = params->num_trees;
    num_cols_ = params->num_cols;
    algo_ = params->algo;
    strategy_ = params->strategy;
    output_ = params->output;
    threshold_ = params->threshold;
    global_bias_ = params->global_bias;
    missing_ = params->missing;
    init_max_shm();
  }

  virtual void infer(predict_params params, cudaStream_t stream) = 0;

  void predict(cudaStream_t stream, float* preds, const float* data,
               size_t num_rows) {
    // Initialize prediction parameters.
    predict_params params;
    params.num_cols = num_cols_;
    params.algo = algo_;
    params.strategy = strategy_;
    params.preds = preds;
    params.data = data;
    params.num_rows = num_rows;
    params.max_shm = max_shm_;
    params.missing = missing_;

    // Predict using the forest.
    infer(params, stream);
	//printf("infer\n");

    // Transform the output if necessary.
    if (output_ != output_t::RAW || global_bias_ != 0.0f) {
      transform_k<<<ceildiv(int(num_rows), FIL_TPB), FIL_TPB, 0, stream>>>(
        preds, num_rows, output_, num_trees_ > 0 ? (1.0f / num_trees_) : 1.0f,
        threshold_, global_bias_);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  } 

  int num_trees_ = 0;
  int depth_ = 0;
  int num_cols_ = 0;
  algo_t algo_ = algo_t::NAIVE;
  strategy_t strategy_ = strategy_t::SHARED_DATA;
  int max_shm_ = 0;
  output_t output_ = output_t::RAW;
  float threshold_ = 0.5;
  float global_bias_ = 0;
  float missing_ = 0.0;
};



/** forest_t is the predictor handle */
typedef forest* forest_t;

// vec wraps float[N] for cub::BlockReduce
template <int N>
struct vec {
  float data[N];
  __host__ __device__ float& operator[](int i) { return data[i]; }
  __host__ __device__ float operator[](int i) const { return data[i]; }
  friend __host__ __device__ vec<N> operator+(const vec<N>& a,
                                              const vec<N>& b) {
    vec<N> r;
#pragma unroll
    for (int i = 0; i < N; ++i) r[i] = a[i] + b[i];
    return r;
  }
};


/** dense_tree represents a dense tree */
struct dense_tree {
  __host__ __device__ dense_tree(dense_node_t* nodes, int node_pitch)
    : nodes_(nodes), node_pitch_(node_pitch) {}
  __host__ __device__ const dense_node_t& operator[](int i) const {
    return nodes_[i * node_pitch_];
  }
  dense_node_t* nodes_ = nullptr;
  int node_pitch_ = 0;
};


/** dense_storage stores the forest as a collection of dense nodes */
struct dense_storage {
  __host__ __device__ dense_storage(dense_node_t* nodes, int num_trees,
                                    int tree_stride, int node_pitch)
    : nodes_(nodes),
      num_trees_(num_trees),
      tree_stride_(tree_stride),
      node_pitch_(node_pitch) {}
  __host__ __device__ int num_trees() const { return num_trees_; }
  __host__ __device__ dense_tree operator[](int i) const {
    return dense_tree(nodes_ + i * tree_stride_, node_pitch_);
  }
  dense_node_t* nodes_ = nullptr;
  int num_trees_ = 0;
  int tree_stride_ = 0;
  int node_pitch_ = 0;
};

/** sparse_tree is a sparse tree */
struct sparse_tree {
  __host__ __device__ sparse_tree(sparse_node_t* nodes) : nodes_(nodes) {}
  __host__ __device__ const sparse_node_t& operator[](int i) const {
    return nodes_[i];
  }
  sparse_node_t* nodes_ = nullptr;
};

/** sparse_storage stores the forest as a collection of sparse nodes */
struct sparse_storage {
  int* trees_ = nullptr;
  sparse_node_t* nodes_ = nullptr;
  int num_trees_ = 0;
  __host__ __device__ sparse_storage(int* trees, sparse_node_t* nodes,
                                     int num_trees)
    : trees_(trees), nodes_(nodes), num_trees_(num_trees) {}
  __host__ __device__ int num_trees() const { return num_trees_; }
  __host__ __device__ sparse_tree operator[](int i) const {
    return sparse_tree(&nodes_[trees_[i]]);
  }
};




#endif

