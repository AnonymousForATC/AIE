#include "iostream"
#include "BaseFilTest.h"
using namespace std;

int main(int argc,char *argv[])
{
	// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold,
	// global_bias, algo, seed, tolerance

	printf("%s\n", argv[1]);
	BaseFilTest* pTest = new BaseFilTest(1000, 500, (float)0.0, 20, 10, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_DATA, argv[1], argv[2]);
	//BaseFilTest* pTest = new BaseFilTest(1000, 500, (float)0.0, 20, 10, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_DATA, "final_model1_3_3.txt", "data_model1_2_30.txt");
	//BaseFilTest* pTest = new BaseFilTest(1000, 500, (float)0.0, 40, 1024, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::BATCH_TREE_REORG, 0, (float)1e-3f, strategy_t::SHARED_DATA);
	//BaseFilTest* pTest = new BaseFilTest(10000, 5000, (float)0.0, 5, 5, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_FOREST);
	//BaseFilTest* pTest = new BaseFilTest(10000, 500, (float)0.0, 5, 10000, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_DATA);
	//BaseFilTest* pTest = new BaseFilTest(10000, 500, (float)0.0, 5, 10000, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SPLIT_FOREST);

	//BaseFilTest* pTest = new BaseFilTest(10000, 500, (float)0.0, 8, 10000, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_DATA);
	//BaseFilTest* pTest = new BaseFilTest(10000, 500, (float)0.0, 8, 10000, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::BATCH_TREE_REORG, 0, (float)1e-3f, strategy_t::SHARED_DATA);
	//BaseFilTest* pTest = new BaseFilTest(10000, 500, (float)0.0, 5, 10000, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_FOREST);
	//BaseFilTest* pTest = new BaseFilTest(10000, 500, (float)0.0, 8, 10000, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SPLIT_FOREST);
	//BaseFilTest* pTest = new BaseFilTest(10000, 5, (float)0.0, 3, 40, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_DATA);
	//BaseFilTest* pTest = new BaseFilTest(10000, 5, (float)0.0, 3, 40, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_FOREST);
	//BaseFilTest* pTest = new BaseFilTest(10000, 5, (float)0.0, 2, 400, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_DATA);
	pTest->SetUp();
	pTest->Free();
	/*
	int depth = 10;
	int num_trees = 3;

	for(int i=1; i<2; i++)
	{
	//for(int j=1; j<10; j++)
	{
	BaseFilTest* pTest1 = new BaseFilTest(10000*i, 500, (float)0.0, depth, num_trees, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_DATA);
	pTest1->SetUp();
	pTest1->Free();
	BaseFilTest* pTest2 = new BaseFilTest(10000*i, 500, (float)0.0, depth, num_trees, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_FOREST);
	pTest2->SetUp();
	pTest2->Free();
	}
	}
	*/

	return 0;
}
