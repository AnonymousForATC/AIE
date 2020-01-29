#include "iostream"
#include "BaseFilTest.h"
using namespace std;

int main(int argc,char *argv[])
{
	// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold,
	// global_bias, algo, seed, tolerance

	printf("%s\n", argv[1]);
	BaseFilTest* pTest = new BaseFilTest(1000, 500, (float)0.0, 20, 10, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_DATA, argv[1], argv[2]);
	pTest->SetUp();
	pTest->Free();

	return 0;
}
