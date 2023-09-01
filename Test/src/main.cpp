#include <PurrfectIntelligence.h>

#include <time.h>

using namespace PurrfectIntelligence;

#include <assert.h>

#include <filesystem>

int main(void) {
	srand(time(0));

	NeuralNetwork* nn;

	if (!std::filesystem::exists("resources/xor.purrnn")) {
		Math::Matrix* td = new Math::Matrix(4, 3);

		for (size_t i = 0; i < 2; ++i)
			for (size_t j = 0; j < 2; ++j) {
				size_t row = i * 2 + j;
				td->Get(row, 0) = i;
				td->Get(row, 1) = j;
				td->Get(row, 2) = i ^ j;
			}

		NNArchitecture arch{};
		arch.arch = std::vector<uint32_t>({ 2, 2, 1 });
		nn = new NeuralNetwork(arch);

		float eps = 1e-1;
		float rate = 1e-1;

		printf("Cost = %f\n", nn->Cost(td));
		for (size_t i = 0; i < 20 * 1000; ++i) {
			nn->FiniteDiff(eps, td);
			nn->Learn(rate);
			printf("%zu: Cost = %f\n", i, nn->Cost(td));
		}

		nn->Save("resources/xor.purrnn");
	}
	else {
		nn = NeuralNetwork::Load("resources/xor.purrnn");
	}

	printf("\nVERIFICATION\n");
	for (size_t i = 0; i < 2; ++i)
		for (size_t j = 0; j < 2; ++j) {
			nn->Input().Get(0, 0) = i;
			nn->Input().Get(0, 1) = j;
			nn->Forward();
			printf("\n%zu ^ %zu = %f", i, j, nn->Output().Get(0, 0));
		}

	return 0;
}
