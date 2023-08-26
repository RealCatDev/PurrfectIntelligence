#include <PurrfectIntelligence.h>

using namespace PurrfectIntelligence;

int main() {
	NeuralNetworkArchitecture arch{};
	arch.inputLayerNeurons = 784;
	arch.hiddenLayersNeurons = { 300 };
	arch.outputLayerNeurons = 10;

	std::unique_ptr<NeuralNetowrk> NN = std::make_unique<NeuralNetowrk>(arch);

	// TODO(CatDev): Create helper function that will load csv into vector<Data>
	std::vector<Data> data = Data::LoadCsv("resources/mnist_train.csv", 10000, 28, 28, 10);

	NN->Train(data, .1);

	auto result = NN->Classify(data[18]);
	printf("%zu = %zu\n", data[18].GetLabel(), result.first);

	return 0;
}
