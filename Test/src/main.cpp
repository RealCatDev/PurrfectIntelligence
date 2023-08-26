#include <PurrfectIntelligence.h>

using namespace PurrfectIntelligence;

int main() {
	NeuralNetworkArchitecture arch{};
	arch.inputLayerNeurons = 784;
	arch.hiddenLayersNeurons = { 300 };
	arch.outputLayerNeurons = 10;

	std::unique_ptr<NeuralNetowrk> NN = std::make_unique<NeuralNetowrk>(arch);

	// TODO(CatDev): Create helper function that will load csv into vector<Data>
	std::vector<Data> data;

	NN->Train(data, .1);

	return 0;
}
