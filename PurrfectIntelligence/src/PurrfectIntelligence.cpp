#include "PurrfectIntelligence.h"

#define MAXCHAR 10000

#define _USE_MATH_DEFINES
#include <math.h>

#include <algorithm>
#include <execution>

#include <random>

namespace PurrfectIntelligence {

	double NormalDistribution(double mean, double standardDeviation) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> distribution(mean, standardDeviation);
		return distribution(gen);
	}

	Layer::Layer()
		: m_Arch(), m_Activation(), m_ActivationDerivative(), m_Cost(), m_CostDerivative(), m_Weights(), m_Biases(), m_CostGradientW(), m_CostGradientB(), m_WeightVelocities(), m_BiasVelocities() {
	}

	Layer::Layer(LayerArchitecture arch, ActivationFunc activation, ActivationDerivativeFunc activationD, CostFunc cost, CostDerivativeFunc costD)
		: m_Arch(arch), m_Activation(activation), m_ActivationDerivative(activationD), m_Cost(cost), m_CostDerivative(costD), m_Weights(arch.neuronsIn* arch.neuronsOut), m_Biases(arch.neuronsOut), m_CostGradientW(m_Weights.size()), m_CostGradientB(m_Biases.size()), m_WeightVelocities(m_Weights.size()), m_BiasVelocities(m_Biases.size()) {
		for (uint32_t i = 0; i < m_Weights.size(); ++i)
			m_Weights[i] = NormalDistribution(0, 1) / sqrt(arch.neuronsOut);
	}

	Layer::~Layer() {

	}

	std::vector<double> Layer::Forward(std::vector<double> inputs) {
		std::vector<double> activations(m_Arch.neuronsOut);

		for (uint32_t i = 0; i < m_Arch.neuronsOut; ++i) {
			double weightedSum = m_Biases[i];
			for (int j = 0; j < m_Weights[i]; ++j) {
				weightedSum += m_Weights[i * m_Arch.neuronsIn + j] * inputs[j];
			}
			activations[i] = m_Activation(weightedSum);
		}

		return activations;
	}

	std::vector<double> Layer::Forward(std::vector<double> inputs, LayerLearnData &learnData) {
		learnData.inputs = inputs;

		for (uint32_t i = 0; i < m_Arch.neuronsOut; ++i) {
			double weightedSum = m_Biases[i];
			for (int j = 0; j < m_Arch.neuronsIn; ++j) {
				weightedSum += m_Weights[i * m_Arch.neuronsIn + j] * inputs[j];
			}
			learnData.weightedInputs[i] = weightedSum;
		}

		for (uint32_t i = 0; i < learnData.activations.size(); ++i)
			learnData.activations[i] = m_Activation(learnData.weightedInputs[i]);

		return learnData.activations;
	}

	void Layer::ApplyGradients(double learnRate, double regularization, double momentum)
	{
		double weightDecay = (1 - regularization * learnRate);

		for (uint32_t i = 0; i < m_Weights.size(); i++)
		{
			double weight = m_Weights[i];
			double velocity = m_WeightVelocities[i] * momentum - m_CostGradientW[i] * learnRate;
			m_WeightVelocities[i] = velocity;
			m_Weights[i] = weight * weightDecay + velocity;
			m_CostGradientW[i] = 0;
		}


		for (int i = 0; i < m_Biases.size(); i++)
		{
			double velocity = m_BiasVelocities[i] * momentum - m_CostGradientB[i] * learnRate;
			m_BiasVelocities[i] = velocity;
			m_Biases[i] += velocity;
			m_CostGradientB[i] = 0;
		}
	}

	void Layer::CalculateOutputLayerNodeValues(LayerLearnData &layerLearnData, std::vector<double> expectedOutputs, CostFunc cost)
	{
		for (int i = 0; i < layerLearnData.nodeValues.size(); i++)
		{
			// Evaluate partial derivatives for current node: cost/activation & activation/weightedInput
			double costDerivative = m_CostDerivative(layerLearnData.activations[i], expectedOutputs[i]);
			double activationDerivative = m_ActivationDerivative(layerLearnData.weightedInputs[i]);
			layerLearnData.nodeValues[i] = costDerivative * activationDerivative;
		}
	}

	void Layer::CalculateHiddenLayerNodeValues(LayerLearnData &layerLearnData, std::unique_ptr<Layer> &oldLayer, std::vector<double> oldNodeValues)
	{
		for (int newNodeIndex = 0; newNodeIndex < m_Arch.neuronsOut; newNodeIndex++)
		{
			double newNodeValue = 0;
			for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.size(); oldNodeIndex++)
			{
				// Partial derivative of the weighted input with respect to the input
				double weightedInputDerivative = oldLayer->m_Weights[oldNodeIndex * oldLayer->m_Arch.neuronsIn + newNodeIndex];
				newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
			}
			newNodeValue *= m_ActivationDerivative(layerLearnData.weightedInputs[newNodeIndex]);
			layerLearnData.nodeValues[newNodeIndex] = newNodeValue;
		}

	}

	void Layer::UpdateGradients(LayerLearnData &layerLearnData)
	{
		std::thread weight(
			[this, layerLearnData]() {
				for (int nodeOut = 0; nodeOut < m_Arch.neuronsOut; nodeOut++)
				{
					double nodeValue = layerLearnData.nodeValues[nodeOut];
					for (int nodeIn = 0; nodeIn < m_Arch.neuronsIn; nodeIn++)
					{
						double derivativeCostWrtWeight = layerLearnData.inputs[nodeIn] * nodeValue;
						m_CostGradientW[nodeOut * m_Arch.neuronsIn + nodeIn] += derivativeCostWrtWeight;
					}
				}
			}
		);

		std::thread bias(
			[this, layerLearnData]() {
				for (int nodeOut = 0; nodeOut < m_Arch.neuronsOut; nodeOut++) {
					double derivativeCostWrtBias = 1 * layerLearnData.nodeValues[nodeOut];
					m_CostGradientB[nodeOut] += derivativeCostWrtBias;
				}
			}
		);

		weight.join();
		bias.join();
	}

	std::vector<double> Layer::CalculateOutputs(std::vector<double> input) {
		std::vector<double> weightedInputs(m_Arch.neuronsOut);

		for (int nodeOut = 0; nodeOut < m_Arch.neuronsOut; nodeOut++)
		{
			double weightedInput = m_Biases[nodeOut];

			for (int nodeIn = 0; nodeIn < m_Arch.neuronsIn; nodeIn++)
			{
				weightedInput += input[nodeIn] * m_Weights[nodeOut * m_Arch.neuronsIn + nodeIn];
			}
			weightedInputs[nodeOut] = weightedInput;
		}

		// Apply activation function
		std::vector<double> activations(m_Arch.neuronsOut);
		for (int outputNode = 0; outputNode < m_Arch.neuronsOut; outputNode++)
		{
			activations[outputNode] = m_Activation(weightedInputs[outputNode]);
		}

		return activations;
	}

	Data::Data()
		: m_Data(), m_Label(), m_ExpectedOutputs() {
	}

	Data::Data(std::vector<double> data, uint32_t label, uint32_t labelCount)
		: m_Data(data), m_Label(label), m_ExpectedOutputs(CreateOneHot(label, labelCount)) {
	}

	Data::~Data() {

	}

	std::vector<double> Data::CreateOneHot(uint32_t index, uint32_t size) {
		std::vector<double> oneHot = std::vector<double>(size);
		for (uint32_t i = 0; i < size; ++i)
			oneHot[i] = i == index ? 1.0 : 0.0;
		return oneHot;
	}

	std::vector<Data> Data::LoadCsv(const char* filepath, uint32_t imageCount, uint32_t width, uint32_t height, uint32_t labelCount) {
		FILE* fp;
		std::vector<Data> images(imageCount);
		char row[MAXCHAR];
		fp = fopen(filepath, "r");

		// Read the first line 
		fgets(row, MAXCHAR, fp);
		int i = 0;
		while (feof(fp) != 1 && i < imageCount) {
			int j = 0;
			fgets(row, MAXCHAR, fp);
			char* token = strtok(row, ",");
			
			uint32_t label = 0;
			std::vector<double> imageData(width * height);
			
			while (token != NULL) {
				if (j == 0) {
					label = (uint32_t) atoi(token);
				}
				else {
					imageData[(j - 1) / width * width + (j - 1) % height] = atoi(token) / 256.0;
				}
				token = strtok(NULL, ",");
				j++;
			}
			
			images[i] = Data(imageData, label, labelCount);

			i++;
		}
		fclose(fp);
		return images;
	}

	NeuralNetowrk::NeuralNetowrk(NeuralNetworkArchitecture arch, ActivationFunc activation, ActivationDerivativeFunc activationD, CostFunc cost, CostDerivativeFunc costD)
		: m_Arch(arch), m_Activation(activation), m_ActivationDerivative(activationD), m_Cost(cost), m_CostDerivative(costD) {
		m_Layers.resize(arch.hiddenLayersNeurons.size()+1);
		for (uint32_t i = 0; i < arch.hiddenLayersNeurons.size()+1; ++i) {
			LayerArchitecture layerArch{};
			layerArch.neuronsIn = m_Arch[i];
			layerArch.neuronsOut = m_Arch[i+1];

			m_Layers[i] = std::make_unique<Layer>(layerArch, activation);
		}
	}

	NeuralNetowrk::~NeuralNetowrk() {

	}

	void NeuralNetowrk::Train(std::vector<Data> data, double learnRate, double regularization, double momentum) {
		if (m_BatchLearnData.empty() || m_BatchLearnData.size() != data.size()) {
			if (!m_BatchLearnData.empty()) m_BatchLearnData.clear();
			m_BatchLearnData.resize(data.size());
			for (uint32_t i = 0; i < data.size(); ++i)
				m_BatchLearnData[i] = NetworkLearnData(m_Layers);
		}

		uint32_t i = 0;
		std::for_each(/*std::execution::par, */data.begin(), data.end(), [this, &i](Data data) {
			printf("Train no. %zu.\n", i);

			std::vector<double> input = data.GetData();
			{
				uint32_t j = 0;
				for (auto& layer : m_Layers) {
					input = layer->Forward(input, m_BatchLearnData[i].layerData[j]);
					++j;
				}
			}

			uint32_t outputLayerIndex = m_Layers.size() - 1;
			auto &outputLayer = m_Layers[outputLayerIndex];
			LayerLearnData outputLearnData = m_BatchLearnData[i].layerData[outputLayerIndex];

			outputLayer->CalculateOutputLayerNodeValues(outputLearnData, data.GetExpectedOutputs(), m_Cost);
			outputLayer->UpdateGradients(outputLearnData);

			for (uint32_t j = outputLayerIndex - 1; j >= 0; j--)
			{
				if (j < 4294967295) break;
				LayerLearnData layerLearnData = m_BatchLearnData[i].layerData[j];
				auto &hiddenLayer = m_Layers[j];

				hiddenLayer->CalculateHiddenLayerNodeValues(layerLearnData, m_Layers[j + 1], m_BatchLearnData[i].layerData[j + 1].nodeValues);
				hiddenLayer->UpdateGradients(layerLearnData);
			}

			++i;
		});

		for (auto& layer : m_Layers)
			layer->ApplyGradients(learnRate / data.size(), regularization, momentum);
	}

	uint32_t MaxValueIndex(std::vector<double> values)
	{
		double maxValue = -DBL_MAX;
		int index = 0;
		for (int i = 0; i < values.size(); i++)
		{
			if (values[i] > maxValue)
			{
				maxValue = values[i];
				index = i;
			}
		}

		return index;
	}

	std::pair<uint32_t, std::vector<double>> NeuralNetowrk::Classify(Data data) {
		auto outputs = CalculateOutputs(data.GetData());
		uint32_t predictedClass = MaxValueIndex(outputs);
		return std::make_pair(predictedClass, outputs);
	}

	std::vector<double> NeuralNetowrk::CalculateOutputs(std::vector<double> input) {
		std::vector<double> result = input;

		for (auto& layer : m_Layers)
			result = layer->CalculateOutputs(result);

		return result;
	}

}