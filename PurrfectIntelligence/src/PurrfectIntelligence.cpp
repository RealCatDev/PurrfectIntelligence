#include "PurrfectIntelligence.h"

#define MAXCHAR 10000

#define _USE_MATH_DEFINES
#include <math.h>

#include <execution>

namespace PurrfectIntelligence {

	double NormalDistribution(double mean, double standardDeviation) {
		double x1 = 1 - std::rand();
		double x2 = 1 - std::rand();

		double y1 = sqrt(-2.0 * log(x1)) * cos(2.0 * M_PI * x2);
		return y1 * standardDeviation + mean;
	}

	Layer::Layer(LayerArchitecture arch, ActivationFunc activation, ActivationDerivativeFunc activationD, CostFunc cost, CostDerivativeFunc costD)
		: m_Arch(arch), m_Activation(activation), m_ActivationDerivative(activationD), m_Cost(cost), m_CostDerivative(costD), m_Weights(arch.neuronsIn* arch.neuronsOut), m_Biases(arch.neuronsOut), m_CostGradientW(m_Weights.size()), m_CostGradientB(m_Biases.size()), m_WeightVelocities(m_Weights.size()), m_BiasVelocities(m_Biases.size()) {
		for (uint32_t i = 0; i < m_Weights.size(); ++i)
			m_Weights[i] = NormalDistribution(0, 1) / sqrt(arch.neuronsOut);
	}

	Layer::~Layer() {

	}

	std::vector<double> Layer::Forward(std::vector<double> inputs) {
		std::vector<double> activations;

		for (uint32_t i = 0; i < m_Weights.size(); ++i) {
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

		for (uint32_t i = 0; i < m_Weights.size(); ++i) {
			double weightedSum = m_Biases[i];
			for (int j = 0; j < m_Weights[i]; ++j) {
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
				double weightedInputDerivative = oldLayer->m_Weights[newNodeIndex * oldLayer->m_Arch.neuronsIn + oldNodeIndex];
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

	NeuralNetowrk::NeuralNetowrk(NeuralNetworkArchitecture arch, ActivationFunc activation, ActivationDerivativeFunc activationD, CostFunc cost, CostDerivativeFunc costD)
		: m_Arch(arch), m_Activation(activation), m_ActivationDerivative(activationD), m_Cost(cost), m_CostDerivative(costD) {
		m_Layers.resize(arch.hiddenLayersNeurons.size() + 1);
		for (uint32_t i = 0; i < m_Layers.size(); ++i) {
			uint32_t inputs = 0;
			if (i == 0) inputs = arch.inputLayerNeurons;
			else if (i == m_Layers.size() - 1) inputs = arch.outputLayerNeurons;
			else inputs = arch.hiddenLayersNeurons[i-1];
			LayerArchitecture layerArch{};
			layerArch.neuronsIn = inputs;
			layerArch.neuronsOut = i < m_Layers.size() ? arch.hiddenLayersNeurons[i] : arch.outputLayerNeurons;

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

		std::for_each(std::execution::par, data.begin(), data.end(), [this](Data data, uint32_t i) {
			printf("Train no. %zu.", i);
			std::vector<double> input = data.GetData();
			for (auto& layer : m_Layers) {
				input = layer->Forward(input);
			}

			auto &outputLayer = m_Layers[m_Layers.size() - 1];
			LayerLearnData outputLearnData = m_BatchLearnData[i].layerData[m_Layers.size() - 1];

			outputLayer->CalculateOutputLayerNodeValues(outputLearnData, data.GetExpectedOutputs(), m_Cost);
			outputLayer->UpdateGradients(outputLearnData);

			for (int j = m_Layers.size() - 2; j >= 0; j--)
			{
				LayerLearnData layerLearnData = m_BatchLearnData[i].layerData[j];
				auto &hiddenLayer = m_Layers[j];

				auto &temp = m_Layers[j + 1];
				hiddenLayer->CalculateHiddenLayerNodeValues(layerLearnData, temp, m_BatchLearnData[i].layerData[j + 1].nodeValues);
				hiddenLayer->UpdateGradients(layerLearnData);
			}
		});

		for (auto& layer : m_Layers)
			layer->ApplyGradients(learnRate / data.size(), regularization, momentum);
	}

}