#pragma once

#include <iostream>
#include <vector>

#include <filesystem>
#include <functional>

namespace PurrfectIntelligence {

	using ActivationFunc = double(*)(double);
	using ActivationDerivativeFunc = double(*)(double);
	using CostFunc = double (*)(::std::vector<double>, ::std::vector<double>);
	using CostDerivativeFunc = double (*)(double, double);

	namespace Activation {

		double Sigmoid(double x) {
			return x / (1 + abs(x));
		}

		double SigmoidD(double x) {
			double a = Sigmoid(x);
			return a * (1 - a);
		}

		double ReLU(double x) {
			return ::std::max(0.0, x);
		}

		double ReLU(double x) {
			return (x > 0) ? 1 : 0;
		}

	}

	namespace Cost {

		double MeanSquareError(::std::vector<double> predictedOutputs, ::std::vector<double> expectedOutputs) {
			double cost = 0;
			for (int i = 0; i < predictedOutputs.size(); i++)
			{
				double error = predictedOutputs[i] - expectedOutputs[i];
				cost += error * error;
			}
			return 0.5 * cost;
		}

		double MeanSquareErrorD(double predictedOutput, double expectedOutput) {
			return predictedOutput - expectedOutput;
		}

	}

	struct LayerLearnData {

		std::vector<double> inputs;
		std::vector<double> weightedInputs;
		std::vector<double> activations;
		std::vector<double> nodeValues;

		LayerLearnData(Layer layer)
			: weightedInputs(layer.GetArchitecture().neuronsOut), activations(layer.GetArchitecture().neuronsOut), nodeValues(layer.GetArchitecture().neuronsOut) {
		}

	};

	struct LayerArchitecture {
		uint32_t neuronsIn;
		uint32_t neuronsOut;
	};

	class Layer {

	public:

		Layer(LayerArchitecture arch, ActivationFunc activation = Activation::Sigmoid, ActivationDerivativeFunc activationD = Activation::SigmoidD, CostFunc cost = Cost::MeanSquareError, CostDerivativeFunc costD = Cost::MeanSquareErrorD);
		~Layer();

		inline void SetActivation(ActivationFunc func, ActivationDerivativeFunc derivative) { m_Activation = func; m_ActivationDerivative = derivative; }
		
		std::vector<double> Forward(std::vector<double> inputs);
		std::vector<double> Forward(std::vector<double> inputs, LayerLearnData &learnData);

		void ApplyGradients(double learnRate, double regularization, double momentum);
		void CalculateOutputLayerNodeValues(LayerLearnData &layerLearnData, std::vector<double> expectedOutputs, CostFunc cost);
		void CalculateHiddenLayerNodeValues(LayerLearnData &layerLearnData, std::unique_ptr<Layer> &oldLayer, std::vector<double> oldNodeValues);
		void UpdateGradients(LayerLearnData &layerLearnData);

		inline LayerArchitecture GetArchitecture() { return m_Arch; }

	private:

		LayerArchitecture m_Arch;
		ActivationFunc m_Activation;
		ActivationDerivativeFunc m_ActivationDerivative;
		CostFunc m_Cost;
		CostDerivativeFunc m_CostDerivative;

		std::vector<double> m_Weights;
		std::vector<double> m_Biases;
		std::vector<double> m_CostGradientW;
		std::vector<double> m_CostGradientB;
		std::vector<double> m_WeightVelocities;
		std::vector<double> m_BiasVelocities;

	};

	class Data {

	public:

		Data(std::vector<double> value, uint32_t label, uint32_t labelCount);
		~Data();

		inline std::vector<double> GetData() { return m_Data; }
		inline std::vector<double> GetExpectedOutputs() { return m_ExpectedOutputs; }

	private:

		static std::vector<double> CreateOneHot(uint32_t index, uint32_t size);

		uint32_t m_Label;

		std::vector<double> m_Data;
		std::vector<double> m_ExpectedOutputs;

	};

	struct NetworkLearnData {

		std::vector<LayerLearnData> layerData;

		NetworkLearnData(std::vector<std::unique_ptr<Layer>> layers)
			: layerData(layers.size()) {
			for (uint32_t i = 0; i < layers.size(); ++i)
				layerData[i] = LayerLearnData(*layers[i].get());
		}

	};

	struct NeuralNetworkArchitecture {
		uint32_t inputLayerNeurons;
		std::vector<uint32_t> hiddenLayersNeurons;
		uint32_t outputLayerNeurons;
	};

	class NeuralNetowrk {

	public:

		NeuralNetowrk(NeuralNetworkArchitecture arch, ActivationFunc activation = Activation::Sigmoid, ActivationDerivativeFunc activationD = Activation::SigmoidD, CostFunc cost = Cost::MeanSquareError, CostDerivativeFunc costD = Cost::MeanSquareErrorD);
		~NeuralNetowrk();

		void Train(std::vector<Data> data, double learnRate, double regularization = 0, double momentum = 0);

	private:

		NeuralNetworkArchitecture m_Arch;
		ActivationFunc m_Activation;
		ActivationDerivativeFunc m_ActivationDerivative;
		CostFunc m_Cost;
		CostDerivativeFunc m_CostDerivative;

		// NOTE(CatDev): Input layer is not a real layer
		std::vector<std::unique_ptr<Layer>> m_Layers;

		std::vector<NetworkLearnData> m_BatchLearnData;

	};

}