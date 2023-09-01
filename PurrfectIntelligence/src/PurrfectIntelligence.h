#pragma once

#include <vector>

#include "PurrMath.h"

namespace PurrfectIntelligence {

	struct LayerArchitecture {
		uint32_t inNeurons;
		uint32_t outNeurons;
	};

	class Layer {
		friend class NeuralNetwork;
	public:

		Layer(LayerArchitecture arch);
		~Layer();

		Math::Matrix *Forward(Math::Matrix *input);

	private:

		uint32_t m_NeuronsIn;
		uint32_t m_NeuronsOut;

		Math::Matrix* m_Weights;
		Math::Matrix* m_Biases;
		Math::Matrix* m_Activation;

	};

	struct Batch {
		size_t begin;
		float cost;
		bool finished;
	};

	struct NNArchitecture {
		std::vector<uint32_t> arch;
	};

	class NeuralNetwork {

	public:

		NeuralNetwork(NNArchitecture arch, bool = false);
		~NeuralNetwork();

		void Print();

		void TrainBathes(Batch *bath, size_t batchSize, Math::Matrix *td, float rate);

		void Forward();

		float Cost(Math::Matrix*);
		void FiniteDiff(float eps, Math::Matrix *td);
		//void Backprop(float, Math::Matrix*);
		void Learn(float);

		inline Math::Matrix &Input() { return *m_Activation; }
		inline Math::Matrix &Output() { return *m_Layers[m_Layers.size()-1]->m_Activation; }

	private:

	private:

		NNArchitecture m_Arch;

		std::vector<std::unique_ptr<Layer>> m_Layers;
		Math::Matrix* m_Activation;

		NeuralNetwork* m_Gradient = nullptr;

	};

}