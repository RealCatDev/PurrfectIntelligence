#include "PurrfectIntelligence.h"

#include <assert.h>

#include <string>
#include <sstream>
#include <fstream>

namespace PurrfectIntelligence {

	Layer::Layer(LayerArchitecture arch) 
		: m_NeuronsIn(arch.inNeurons), m_NeuronsOut(arch.outNeurons) {
		m_Weights = new Math::Matrix(m_NeuronsIn, m_NeuronsOut);
		m_Biases = new Math::Matrix(1, m_NeuronsOut);
		m_Activation = new Math::Matrix(1, m_NeuronsOut);

		m_Weights->Random(-1, 1);
		m_Biases->Random(-1, 1);
	}

	Layer::~Layer() {
		delete m_Weights;
		delete m_Biases;
	}

	Math::Matrix* Layer::Forward(Math::Matrix* input) {
		m_Activation->Dot(input, m_Weights);
		m_Activation->Sum(m_Biases);
		m_Activation->ApplySigmoid();

		return m_Activation;
	}

	NeuralNetwork* NeuralNetwork::Load(const char* filepath) {
		std::ifstream file(filepath);
		assert(file.is_open() && "Failed to load neural network.");

		NNArchitecture arch{};

		std::string buf;
		std::getline(file, buf);
		arch.arch.resize(std::stoi(buf));
		for (size_t i = 0; i < arch.arch.size(); ++i) {
			std::getline(file, buf);
			arch.arch[i] = std::stoi(buf);
		}

		NeuralNetwork* nn = new NeuralNetwork(arch);

		for (size_t i = 0; i < arch.arch.size()-1; ++i) {
			for (size_t j = 0; j < 2; j++) {
				std::getline(file, buf);
				char type = buf.at(0);
				Math::Matrix* mat;

				int rows = 1;
				int cols = 1;

				switch (type) {

				case 'w': {
					std::getline(file, buf);
					rows = std::stoi(buf);
					std::getline(file, buf);
					cols = std::stoi(buf);
					mat = new Math::Matrix(rows, cols);
				} break;

				case 'b': {
					std::getline(file, buf);
					cols = std::stoi(buf);
					mat = new Math::Matrix(1, cols);
				} break;

				default:
					printf("File not valid!");
					return nullptr;
				}

				for (size_t y = 0; y < cols; ++y) {
					std::getline(file, buf);
					std::istringstream iss(buf);
					for (size_t x = 0; x < rows; ++x) {
						float value;
						if (!(iss >> value)) {
							throw std::runtime_error("Invalid input string");
						}
						mat->Get(x, y) = value;
					}
				}

				if (type == 'w')
					nn->m_Layers[i]->m_Weights = mat;
				else if (type == 'b')
					nn->m_Layers[i]->m_Biases = mat;
			}
		}

		return nn;
	}

	NeuralNetwork::NeuralNetwork(NNArchitecture arch, bool isGradient) 
		: m_Arch(arch) {
		m_Layers.resize(arch.arch.size() - 1);
		for (size_t i = 0; i < m_Layers.size(); ++i)
			m_Layers[i] = std::make_unique<Layer>(LayerArchitecture(arch.arch[i], arch.arch[i+1]));
		m_Activation = new Math::Matrix(1, arch.arch[0]);
		if (!isGradient) m_Gradient = new NeuralNetwork(arch, true);
	}

	NeuralNetwork::~NeuralNetwork() {
		delete m_Activation;
	}

	void NeuralNetwork::Save(const char* filepath) {
		std::ofstream file(filepath);
		
		file << m_Arch.arch.size() << std::endl;
		for (size_t i = 0; i < m_Arch.arch.size(); ++i)
			file << m_Arch.arch[i] << std::endl;

		for (size_t i = 0; i < m_Arch.arch.size()-1; ++i) {
			file << "w" << std::endl << m_Layers[i]->m_Weights->GetRows() << std::endl << m_Layers[i]->m_Weights->GetCols() << std::endl;
			file << m_Layers[i]->m_Weights->ToString();
			file << "b" << std::endl << m_Layers[i]->m_Weights->GetCols() << std::endl;
			file << m_Layers[i]->m_Biases->ToString();
		}

		file.close();
	}

	void NeuralNetwork::Print() {
		printf("NeuralNetwork: {\n");
		for (size_t i = 0; i < m_Layers.size(); ++i) {
			auto& layer = m_Layers[i];
			printf("    Layer no. %zu: [\n", i + 1);
			layer->m_Weights->Print("Weights", 4 * 2);
			layer->m_Biases->Print("Biases", 4*2);
			printf("    ]\n");
		}
		printf("]\n");
	}

	void NeuralNetwork::TrainBathes(Batch* batch, size_t batchSize, Math::Matrix* td, float rate) {
		//if (batch->finished) {
		//	batch->finished = false;
		//	batch->begin = 0;
		//	batch->cost = 0;
		//}

		//size_t size = batchSize;
		//if (batch->begin + batchSize >= td->GetRows()) {
		//	size = td->GetRows() - batch->begin;
		//}

		//// TODO: introduce similar to row_slice operation but for Mat that will give you subsequence of rows
		//Math::Matrix *batch_t = new Math::Matrix(size, td->GetCols());
		//batch_t->SetData(&td->Get(batch->begin, 0));

		//Backprop(rate, batch_t);
		////m_Gradient->Learn(rate);
		//batch->cost += Cost(batch_t);
		//batch->begin += batchSize;

		//if (batch->begin >= td->GetRows()) {
		//	size_t batch_count = (td->GetRows() + batchSize - 1) / batchSize;
		//	batch->cost /= batch_count;
		//	batch->finished = true;
		//}
	}

	void NeuralNetwork::Forward() {
		Math::Matrix* input = m_Activation;
		for (auto& layer : m_Layers)
			input = layer->Forward(input);
	}

	float NeuralNetwork::Cost(Math::Matrix* td) {
		assert(Input().GetCols() + Output().GetCols() == td->GetCols());
		size_t n = td->GetRows();

		float c = 0;
		for (size_t i = 0; i < n; ++i) {
			Math::Matrix* row = td->GetRow(i);
			Math::Matrix* x = new Math::Matrix(1, Input().GetCols());
			x->SetData(&row->Get(0, 0));
			Math::Matrix* y = new Math::Matrix(1, Output().GetCols());
			y->SetData(&row->Get(0, Input().GetCols()));

			Input().Copy(x);
			Forward();

			size_t q = y->GetCols();
			for (size_t j = 0; j < q; ++j) {
				float d = Output().Get(0, j) - y->Get(0, j);
				c += d * d;
			}
		}

		return c / n;
	}

	void NeuralNetwork::FiniteDiff(float eps, Math::Matrix* td) {
		float saved;
		float c = Cost(td);
		for (size_t i = 0; i < m_Layers.size(); ++i) {
			for (size_t j = 0; j < m_Layers[i]->m_Weights->GetRows(); ++j) {
				for (size_t k = 0; k < m_Layers[i]->m_Weights->GetCols(); ++k) {
					saved = m_Layers[i]->m_Weights->Get(j, k);
					m_Layers[i]->m_Weights->Get(j, k) += eps;
					m_Gradient->m_Layers[i]->m_Weights->Get(j, k) = (Cost(td) - c) / eps;
					m_Layers[i]->m_Weights->Get(j, k) = saved;
				}
			}

			for (size_t k = 0; k < m_Layers[i]->m_Biases->GetCols(); ++k) {
				saved = m_Layers[i]->m_Biases->Get(0, k);
				m_Layers[i]->m_Biases->Get(0, k) += eps;
				m_Gradient->m_Layers[i]->m_Biases->Get(0, k) = (Cost(td) - c) / eps;
				m_Layers[i]->m_Biases->Get(0, k) = saved;
			}
		}
	}

//	NeuralNetwork* NeuralNetwork::Backprop(float rate, Math::Matrix *td) {
//		size_t n = td->GetRows();
//		assert(m_Layers[0]->m_Activation->GetCols() + m_Layers[m_Layers.size()-1]->m_Activation->GetCols() == td->GetCols());
//
//		NeuralNetwork *g = new NeuralNetwork(m_Arch);
//
//		// i - current sample
//		// l - current layer
//		// j - current activation
//		// k - previous activation
//
//		for (size_t i = 0; i < n; ++i) {
//			Math::Matrix *row = td->GetRow(i);
//			Math::Matrix* in = new Math::Matrix(1, m_Activation->GetCols());
//			in->SetData(&row->Get(0, 0));
//			Math::Matrix *out = new Math::Matrix(1, m_Layers[m_Layers.size()-1]->m_Activation->GetCols());
//			in->SetData(&row->Get(0, m_Activation->GetCols()));
//
//			Forward();
//
//			for (size_t j = 0; j < m_Layers.size()+1; ++j) {
//				auto mat = j == 0 ? m_Activation : m_Layers[j-1]->m_Activation;
//				mat->Fill(0);
//			}
//
//			for (size_t j = 0; j < out->GetCols(); ++j) {
//#if 1
//				g->m_Layers[m_Layers.size() - 1]->m_Activation->Get(0, j) = 2 * (m_Layers[m_Layers.size() - 1]->m_Activation->Get(0, j) - out->Get(0, j));
//#else
//				g->m_Layers[m_Layers.size() - 1]->m_Activation->Get(0, j) = m_Layers[m_Layers.size() - 1]->m_Activation->Get(0, j) - out->Get(0, j);
//#endif // BACKPROP_TRADITIONAL
//			}
//
//#if 1
//			float s = 1;
//#else
//			float s = 2;
//#endif // BACKPROP_TRADITIONAL
//
//			for (size_t l = m_Layers.size()-1; l > 0; --l) {
//				for (size_t j = 0; j < m_Layers[l]->m_Activation->GetCols(); ++j) {
//					float a = m_Layers[l]->m_Activation->Get(0, j);
//					float da = g->m_Layers[l]->m_Activation->Get(0, j);
//					float qa = a*(1-a);//dactf(a, NN_ACT);
//					g->m_Layers[l - 1]->m_Biases->Get(0, j) += s * da * qa;
//					for (size_t k = 0; k < m_Layers[l-1]->m_Activation->GetCols(); ++k) {
//						// j - weight matrix col
//						// k - weight matrix row
//						float pa = m_Layers[l]->m_Activation->Get(0, k);
//						float w = m_Layers[l]->m_Weights->Get(k, j);
//						m_Layers[l]->m_Activation->Get(k, j) += s * da * qa * pa;
//						m_Layers[l]->m_Weights->Get(0, k) += s * da * qa * w;
//					}
//				}
//			}
//		}
//
//		for (size_t i = 0; i < g->m_Layers.size(); ++i) {
//			for (size_t j = 0; j < g->m_Layers[i]->m_Weights->GetRows(); ++j) {
//				for (size_t k = 0; k < g->m_Layers[i]->m_Weights->GetCols(); ++k) {
//					g->m_Layers[i]->m_Weights->Get(j, k) /= n;
//				}
//			}
//			for (size_t k = 0; k < g->m_Layers[i]->m_Biases->GetCols(); ++k) {
//				g->m_Layers[i]->m_Biases->Get(0, k) /= n;
//			}
//		}
//
//		return g;
//	}

	void NeuralNetwork::Learn(float rate) {
		for (size_t i = 0; i < m_Layers.size(); ++i) {
			for (size_t j = 0; j < m_Layers[i]->m_Weights->GetRows(); ++j) {
				for (size_t k = 0; k < m_Layers[i]->m_Weights->GetCols(); ++k) {
					m_Layers[i]->m_Weights->Get(j, k) -= rate * m_Gradient->m_Layers[i]->m_Weights->Get(j, k);
				}
			}

			for (size_t k = 0; k < m_Layers[i]->m_Biases->GetCols(); ++k) {
				m_Layers[i]->m_Biases->Get(0, k) -= rate * m_Gradient->m_Layers[i]->m_Biases->Get(0, k);
			}
		}
	}

}