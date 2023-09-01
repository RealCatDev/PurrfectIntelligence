#include "PurrMath.h"

#include <random>
#include <assert.h>

namespace PurrfectIntelligence {

	namespace Utils {

		double NormalDistribution(double mean, double standardDeviation) {
			std::random_device rd;
			std::mt19937 gen(rd());
			std::normal_distribution<double> distribution(mean, standardDeviation);
			return distribution(gen);
		}

		float RandomFloat() {
			return (float)rand() / (float)RAND_MAX;
		}

	}

	namespace Math {

		float sigmoidf(float x) {
			return 1.0f / (1.0f + expf(-x));
		}

		Matrix::Matrix(size_t rows, size_t cols)
			: m_Rows(rows), m_Cols(cols) {
			m_Es = (MatrixType*) malloc(sizeof(*m_Es) * rows * cols);
			assert(m_Es != NULL);
			Fill(0.0f);
		}

		Matrix::~Matrix() {
			delete[] m_Es;
		}

		void Matrix::Copy(Matrix* src) {
			assert(m_Rows == src->m_Rows);
			assert(m_Cols == src->m_Cols);
			
			for (size_t i = 0; i < m_Rows; ++i)
				for (size_t j = 0; j < m_Cols; ++j)
					Get(i, j) = src->Get(i, j);
		}

		Matrix* Matrix::GetRow(size_t row) {
			Matrix* m = new Matrix(1, m_Cols);
			m->m_Es = &Get(row, 0);
			return m;
		}

		void Matrix::Print(const char* name, size_t padding) {
			printf("%*s%s: [\n", (int)padding, "", name);
			for (size_t y = 0; y < m_Cols; ++y)
				for (size_t x = 0; x < m_Rows; ++x)
					printf("%*s%s%f%s", x==0?padding:0, "", x == 0 ? "\t" : "", Get(x, y), x < m_Rows - 1 ? ", " : "\n");
			printf("%*s]\n", (int)padding, "");
		}

		void Matrix::Fill(MatrixType value) {
			for (size_t x = 0; x < m_Rows; ++x)
				for (size_t y = 0; y < m_Cols; ++y)
					Get(x, y) = value;
		}

		void Matrix::Random(float low, float high) {
			for (size_t i = 0; i < m_Rows; ++i)
				for (size_t j = 0; j < m_Cols; ++j)
					Get(i, j) = Utils::RandomFloat() * (high - low) + low;
		}

		void Matrix::Sum(Matrix* a) {
			assert(m_Rows == a->m_Rows);
			assert(m_Cols == a->m_Cols);
			for (size_t i = 0; i < m_Rows; ++i)
				for (size_t j = 0; j < m_Cols; ++j)
					Get(i, j) += a->Get(i, j);
		}

		void Matrix::Dot(Matrix* a, Matrix* b) {
			assert(a->m_Cols == b->m_Rows);
			size_t n = a->m_Cols;
			assert(m_Rows == a->m_Rows);
			assert(m_Cols == b->m_Cols);

			for (size_t i = 0; i < m_Rows; ++i)
				for (size_t j = 0; j < m_Cols; ++j) {
					Get(i, j) = 0;
					for (size_t k = 0; k < n; ++k)
						Get(i, j) += a->Get(i, k) * b->Get(k, j);
				}
		}

		void Matrix::ApplySigmoid() {
			for (size_t i = 0; i < m_Rows; ++i)
				for (size_t j = 0; j < m_Cols; ++j)
					Get(i, j) = sigmoidf(Get(i, j));
		}

		void Matrix::ShuffleRows() {
			for (size_t i = 0; i < m_Rows; ++i) {
				size_t j = i + rand() % (m_Rows - i);
				if (i != j) {
					for (size_t k = 0; k < m_Cols; ++k) {
						float t = Get(i, k);
						Get(i, k) = Get(j, k);
						Get(j, k) = t;
					}
				}
			}
		}

	}

}
