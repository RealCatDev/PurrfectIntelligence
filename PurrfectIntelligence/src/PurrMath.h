#pragma once

#include <iostream>

namespace PurrfectIntelligence {

	namespace Math {

		float sigmoidf(float x);

#ifndef MATRIX_TYPE
		typedef float MatrixType;
#endif

		class Matrix {

		public:

			Matrix(size_t rows, size_t cols);
			~Matrix();

			void Copy(Matrix* src);
			Matrix* GetRow(size_t row);

			void Print(const char *name = "Matrix", size_t padding = 0);

			void Fill(MatrixType value);
			void Random(float low, float high);

			void Sum(Matrix* other);
			void Dot(Matrix* a, Matrix* b);

			void ApplySigmoid();

			void ShuffleRows();

			inline const MatrixType Get(size_t x, size_t y) const { return m_Es[x * m_Cols + y]; }
			inline MatrixType &Get(size_t x, size_t y) { return m_Es[x*m_Cols+y]; }

			inline size_t GetRows() const { return m_Rows; }
			inline size_t GetCols() const { return m_Cols; }

			inline void SetData(MatrixType* data) { m_Es = data; }

		private:

			size_t m_Rows;
			size_t m_Cols;
			MatrixType* m_Es;

		};

	}

}
