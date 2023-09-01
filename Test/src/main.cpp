#include <PurrfectIntelligence.h>

#include <time.h>

using namespace PurrfectIntelligence;

#include <assert.h>

//typedef struct {
//	Matrix* a0 = new Matrix(1, 2);
//
//	Matrix* w1 = new Matrix(2, 2);
//	Matrix* b1 = new Matrix(1, 2);
//	Matrix* a1 = new Matrix(1, 2);
//
//	Matrix* w2 = new Matrix(2, 1);
//	Matrix* b2 = new Matrix(1, 1);
//	Matrix* a2 = new Matrix(1, 1);
//} Xor;
//
//void forward_xor(Xor model) {
//	model.a1->Dot(model.a0, model.w1);
//	model.a1->Sum(model.b1);
//	model.a1->ApplySigmoid();
//
//	model.a2->Dot(model.a1, model.w2);
//	model.a2->Sum(model.b2);
//	model.a2->ApplySigmoid();
//}
//
//float cost_xor(Xor model, Matrix *ti, Matrix *to) {
//	assert(ti->GetRows() == to->GetRows());
//	assert(to->GetCols() == model.a2->GetCols());
//	size_t n = ti->GetRows();
//
//	float c = 0;
//	for (size_t i = 0; i < n; ++i) {
//		model.a0->Copy(ti->GetRow(i));
//		Matrix* y = to->GetRow(i);
//		forward_xor(model);
//
//		for (size_t j = 0; j < to->GetCols(); ++j) {
//			float dst = model.a2->Get(0, j) - y->Get(0, j);
//			c += dst*dst;
//		}
//	}
//	return c/n;
//}
//
//void finite_diff(Xor model, Xor g, float eps, Matrix *ti, Matrix *to) {
//	float saved;
//
//	float c = cost_xor(model, ti, to);
//
//	for (size_t i = 0; i < model.w1->GetRows(); ++i)
//		for (size_t j = 0; j < model.w1->GetCols(); ++j) {
//			saved = model.w1->Get(i, j);
//			model.w1->Get(i, j) += eps;
//			g.w1->Get(i, j) = (cost_xor(model, ti, to) - c) / eps;
//			model.w1->Get(i, j) = saved;
//		}
//
//	for (size_t i = 0; i < model.b1->GetRows(); ++i)
//		for (size_t j = 0; j < model.b1->GetCols(); ++j) {
//			saved = model.b1->Get(i, j);
//			model.b1->Get(i, j) += eps;
//			g.b1->Get(i, j) = (cost_xor(model, ti, to) - c) / eps;
//			model.b1->Get(i, j) = saved;
//		}
//
//	for (size_t i = 0; i < model.w2->GetRows(); ++i)
//		for (size_t j = 0; j < model.w2->GetCols(); ++j) {
//			saved = model.w2->Get(i, j);
//			model.w2->Get(i, j) += eps;
//			g.w2->Get(i, j) = (cost_xor(model, ti, to) - c) / eps;
//			model.w2->Get(i, j) = saved;
//		}
//
//	for (size_t i = 0; i < model.b2->GetRows(); ++i)
//		for (size_t j = 0; j < model.b2->GetCols(); ++j) {
//			saved = model.b2->Get(i, j);
//			model.b2->Get(i, j) += eps;
//			g.b2->Get(i, j) = (cost_xor(model, ti, to) - c) / eps;
//			model.b2->Get(i, j) = saved;
//		}
//}
//
//void learn(Xor m, Xor g, float rate) {
//	for (size_t i = 0; i < m.w1->GetRows(); ++i)
//		for (size_t j = 0; j < m.w1->GetCols(); ++j) {
//			m.w1->Get(i, j) -= rate * g.w1->Get(i, j);
//		}
//
//	for (size_t i = 0; i < m.b1->GetRows(); ++i)
//		for (size_t j = 0; j < m.b1->GetCols(); ++j) {
//			m.b1->Get(i, j) -= rate * g.b1->Get(i,j);
//		}
//
//	for (size_t i = 0; i < m.w2->GetRows(); ++i)
//		for (size_t j = 0; j < m.w2->GetCols(); ++j) {
//			m.w2->Get(i, j) -= rate * g.w2->Get(i,j);
//		}
//
//	for (size_t i = 0; i < m.b2->GetRows(); ++i)
//		for (size_t j = 0; j < m.b2->GetCols(); ++j) {
//			m.b2->Get(i, j) -= rate * g.b2->Get(i,j);
//		}
//}

int main(void) {
	srand(time(0));

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
	NeuralNetwork* nn = new NeuralNetwork(arch);
	
	float eps = 1e-1;
	float rate = 1e-1;

	printf("Cost = %f\n", nn->Cost(td));
	for (size_t i = 0; i < 20*1000; ++i) {
		nn->FiniteDiff(eps, td);
		nn->Learn(rate);
		printf("%zu: Cost = %f\n", i, nn->Cost(td));
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
