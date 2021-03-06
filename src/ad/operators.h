#pragma once

#include <memory>

#include "graph.h"

namespace ad {

Var operator+(const Var& v1, const Var& v2);
Var operator-(const Var& v1, const Var& v2);
Var operator-(float a, const Var& v2);
Var operator-(const Var& v1, float a);
Var operator*(const Var& v1, const Var& v2);
Var operator*(float a, const Var& v2);
Var operator*(const Var& v1, float a);
Var Relu(const Var& v1);
Var EltSquare(const Var& v1);
Var operator^(const Var& v1, const Var& v2);
Var Log(const Var& x);
Var NLog(const Var& x);
Var CrossEntropy(const Var& h, const Var& y);
Var Softmax(const Var& x);
Var SoftmaxLoss(Var h, Var y);
Var Sigmoid(const Var& x);
Var Sum(const Var& a);
Var Mean(const Var& a);
Var MSE(const Var& h, const Var& y);
Var SSE(const Var& h, const Var& y);
Var NthCol(const Var& w, int n);
Var Tanh(const Var& x);
Var ColAppend(Var x, Var y);
Var ColSplit(Var x, int from, int len);

}
