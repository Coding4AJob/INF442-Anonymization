//
// Created by jijingtian on 4/11/21.
//

#ifndef INF442_PROJECT_SVM_H
#define INF442_PROJECT_SVM_H
#include <cmath>
#include <math.h>
#include <Eigen/Dense>
#include "kernel.h"
#include <set>

#define EPSILON = 1e-5;
using namespace Eigen;

// binary classification
class SVM {
public:
    SVM(double c, Kernel* kernel);
    ~SVM();

    void fit(const MatrixXd& x,const VectorXd& y);
    VectorXd predict(const MatrixXd& x);
    double score(const MatrixXd& x,const VectorXd& y);

private:
    // f(x) = sign(sum(_ai * yi * K(x,xi) )+ _b)
    VectorXd _a;
    double _b = 0.0;
    // _n: number of trained data, _d dim
    int _n;
    int _d;
    double _c;

    MatrixXd _train_x;
    VectorXd _train_y;
    MatrixXd _kernal_val;
    Kernel* _kernal;

    // member function
    void _init(const MatrixXd& x,const VectorXd& y);
    // SMO sovler
    VectorXd _E;
    std::set<int> _zero_a,_sup_vec;
    int _find_first();
    int _find_second(int i);
    double _clip_a(int i,int j,double a_j_new);
    void _update_b_E(int i,int j,double ainew,double ajnew);

    void _smo();






};


#endif //INF442_PROJECT_SVM_H
