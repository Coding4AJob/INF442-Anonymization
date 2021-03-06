//
// Created by jijingtian on 4/11/21.
//

#include "svm.h"

#include <utility>

SVM::SVM(double c, Kernel* kernel){
    this->_kernal = kernel;
    this->_c = c;
}

SVM::~SVM(){
}

void SVM::_init(const MatrixXd& x, const VectorXd& y) {
    this->_n = x.rows();
    this->_d = x.cols();
    this->_train_x = x;
    this->_train_y = y;
    this->_a = VectorXd::Zero(x.rows());
    this->_b = 0.0;
    // kernel values precalculated, symmetric positive
    this->_kernal_val = MatrixXd(this->_n,this->_n);
    for(int i = 0;i<this->_n;i++){
        for(int j = i;j<this->_n;j++){
            this->_kernal_val(i,j) = this->_kernal->val(x.row(i),x.row(j));
            this->_kernal_val(j,i) = this->_kernal_val(i,j);
        }
    }
    // smo
    // E
    this->_E = VectorXd(this->_n);
    for(int i = 0;i<this->_n;i++){

        this->_E(i) = -1.0*this->_train_y(i);
    }
    // at initial time, a equals all zero, no support vect
    for(int i = 0;i<this->_n;i++){
        this->_zero_a.insert(i);
    }


}

int SVM::_find_first() {
    // find the index of a which violate the kkt condition most
    // satisfy the KKT condition one of them:
    // ai =0 && yi * g(xi) >= 1, normal
    // 0<ai < C && yi*g(xi) = 1 , support vec
    // ai = C && yi*g(xi) <=1, between the two boundary
    // g(xi) = _E(i) + yi

    // violate:
    double gi;
    double ai;
    double yi;
    /*
    for(int i = 0 ;i<this->_n;i++){
        gi = this->_E(i) + this->_train_y(i);
        ai = this->_a(i);
        yi = this->_train_y(i);
        if(((yi*gi <= 1.0)&&ai<this->_c)|| (yi*gi >=1 && ai > 0) || (abs(yi*gi - 1)<1e-5 && ((abs(ai - this->_c)<1e-5)||(abs(ai)<1e-5)))){
            return i;
        }
    }
     */
    for(auto it = _sup_vec.begin();it!=_sup_vec.end();it++){
        // support a, a!=0
        int i = *it;
        gi = this->_E(i) + this->_train_y(i);
        ai = this->_a(i);
        yi = this->_train_y(i);
        if((ai >= _c - _epsilon && yi*gi > 1) || (ai < _c -_epsilon && abs(yi*gi - 1)>_epsilon)){
            return i;
        }

    }

    for(auto it = _zero_a.begin();it!=_zero_a.end();it++){
        int i = *it;
        gi = this->_E(i) + this->_train_y(i);
        ai = this->_a(i);
        yi = this->_train_y(i);
        if(yi*gi <= 1){
            return i;
        }
    }

    return -1;
}

int SVM::_find_second(int i) {
    double Ei = this->_E(i);
    double max = 0.0;
    int res = -1;
    for(int j =0;j<this->_n;j++){
        if(abs(this->_E(j) - Ei) > max){
            max = abs(this->_E(j) - Ei);
            res = j;
        }
    }
    return res;

}

double SVM::_clip_a(int i, int j, double ajnew) {
    double L = 0.0,H = 0.0;
    if(abs(_train_y(i)-_train_y(j))>_epsilon){
        // yi != yj
        L = std::max(0.0,_a(j)-_a(i));
        H = std::min(_c,_c+_a(j)-_a(i));
    }else{
        L = std::max(0.0,_a(j)+_a(i) - _c);
        H = std::min(_c,_a(i) + _a(j));
    }
    ajnew = std::min(H,ajnew);
    ajnew = std::max(L,ajnew);
    return ajnew;
}

void SVM::_update_b_E(int i, int j, double ainew, double ajnew) {
    _update_a(i,ainew);
    _update_a(j,ajnew);

    double bi = -1 * _E(i) - _train_y(i)*_kernal_val(i,i)*(ainew - _a(i)) - _train_y(j)*_kernal_val(j,i)*(ajnew - _a(j)) + _b;
    double bj = -1 * _E(j) - _train_y(i)*_kernal_val(i,j)*(ainew - _a(i)) - _train_y(j)*_kernal_val(j,j)*(ajnew - _a(j)) + _b;
    double newb = bi + (bj - bi)/2;
    //_E(i) = _E(i) + (ainew - _a(i))*_train_y(i)*_kernal_val(i,i) + (ajnew - _a(j))*_train_y(j)*_kernal_val(i,j) + newb - _b;
    //_E(j) = _E(j) + (ainew - _a(i))*_train_y(j)*_kernal_val(j,i) + (ajnew - _a(j))*_train_y(j)*_kernal_val(j,j) + newb - _b;
    // update _E(i) with support
    double ei = 0.0;
    double ej = 0.0;
    for(auto it = _sup_vec.begin();it!=_sup_vec.end();it++){
        ei+=_train_y(*it)*_a(*it)*_kernal_val(i,*it);
        ej+=_train_y(*it)*_a(*it)*_kernal_val(j,*it);
    }
    _E(i) = ei + newb - _train_y(i);
    _E(j) = ej + newb - _train_y(j);
    _b = newb;
}

void SVM::_update_a(int i, double val) {
    if(_zero_a.find(i) != _zero_a.end() && val > 0 + _toler){
        _zero_a.erase(i);
        _sup_vec.insert(i);
    }else if(_sup_vec.find(i) != _sup_vec.end() && val <= 0+_toler){
        _zero_a.insert(i);
        _sup_vec.erase(i);
    }
}

void SVM::_smo(){
    while(true){
        int i = _find_first();
        if(i<0){
            break;
        }
        int j = _find_second(i);
        if(j<0){
            break;
        }
        double eta = _kernal_val(i,i) + _kernal_val(j,j) - 2*_kernal_val(i,j);
        if(eta == 0){
            continue;
        }
        double ajnew = _a(j) + _train_y(j)*(_E(i)-_E(j))/eta;
        ajnew = _clip_a(i,j,ajnew);
        double ainew = _a(i) + _train_y(i)*_train_y(j)*(ajnew - _a(j));

        double diff  = abs(ainew - _a(i)) + abs(ajnew - _a(j));
        _update_b_E(i,j,ainew,ajnew);
        _a(i) = ainew;
        _a(j) = ajnew;

        if(_check_stop()){
            break;
        }
    }
}

bool SVM::_check_stop() {
    double sum;
    for(auto it=_sup_vec.begin();it!=_sup_vec.end();it++){
        sum+=_train_y(*it);
    }
    if(abs(sum) > _epsilon){
        return false;
    }
    return true;
}

void SVM::fit(const MatrixXd& x, const VectorXd& y) {
    _init(x,y);
    _smo();
}

VectorXd SVM::predict(const MatrixXd& x) {
    assert(x.cols() == this->_d);
    int nums = x.rows();
    VectorXd y_preds(nums);
    for(int i = 0;i<nums;i++){
        double y = 0.0;
        for(int j = 0;j< this->_n;j++){
            y+=_a(j) * _train_y(j) * _kernal->val(x.row(i),_train_x.row(j));
        }
        y_preds(i) = (y+_b)>0 ? 1 : 0;
    }
    return y_preds;
}

double SVM::score(const MatrixXd &x, const VectorXd &y) {
    VectorXd y_preds = predict(x);
    assert(y_preds.rows() == y.rows());
    double error = 0.0;
    for(int i = 0 ;i<y_preds.rows();i++){
        error += std::abs(y_preds(i) - y(i));
    }

    return error/y.rows();

}