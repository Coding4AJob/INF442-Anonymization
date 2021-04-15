#include "LogisticRegression.hpp"

int main(){
    Matrix<double,Dynamic,Dynamic> X = loadEigen("eng.testa.representation.csv",sampleNum,attriNum);
    Matrix<double,Dynamic,1> Y = loadEigen("eng.testa.true_labels.csv",sampleNum,1);
    Matrix<double,Dynamic,1> B = MatrixXd::Ones(sampleNum,1);
    MatrixXd X_bar(X.rows(), X.cols()+B.cols()); // <-- D(A.rows() + B.rows(), ...)
    X_bar << X, B;
    // Matrix<double,Dynamic,Dynamic> X_bar;
    // X_bar << X , MatrixXd::Ones(Dynamic,1) ;
    Matrix<double,Dynamic,1> beta = MatrixXd::Zero(attriNum+1,1);
    
    cout << "Loading Done!" << endl;
    cout << "Start Trainning: " << endl;

    int max_iter = 50;
    double mu = 0.001; // 学习率
    double alpha = 0.0001; // L2正则项的权重
    for(int i=0;i<max_iter;i++){
        beta -= mu*theta_gradient(X_bar,Y,beta,alpha);
        // cout << beta << endl;
        // cout << "==========" << endl;
        // if(i%(max_iter/100)==0)
            // cout << i/(max_iter/100) << "% Accomplished" << endl; 
        cout << i << "/" << max_iter << endl;
    }

    cout << "Start Evaluating" << endl;

    Matrix<double,Dynamic,1> y_pred = predict(X_bar,beta);
    cout << "Accuracy " << accuracy(y_pred,Y) << endl;
    // for(int i=0;i<attriNum+1;i++){
    //     cout<<beta(i,0)<<",";
    // }
    return 0;
}