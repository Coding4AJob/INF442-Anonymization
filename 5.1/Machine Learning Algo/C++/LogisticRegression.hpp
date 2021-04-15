#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

const double eps = 0.0001;   // 用于浮点数比较大小
const int sampleNum = 5000; // 样本数目,100
const int attriNum = 768;  // 一个样本特征数目,2


/*
// 用于导入数据 [这个函数本应该从LoadData中调用的，但是我没成功...]
double** load(string filename, int m, int n){
    // 初始化数组
    double** ans = (double **)malloc(m * sizeof(double *));
    for(int i=0;i<m;i++)
        ans[i] = (double *)malloc(n * sizeof(double));
    
    ifstream inFile(filename, ios::in);
	string lineStr; //每一行的结果
    for(int i=0;i<m;i++){
        getline(inFile, lineStr);
        stringstream ss(lineStr);
        string str;
        for(int j=0;j<n;j++){
            getline(ss, str, ',');
            ans[i][j] = atof(str.c_str());
        }
    }
    return ans;
}

// 损失函数
// X_bar sampleNum * (attriNum+1), y sampleNum*1, theta (attriNum+1)
// 但实际上输入的是 X sampleNum * attriNum，在计算的时候手动加入一项 -1 
double loss_theta(double ** X_bar,double ** y,double* theta){
    double * tmp = new double[sampleNum]();
    for(int i=0;i<sampleNum;i++){
        double xTheta = 0;
        for(int j=0;j<attriNum;j++){
            xTheta += X_bar[i][j]*theta[j];
        }
        xTheta -= theta[attriNum];
        tmp[i] = y[i][0]*xTheta;
    }
    double sum = 0;
    for(int i=0;i<sampleNum;i++)
        sum += log(1+exp(-tmp[i]));
    return (double)sum/sampleNum;
}

// 用来统计准确率
double score(double* orgLabel,double* newLabel){
    int sum = 0;
    for(int i=0;i<sampleNum;i++){
        if((orgLabel[i]-newLabel[i])<eps)
            sum+=1;
    }
    return (double)sum/sampleNum;
}

// 用来打印一个二维数组(测试用)
void printTest(double** tmp,int m,int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            cout << tmp[i][j] << " ";
        }
        cout << "\n";
    }
}

*/


// 需要指定数据的行和列
Matrix<double, Dynamic, Dynamic> loadEigen(string filename, int m ,int n){
    Matrix<double, Dynamic, Dynamic> ans = MatrixXd::Zero(m, n);;
    ifstream inFile(filename, ios::in);
	string lineStr; //每一行的结果
    for(int i=0;i<m;i++){
        getline(inFile, lineStr);
        stringstream ss(lineStr);
        string str;
        for(int j=0;j<n;j++){
            getline(ss, str, ',');
            ans(i,j) = atof(str.c_str());
        }
    }
    return ans;
}

/*
def theta_gradient(X_bar,y,beta,alpha=0):
    ans = np.zeros(beta.shape)+alpha*beta
    for i in range(X_bar.shape[0]):
        tmp = np.exp(beta.dot(X_bar[i]))
        ans += X_bar[i]*(y[i]-tmp/(1+tmp))
    return -ans
*/


Matrix<double,attriNum+1,1> theta_gradient(MatrixXd X_bar ,Matrix<double,sampleNum,1> Y,Matrix<double,attriNum+1,1> beta){
    Matrix<double,attriNum+1,1> ans;
    ans.fill(0);
    for(int i=0;i<sampleNum;i++){
        // cout << X_bar.row(i).rows() << "\t" << X_bar.row(i).cols() <<endl;
        double tmp = exp(X_bar.row(i)*beta);
        // cout << X_bar.row(i).transpose() << endl;
        ans += X_bar.row(i).transpose()*(Y(i,0)-tmp/(1+tmp));
    }
    return -ans;
}

// // 训练
void train(){

}

// // 预测
Matrix<double,sampleNum,1> predict(MatrixXd X_bar,Matrix<double,attriNum+1,1> beta){
    Matrix<double,sampleNum,1> y_pred;
    y_pred.fill(0);
    for(int i=0;i<sampleNum;i++){
        double tmp = 1/(1+exp(X_bar.row(i)*beta));
        // cout << tmp << endl;
        if(tmp <0.5)
           y_pred(i,0)= 1; // 否则默认是0
    }
    return y_pred;
}

double accuracy(Matrix<double,sampleNum,1> y_pred, Matrix<double,sampleNum,1> y){
    int num = 0;
    for(int i=0;i<sampleNum;i++)
        if((y_pred(i,0)-y(i,0))<eps)
            num+=1;
    cout << num << " correct out of " <<sampleNum << endl;
    return (double)num/sampleNum;
}