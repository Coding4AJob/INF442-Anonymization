/*
提供了两种导入数据(double)的方法，一种需要提前知道的数据的尺寸(m*n)，另一种不需要
double ** load(string filename, int m, int n)
void load(string filename, vector<vector<double> > data)
*/
// #include <iostream>
// #include <string>
// #include <vector>
// #include <fstream>
// #include <sstream>
// using namespace std;

#include <Eigen/Dense>
#include "LoadData.hpp"

using namespace Eigen;

// 需要指定数据的行和列
Matrix<double, Dynamic, Dynamic> loadEigen(string filename, int m ,int n){
    Matrix<double, Dynamic, Dynamic> ans(m,n);
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

void load(string filename, vector<vector<double> >& data){
    ifstream inFile(filename, ios::in);
	string lineStr; //每一行的结果

    // 每行的结果会存在lineStr中
	while (getline(inFile, lineStr))
	{
		// 存成二维表结构
		stringstream ss(lineStr);
		string str;
        vector<double> doubleArray;
		// 按照逗号分隔
		while (getline(ss, str, ',')){
            doubleArray.push_back(atof(str.c_str()));
        }
        data.push_back(doubleArray);
	}

}

// 测试用
void test(){
    // 第一种导入方法测试
    // vector<vector<double> > data;
    // LoadData::load("testX.csv",data);
    // for(int i=0;i<data.size();i++){
    //     for(int j=0;j<data.at(0).size();j++){
    //         cout << data.at(i).at(j) << " ";
    //     }
    //     cout << "\n";
    // }

    // 第二种：导入m行n列的数据
    // int m=3,n=1;
    // double ** ans = load("testX.csv",m,n);
    // for(int i=0;i<m;i++){
    //     for(int j=0;j<n;j++){
    //         cout << ans[i][j] << " ";
    //     }
    //     cout << "\n";
    // }
    
    // 第三种：导入为Matrix的形式
    int m = 100, n = 1;
    Matrix<double,Dynamic,Dynamic> ans = loadEigen("testX.csv",m,n);
    // ans.resize(m,n);
    cout << ans << endl;
    cout << "Shape " << "( " << ans.rows() << ", " << ans.cols() << " )" << endl;
}

int main(){
    cout << "Hello World" << endl;
    test();
    return -1;
}