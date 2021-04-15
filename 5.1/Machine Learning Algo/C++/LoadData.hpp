#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;
static double **load(string filename, int m, int n);
static void load(string filename, vector<vector<double> > &data);
static void test();