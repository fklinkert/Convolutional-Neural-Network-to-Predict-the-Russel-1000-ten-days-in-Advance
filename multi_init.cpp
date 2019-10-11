#include <iostream>
#include <cmath>
#include <ctime>
#include <random>
#include <fstream>

using namespace std;

const int ROW = 746;
const int COL = 487;

void normalize(double** data, int row, int col){
	for(int c = 1; c < col; c++){
		double sum = 0;
		double mean = 0;
		double variance = 0;
		double standard_deviation = 0;
		for(int r = 0; r < row; r++){
			sum += data[r][c];
		}
		mean = sum/row;
		for(int r=0; r < row; r++){
			variance += pow((data[r][c]-mean),2);
		}
		standard_deviation = sqrt(variance/(row - 1));

		for(int r = 0; r < row; r++){
			data[r][c] = (data[r][c]-mean)/standard_deviation;
		}
	}
}

void split_XY(double** data, double** X, double* Y, int row, int col) {
	for (int i = 0; i < row; i++) {
		Y[i] = data[i][0];
		for (int j = 1; j < col; j++) {
			X[i][j - 1] = data[i][j];
		}
	}
}

void deleteMat(double** m, int row){
	for(int i = 0; i < row; i++)
		delete[] m[i];
	delete[] m;
}

void getYX(double** price, string line, char delimeter, int row){
	// split the whole line by delimeter and store them into array of doubles
	int i = 0;
	int start = 0;
	int count = 0;

	while(i < line.length()){
		if(line[i] == delimeter){
			string str = line.substr(start, i-start);
			start = i+1;
			double val = 0;
			if(str != "") val = stod(str);
			if(count == 0 && row >= 10) price[row-10][0] = val;
			if(count > 1 &&  row < ROW) {
				if(row == 30){
				}
				price[row][count-1]= val;
			}
			count ++;
		}
		i ++;
	}
	if(row < ROW){
		string str = "";
		if(line.length()-1 > start)
			str = line.substr(start, line.length()-start);
		double val = 0;
		if(str != "") val = stod(str);
		price[row][count-1]= val;
	}
}

void getData(double** price, string filename){
	// store data into index and price
	ifstream file(filename);

	string line = "";

	int i = 0;
	while (getline(file, line)){
		if(i > 0) { 
			getYX(price, line, ',', i-1);
		}
		i ++;

	}
	file.close();
	normalize(price, ROW, COL);
}

void getTrainingAndTesting(double** training, double** testing, string filename){
	double** data = new double*[ROW];
	for(int i = 0; i < ROW; ++i)
		data[i] = new double[COL];
	int count = 1;
	getData(data, filename);
	mt19937 g(static_cast<uint32_t>(time(0)));
	shuffle(data, data + ROW, g);
	
	for(int i = 0; i < ROW; i++){
		if(i < count*5-1){
			for(int j = 0; j < COL; j++)
				training[(count-1)*4 + i - (count-1)*5][j] = data[i][j];
		}
		else{
			for(int j = 0; j < COL; j++){
				testing[count-1][j] = data[i][j];
			}
			count ++;
		}
	}
	


	deleteMat(data, ROW);
}

double** transpose(double** mat, int row, int col) {
	double** res = new double*[col];
	for (int i = 0; i < col; i++) {
		res[i] = new double[row];
	}

	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			res[i][j] = mat[j][i];
		}
	}
	return res;
}

double** matPlus(double** mat1, double** mat2, int row, int col){
	double** res = new double*[row];

	for(int i = 0; i < row; ++i)
		res[i] = new double[col];

	for(int i = 0; i < row; i++) {
		for(int j = 0; j < row; j++)
			res[i][j] = mat1[i][j] + mat2[i][j];
	}
	return res;
}

double** matMinus(double** mat1, double** mat2, int row, int col){
	double** res = new double*[row];
	for(int i = 0; i < row; ++i)
		res[i] = new double[col];

	for(int i = 0; i < row; i++) {
		for(int j = 0; j < row; j++) {
			res[i][j] = mat1[i][j] - mat2[i][j];
		}
	}
	return res;
}

void matMinus(double** mat1, double** mat2, double L_Rate, int row, int col){
	for(int i = 0; i < row; i++) {
		for(int j = 0; j < row; j++) {
			mat1[i][j] -= L_Rate * mat2[i][j];
		}
	}
	return;
}

double** matSquare(double** mat, int row, int col) {
	double** res = new double*[row];
	for(int i = 0; i < row; ++i)
		res[i] = new double[col];

	for(int i = 0; i < row; i++) {
		for(int j = 0; j < row; j++) {
			res[i][j] = mat[i][j] * mat[i][j];
		}
	}
	return res;
}

double** matSqrt(double** mat, int row, int col) {
	double** res = new double*[row];
	for(int i = 0; i < row; ++i)
		res[i] = new double[col];

	for(int i = 0; i < row; i++) {
		for(int j = 0; j < row; j++) {
			res[i][j] = sqrt(mat[i][j]);
		}
	}
	return res;
}

double* vecPlus(double* vec1, double* vec2, int row){
	double* res = new double[row];
	for(int i = 0; i < row; i++)
		res[i] = vec1[i] + vec2[i];
	return res;
}

double* vecMinus(double* vec1, double* vec2, int row){
	double* res = new double[row];
	for(int i = 0; i < row; i++)
		res[i] = vec1[i] - vec2[i];
	return res;
}

void vecMinus(double* vec1, double* vec2, double L_Rate, int row){
	for(int i = 0; i < row; i++)
		vec1[i] -= L_Rate * vec2[i];
	return;
}

double* vecSquare(double* vec, int col) {
	double* res = new double[col];
	for(int i = 0; i < col; i++)
		res[i] = vec[i] * vec[i];
	return res;
}

double* vecSqrt(double* vec, int col) {
	double* res = new double[col];
	for(int i = 0; i < col; i++)
		res[i] = sqrt(vec[i]);
	return res;
}

void VecAddDouble(double eps, double* vec, int col) {
	for (int i = 0; i < col; i++)
		vec[i] += eps;
}

double* vecDivi(double* vec1, double* vec2, int col) {
	double* res = new double[col];
	for(int i = 0; i < col; i++)
		res[i] = vec1[i] / vec2[i];
	return res;
}

double** matDotProduct(double** mat1, double** mat2, int row1, int col1, int row2, int col2){
	if(col1 != row2){
		cerr << "Matrices' dimensions not match\n";
		exit(1);
	}

	double** res = new double*[row1];
	for(int i = 0; i < row1; ++i)
		res[i] = new double[col2];

	for(int r = 0; r < row1; r++){
		for(int c = 0; c < col2; c++){
			double sum = 0;
			for(int i = 0; i < row2; i++){
				sum += mat1[r][i] * mat2[i][c];
			}
			res[r][c] = sum;
		}
	}

	return res;
}

double** matMulti(double** mat1, double** mat2, int row, int col){

	double** res = new double*[row];
	for (int i = 0; i < row; i++) {
		res[i] = new double[col];
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			res[i][j] = mat1[i][j] * mat2[i][j];
		}
	}

	return res;
}

double** matDivi(double** mat1, double** mat2, int row, int col){
	double** res = new double*[row];
	for (int i = 0; i < row; i++) {
		res[i] = new double[col];
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			res[i][j] = mat1[i][j] / mat2[i][j];
		}
	}

	return res;
}

double* vecMulti(double* vec1, double* vec2, int row1, int row2){
	if(row1 != row2){
		cerr << "Vectors' dimensions not match\n";
		exit(1);
	}
	
	double* res = new double[row1];

	for(int i = 0; i < row1; i++){
		res[i] = vec1[i] * vec2[i];
	} 

	return res;
}

double** linearLoss(double** Mat, double* Vec, int col) {
	double** res = new double*[1];
	res[0] = new double[col];

	for(int i = 0; i < col; i++) {
		res[0][i] = Mat[0][i] - Vec[i];
	}
	return res;
}

void MatAddDouble(double eps, double** Mat, int row, int col) {
	for(int i = 0; i < row; i++) {
		for(int j = 0; j < col; j++) {
			Mat[i][j] += eps;
		}
	}
	return;
}

double** MatAddVec(double** Mat, double* Vec, int row, int col) {
	double** res = new double*[row];
	for (int i = 0; i < row; i++) {
		res[i] = new double[col];
	}

	for(int i = 0; i < row; i++) {
		for(int j = 0; j < col; j++) {
			res[i][j] = Mat[i][j] + Vec[i];
		}
	}
	return res;
}

void doubleVec(double beta, double* Vec, int col) {
	for (int i = 0; i < col; i++)
		Vec[i] *= beta;
}

double* doubleVec_copy(double beta, double* Vec, int col) {
	double* res = new double[col];
	for (int i = 0; i < col; i++)
		res[i] = beta * Vec[i];
	return res;
}

void doubleMat(double beta, double** Mat, int row, int col) {

	for(int i = 0; i < row; i++) {
		for(int j = 0; j < col; j++) {
			Mat[i][j] *= beta;
		}
	}
	return;
}

double** doubleMat_copy(double beta, double** Mat, int row, int col) {
	double** res = new double*[row];
	for (int i = 0; i < row; i++) {
		res[i] = new double[col];
	}

	for(int i = 0; i < row; i++) {
		for(int j = 0; j < col; j++) {
			res[i][j] = Mat[i][j] * beta;
		}
	}
	return res;
}

double** MatRandomInit(int row, int col){
	random_device rd;
    mt19937_64 generator(rd());
    normal_distribution<> distribution(0.0, 1.0);
	double** res = new double*[row];
	//mt19937 g(static_cast<uint32_t>(time(0)));
	//default_random_engine generator;
	//normal_distribution<double> distribution(0.0,1.0);
	for(int i = 0; i < row; ++i)
		res[i] = new double[col];

	for(int i = 0; i < row; i++) {
		for(int j = 0; j < col; j++) {
			res[i][j] = distribution(generator) * sqrt(2.0/col);
		}
	}

	return res;
}

double* VecRandomInit(int col){
	double* res = new double[col];
	random_device rd;
    mt19937_64 generator(rd());
    normal_distribution<> distribution(0.0, 1.0);

	for(int i = 0; i < col; i++) {
		res[i] = distribution(generator) * sqrt(2.0);
	}
	
	return res;
}

double** zeros(int row, int col){
	double** data = new double*[row];
	for(int i = 0; i < row; i++){
		data[i] = new double[col];
		for(int j = 0; j < col; j++)
			data[i][j] = 0;
	}
	return data;
}

double* zeros(int n){
	double* data = new double[n];
	for(int i = 0; i < n; i++){
		data[i] = 0;
	}
	return data;
}

