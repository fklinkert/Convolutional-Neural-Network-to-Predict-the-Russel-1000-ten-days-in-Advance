/*
* Group 2 HW3
* main file
* Haoxuan Huang(hh2773)
* Jiaxin Lin(jl5304)
* Federico Klinkert(fgk2106)
* Mateo Gomez(mg4010)
*/


#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include "multi_init.cpp"
using namespace std;
using namespace std::chrono;


const int NUM_LAYER = 4;
const int BATCH_SIZE = 50;
const double RATIO = 0.8;
const double L_RATE = 3e-4;
const int ROW_TEST = int(ROW*(1-RATIO));
const int ROW_TR = int(ROW*RATIO)+1;

class Layer {

	public:

		Layer(int input_size, int output, Layer* layer);

		~Layer();

		double** forward(double** input);

		void backward(double** w_gradient, double* b_gradient);

		Layer* prev_layer;

		double** fx;

		double** output;

		double** weight;

		double* bias;

		int in_size;

		int out_size;

	private:
		double ** mw;
		double ** vw;
		double * mb;
		double * vb;
};

double** Relu(double** fx, int row, int col) {
	double** output = new double*[row];
	for (int i = 0; i < row; i++)
		output[i] = new double[col];

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			output[i][j] = fx[i][j] < 0 ? 0 : fx[i][j];
		}
	}
	return output;
}

double** deriv_R(double** fx, int row, int col) {
	double** d_out = new double*[row];
	for (int i = 0; i < row; i++)
		d_out[i] = new double[col];

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			d_out[i][j] = fx[i][j] <= 0 ? 0 : 1;
		}
	}
	return d_out;
}

Layer::Layer(int input_size, int output_size, Layer* layer) {
	in_size = input_size;
	out_size = output_size;
	prev_layer = layer;
	weight = MatRandomInit(output_size, input_size);
	bias = VecRandomInit(output_size);
	fx = NULL;
	output = NULL;
	mw = zeros(output_size, input_size);
	vw = zeros(output_size, input_size);
	mb = zeros(output_size);
	vb = zeros(output_size);
}

Layer::~Layer() {
	for (int row = 0; row < out_size; row++) {
		delete [] weight[row];
		delete [] fx[row];
		delete [] output[row];
		delete [] mw[row];
		delete [] vw[row];
	}
	delete [] bias;
	delete [] mb;
	delete [] vb;
}

double** Layer::forward(double** input) {
	double** WX = matDotProduct(weight, input, out_size, in_size, in_size, BATCH_SIZE);
	if (fx)
		deleteMat(fx, out_size);
	if (output)
		deleteMat(output, out_size);
	fx = MatAddVec(WX, bias, out_size, BATCH_SIZE);
	output = Relu(fx, out_size, BATCH_SIZE);
	deleteMat(WX, out_size);
	return output;
}

void Layer::backward(double** w_gradient, double* b_gradient) {

	// Adam Optimization
	double beta1 = 0.9;
	double beta2 = 0.999;
	double eps = 1e-5;
	// w update

	double** newmw = doubleMat_copy(beta1, mw, out_size, in_size);

	double** b1w = doubleMat_copy(1-beta1, w_gradient, out_size, in_size);
	deleteMat(mw, out_size);
	mw = matPlus(newmw, b1w, out_size, in_size);
	deleteMat(newmw, out_size);
	deleteMat(b1w, out_size);


	double** sqMat = matSquare(w_gradient, out_size, in_size);
	double** newvw = doubleMat_copy(beta2, vw, out_size, in_size);
	double** b2w = doubleMat_copy(1-beta2, sqMat, out_size, in_size);
	deleteMat(vw, out_size);
	deleteMat(sqMat, out_size);
	vw = matPlus(newvw, b2w, out_size, in_size);
	deleteMat(newvw, out_size);
	deleteMat(b2w, out_size);

	double** sqrMat = matSqrt(vw, out_size, in_size);
	MatAddDouble(eps, sqrMat, out_size, in_size);
	double** div = matDivi(mw, sqrMat, out_size, in_size);
	deleteMat(sqrMat, out_size);
	matMinus(weight, div, L_RATE, out_size, in_size);
	deleteMat(div, out_size);

	// b update
	double* newmb = doubleVec_copy(beta1, mb, out_size);
	double* b1b = doubleVec_copy(1-beta1, b_gradient, out_size);
	delete [] mb;
	mb = vecPlus(newmb, b1b, out_size);
	delete [] newmb;
	delete [] b1b;

	double* sqVec = vecSquare(b_gradient, out_size);
	double* newvb = doubleVec_copy(beta2, vb, out_size);
	double* b2b = doubleVec_copy(1-beta2, sqVec, out_size);
	delete [] sqVec;
	delete [] vb;
	vb = vecPlus(newvb, b2b, out_size);
	delete [] newvb;
	delete [] b2b;

	double* sqrVec = vecSqrt(vb, out_size);
	VecAddDouble(eps, sqrVec, out_size);
	double* divVec = vecDivi(mb, sqrVec, out_size);
	delete [] sqrVec;
	vecMinus(bias, divVec, L_RATE, out_size);
	delete [] divVec;
}

double calc_loss(double* batch_Y, double* batch_Yhat){
	double* loss_vec = vecMinus(batch_Y, batch_Yhat, BATCH_SIZE);
	double loss = 0.0;
	for (int i = 0; i < BATCH_SIZE; i++) {
		loss += pow(loss_vec[i], 2.0);
	}
	delete [] loss_vec;
	return loss / BATCH_SIZE;
}

void train(Layer** layer_array, double** data, int epoch){
	double total_loss = 0.0;
	
	for (int i = 0; i < epoch; i++){
		cout << i << endl;
		mt19937 g(static_cast<uint32_t>(time(0)));
		shuffle(data, data + ROW_TR, g);

		double** X = new double*[ROW_TR];
		double* Y = new double[ROW_TR];
		for (int a = 0; a < ROW_TR; a++) {
			X[a] = new double[COL - 1];
		}
		split_XY(data, X, Y, ROW_TR, COL);
		

		int start; int end;
		
		for (int itera = 0; itera < ceil(ROW_TR / BATCH_SIZE); itera++){
			start = BATCH_SIZE * itera;
			if (BATCH_SIZE * (itera + 1) > ROW_TR)
				end = ROW_TR;
			else
				end = BATCH_SIZE * (itera + 1);
			
			double** prebatch_X = new double*[end-start];
			double** prebatch_Yhat = new double*[end-start];
			double* batch_Y = new double[end-start];


			for (int r = 0, a = start; a < end; r++, a++){
				prebatch_X[r] = X[a];
				prebatch_Yhat[r] = X[a];
				batch_Y[r] =Y[a];
			}



			double** batch_X = transpose(prebatch_X, BATCH_SIZE, COL-1);
			double** batch_Yhat = transpose(prebatch_Yhat, BATCH_SIZE, COL-1);
			delete [] prebatch_X;
			delete [] prebatch_Yhat;

			
			// forward
			for(int l = 0; l < NUM_LAYER; l++) {
				batch_Yhat = layer_array[l]->forward(batch_Yhat);
			}


			// calculate square loss
			double loss = calc_loss(batch_Y, *batch_Yhat);
			total_loss += loss;

			// backpropogation
			double** first_loss = linearLoss(batch_Yhat, batch_Y, BATCH_SIZE);

			double** last_w = NULL;
			int last_w_row = 0;
			int last_w_col = 0;
			double** gradient_b = NULL;
			double** gradient_w = NULL;
			bool first_round = true;

			
			for (int l = NUM_LAYER - 1; l >= 0; l--) {
				Layer* ly = layer_array[l];
				if (first_round) {
					double** deriv_Relu = deriv_R(ly->fx, ly->out_size, BATCH_SIZE);
					gradient_b = matMulti(first_loss, deriv_Relu, 1, BATCH_SIZE);
					first_round = false;
					deleteMat(deriv_Relu, ly->out_size);
				}
				else {
					double** last_w_T = transpose(last_w, last_w_row, last_w_col);
					double** matProduct = matDotProduct(last_w_T, gradient_b, last_w_col, last_w_row, last_w_row, BATCH_SIZE);
					deleteMat(last_w_T, last_w_col);
					deleteMat(gradient_b, last_w_row);
					double** deriv_Relu = deriv_R(ly->fx, ly->out_size, BATCH_SIZE);
					gradient_b = matMulti(matProduct, deriv_Relu, last_w_col, BATCH_SIZE);
					deleteMat(matProduct, last_w_col);
					deleteMat(deriv_Relu, ly->out_size);
					//cout << "first else" << endl;
				}
					
				if (!ly->prev_layer) {
					double** batch_X_T = transpose(batch_X, BATCH_SIZE, COL-1);
					deleteMat(gradient_w, layer_array[l+1]->out_size);
					gradient_w = matDotProduct(gradient_b, batch_X_T, ly->out_size, BATCH_SIZE, BATCH_SIZE, COL-1);
					//cout << "second if" << endl;
				}
				else {
					if (gradient_w)
						deleteMat(gradient_w, layer_array[l+1]->out_size);
					gradient_w = matDotProduct(gradient_b, transpose(ly->prev_layer->output, ly->prev_layer->out_size, BATCH_SIZE),
						ly->out_size, BATCH_SIZE, BATCH_SIZE, ly->prev_layer->out_size);

				}
				double* b = new double[ly->out_size];
				for (int row = 0; row < ly->out_size; row++) {
					double sum = 0.0;
					for (int col = 0; col < BATCH_SIZE; col++) {
						sum += gradient_b[row][col];
					}
					b[row] = sum / BATCH_SIZE;
				}
				
				
				ly->backward(gradient_w, b);

				delete [] b;
				last_w = ly->weight;
				last_w_row = ly->out_size;
				last_w_col = ly->in_size;
			}

			deleteMat(batch_X, BATCH_SIZE);
			
			delete [] batch_Y;

			deleteMat(first_loss, 1);
			
		}
		deleteMat(X, ROW_TR);
		delete [] Y;
		cout << total_loss / ceil(ROW_TR / BATCH_SIZE) << endl;
		total_loss = 0.0;
	}
	return;

}
void test(Layer** layer_array, double** data) {
	double** X = new double*[ROW_TEST];
	double* Y = new double[ROW_TEST];
	for (int a = 0; a < ROW_TEST; a++) {
		X[a] = new double[COL - 1];
	}
	split_XY(data, X, Y, ROW_TEST, COL);
	double Y_hat[ROW_TEST];
	for (int i = 0; i < ROW_TEST; i++) {
		double** prebatch_Yhat = new double*[1];
		prebatch_Yhat[0] = X[i];
		double** batch_Yhat = transpose(prebatch_Yhat, 1, COL-1);
		delete [] prebatch_Yhat;
		// forward
		for(int l = 0; l < NUM_LAYER; l++) {
			batch_Yhat = layer_array[l]->forward(batch_Yhat);
		}
		Y_hat[i] = batch_Yhat[0][0];
		cout << "Y_hat: " << batch_Yhat[0][0] << " Y: " << Y[i] << endl;
	}
	fstream file;
	file.open("write.csv");
  	for (int i=0;i<ROW_TEST;i++)
  	{
    	file << Y_hat[i]<<",";
  	}
  	for (int i=0;i<ROW_TEST;i++)
  	{
    	file << Y[i]<<",";
  	}
  	file.close();
	deleteMat(X, ROW_TEST);
	delete[] Y;

}


void print(double** m, int r, int c){
	for(int i = 0; i < r; i++){
		for(int j = 0; j < c; j++){
			cout << m[i][j] << "  ";
		}
		cout << endl;    
	}
	cout << endl;
}

void print(double* v, int c){
	for(int i = 0; i < c; i++)
		cout << v[i] << "  ";
	cout << endl << endl;
}

int main(int argc, const char * argv[]) {
	// data reading
	if (argc != 2){
		cerr << "usage: ./a.out <filename>\n";
		exit(1);
	}
	cout << fixed;
	cout << setprecision(8);

	srand (time(NULL));

	// read file and store
	string filename = argv[1];

	double** testing = new double*[ROW_TEST];
	for(int i = 0; i < ROW_TEST; ++i)
		testing[i] = new double[COL];
	double** training = new double*[ROW_TR];
	for(int i = 0; i < ROW_TR; ++i)
		training[i] = new double[COL];

	//normalize(double** data, int row, int col)
	getTrainingAndTesting(training, testing, filename);
	
	Layer** layer_array = new Layer*[NUM_LAYER];
	layer_array[0] = new Layer(486, 500, 0);
	layer_array[1] = new Layer(500, 250, layer_array[0]);
	layer_array[2] = new Layer(250, 125, layer_array[1]);
	layer_array[3] = new Layer(125, 1, layer_array[2]);

	int epoch = 100;
	train(layer_array, training, epoch);
	test(layer_array, testing);
	
	// //delete
	for (int i = 0; i < NUM_LAYER; i++)
		delete layer_array[i];
	delete [] layer_array;
	deleteMat(testing,ROW_TEST);
	deleteMat(training,ROW_TR);
	return 0;
}
