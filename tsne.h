/*
 *  tsne.h
 *  Header file for t-SNE.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 */


#ifndef TSNE_H
#define TSNE_H


static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }


class TSNE
{
public:
    TSNE(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta);
    ~TSNE();
    void run();
    void step();

    void symmetrizeMatrix(int** row_P, int** col_P, double** val_P, int N); // should be static?!


private:
    void computeGradient(double* P, int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);
    void computeExactGradient(double* P, double* Y, int N, int D, double* dC);
    double evaluateError(double* P, double* Y, int N);
    double evaluateError(int* row_P, int* col_P, double* val_P, double* Y, int N, double theta);
    void zeroMean(double* X, int N, int D);
    void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity);
    void computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, int K);
    void computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, double threshold);
    void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);

    // Learning parameters
    double theta;
    int max_iter, stop_lying_iter, mom_switch_iter;
    double momentum, final_momentum;
    double eta;

    // Current data
    int N, D;
    double *Y;
    int no_dims;

    // Learning state
    double* P; int* row_P; int* col_P; double* val_P;
    double* dY;
    double* uY;
    double* gains;

    bool exact;

};

#endif

