/*
 *  quadtree.h
 *  Header file for a quadtree.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 */

#ifndef QUADTREE_H
#define QUADTREE_H

using namespace std;

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }
static inline double abs(double x) { return (x < .0 ? -x : x); }

class Cell {

public:
    double x;
    double y;
    double hw;
    double hh;
    bool   containsPoint(double point[]);
};


class QuadTree
{
    int no_dims;
    static const int QT_MAX_DIMS = 3;
    static const int QT_NODE_CAPACITY = 1;

    // A buffer we use when doing force computations
    double buff[QT_MAX_DIMS];

    // Properties of this node in the tree
    QuadTree* parent;
    bool is_leaf;
    int size;
    double cum_size;

    // Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
    Cell boundary;

    // Indices in this quad tree node, corresponding center-of-mass, and list of all children
    double* data;
    double* weights;
    double center_of_mass[QT_MAX_DIMS];
    int index[QT_NODE_CAPACITY];

    // Children
    QuadTree* northWest;
    QuadTree* northEast;
    QuadTree* southWest;
    QuadTree* southEast;

public:
    QuadTree(double* inp_data, double* weights, int N, int no_dims);
    QuadTree(double* inp_data, double* weights, int no_dims, double inp_x, double inp_y, double inp_hw, double inp_hh);
    QuadTree(double* inp_data, double* weights, int N, int no_dims, double inp_x, double inp_y, double inp_hw, double inp_hh);
    QuadTree(QuadTree* inp_parent, double* inp_data, double* weights, int N, int no_dims, double inp_x, double inp_y, double inp_hw, double inp_hh);
    QuadTree(QuadTree* inp_parent, double* inp_data, double* weights, int no_dims, double inp_x, double inp_y, double inp_hw, double inp_hh);
    ~QuadTree();
    void setData(double* inp_data);
    QuadTree* getParent();
    void construct(Cell boundary);
    bool insert(int new_index);
    void subdivide();
    bool isCorrect();
    void rebuildTree();
    void getAllIndices(int* indices);
    int getDepth();
    void computeNonEdgeForces(int point_index, double theta, double neg_f[], double* sum_Q);
    void computeEdgeForces(int* row_P, int* col_P, double* val_P, int N, double* pos_f);
    void print();

private:
    void init(QuadTree* inp_parent, double* inp_data, double* weights, double inp_x, double inp_y, double inp_hw, double inp_hh);
    void fill(int N);
    int getAllIndices(int* indices, int loc);
    bool isChild(int test_index, int start, int end);
    double getWeight(int idx) { if (weights != NULL) return weights[idx]; else return 1.; }
};

#endif
