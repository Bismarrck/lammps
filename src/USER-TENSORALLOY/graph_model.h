//
// Created by Xin Chen on 2019-07-26.
//

#ifndef LMP_TENSORALLOY_GRAPH_MODEL_H
#define LMP_TENSORALLOY_GRAPH_MODEL_H

#include "pair.h"

#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace LAMMPS_NS {

    using tensorflow::string;
    using tensorflow::Tensor;
    using tensorflow::Status;
    using tensorflow::int32;

    class GraphModel {

    public:
        GraphModel();
        ~GraphModel() {};

        Status read(const Tensor&, const string&, const std::vector<string>&);
        const bool is_initialized() const { return decoded && max_occurs_initialized; }
        const int get_n_elements() const { return n_elements; }
        const bool angular() const { return use_angular; }
        const double get_cutoff() { return rc; }
        const unsigned int get_max_occur(int index) const { return max_occurs[index]; }
        const int get_ndim(bool is_angular=false) const {
            if (is_angular) {
                return n_gamma * n_beta * n_zeta;
            } else {
                return n_omega * n_eta;
            }
        }

        void compute_max_occurs(const int natoms, const int* atom_types);
        void print();

    protected:
        std::vector<unsigned int> max_occurs;
        bool use_angular;
        double rc;
        int n_elements;
        std::vector<string> symbols;
        int n_eta;
        int n_omega;
        int n_beta;
        int n_gamma;
        int n_zeta;
        string filename;

    private:
        bool decoded;
        bool max_occurs_initialized;

    };
}

#endif //LMP_TENSORALLOY_GRAPH_MODEL_H
