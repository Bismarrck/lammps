//
// Created by Xin Chen on 2019-06-11.
//

#ifdef PAIR_CLASS

PairStyle(tensoralloy, PairTensorAlloy)

#else

#ifndef LMP_PAIR_TENSORALLOY_H
#define LMP_PAIR_TENSORALLOY_H

#include <atom.h>
#include "pair.h"
#include "virtual_atom_approach.h"
#include "graph_model.h"

#include "tensorflow/core/public/session.h"


namespace LAMMPS_NS {

    using tensorflow::Tensor;
    using tensorflow::string;
    using tensorflow::int32;
    using tensorflow::Status;
    using tensorflow::DataType;
    using std::vector;

    class PairTensorAlloy : public Pair {
    public:
        explicit PairTensorAlloy(class LAMMPS *);
        ~PairTensorAlloy() override;

        void compute(int, int) override;
        void settings(int, char **) override;
        void coeff(int, char **) override;
        void init_style() override;
        double init_one(int, int) override;

        double memory_usage() override;

    protected:

        GraphModel *graph_model;
        double cutforcesq, cutmax;

        int32 **radial_interactions;
        int32 **radial_counters;

        VirtualAtomMap *vap;

        template <typename T> double update_cell ();
        template <typename T> void run(int eflag, int vflag, DataType dtype);
        template <typename T> void allocate_with_dtype(DataType dtype);

        /*
         * Return the atom index in the local frame.
         * vap->get_index_map()[local_idx] will map the local index to VAP index.
         * */
        int inline get_local_idx(const unsigned int i) {
            return atom->tag[i] - 1;
        }

        // Original Pair variables
        int nmax;
        void allocate();

    private:

        bool serial_mode;

        // Electron temperature (eV)
        double etemp;

        Tensor *h_tensor;
        Tensor *R_tensor;
        Tensor *volume_tensor;
        Tensor *n_atoms_vap_tensor;
        Tensor *nnl_max_tensor;
        Tensor *pulay_stress_tensor;
        Tensor *etemperature_tensor;
        Tensor *atom_mask_tensor;
        Tensor *row_splits_tensor;

        double dynamic_bytes;
        double tensors_memory_usage();
    };
}

#endif
#endif
