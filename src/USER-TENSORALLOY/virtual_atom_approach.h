//
// Created by Xin Chen on 2019-07-02.
//

#ifndef LMP_TENSORALLOY_VAP_H
#define LMP_TENSORALLOY_VAP_H

#include <tensorflow/core/platform/default/integral_types.h>
#include "pair.h"
#include "memory.h"

namespace LAMMPS_NS {

    class VirtualAtomMap {
    public:
        VirtualAtomMap();
        VirtualAtomMap(Memory *, int, const int *, int, int *);
        ~VirtualAtomMap();

        int *get_row_splits() { return splits; }
        int *get_index_map() { return index_map; }
        float *get_atom_mask() { return mask; }
        int get_n_atoms_vap() { return n_atoms_vap; }

        void print();
        double memory_usage();

    private:
        // Variables from outside
        int _n_symbols;
        int _inum;
        int *_itypes;

    protected:
        // Internal variables
        int n_atoms_vap;
        int *element_map;
        int *offsets;
        int *index_map;
        int *reverse_map;
        tensorflow::int32 *splits;
        float *mask;

        Memory *_memory;
    };
}

#endif //LMP_TENSORALLOY_VAP_H
