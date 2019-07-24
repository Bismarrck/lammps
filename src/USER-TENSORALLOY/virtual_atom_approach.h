//
// Created by Xin Chen on 2019-07-02.
//

#ifndef LMP_TENSORALLOY_VAP_H
#define LMP_TENSORALLOY_VAP_H

#include <tensorflow/core/platform/default/integral_types.h>
#include "pair.h"
#include "memory.h"

namespace LAMMPS_NS {

    using tensorflow::int32;

    class VirtualAtomMap {
    public:
        VirtualAtomMap();
        VirtualAtomMap(Memory *, int, const int *, int, const int *);
        ~VirtualAtomMap();

        const int32 *get_row_splits() { return splits; }
        const int32 *get_index_map() { return index_map; }
        const int32 *get_reverse_map() { return reverse_map; }
        float *get_atom_mask() { return mask; }
        int get_n_atoms_vap() { return n_atoms_vap; }

        void print();
        double memory_usage();

    private:
        // Variables from outside
        int _n_symbols;
        int _inum;

    protected:
        // Internal variables
        int n_atoms_vap;
        int32 *element_map;
        int32 *offsets;
        int32 *index_map;
        int32 *reverse_map;
        int32 *splits;
        float *mask;

        Memory *_memory;
    };
}

#endif //LMP_TENSORALLOY_VAP_H
