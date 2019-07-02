//
// Created by Xin Chen on 2019-07-02.
//

#ifndef LMP_TENSORALLOY_VAP_H
#define LMP_TENSORALLOY_VAP_H

#include "pair.h"
#include "memory.h"

namespace LAMMPS_NS {

    class VirtualAtomMap {
    public:

        VirtualAtomMap(Memory *, int, int *, int, int *);

        ~VirtualAtomMap();

        void print();

        double memory_usage();

    private:
        // Variables from outside
        int _n_elements;
        int *_max_occurs;
        int _inum;
        int *_itypes;

    protected:
        // Internal variables
        int n_atoms_vap;
        int *element_map;
        int *offsets;
        int *index_map;
        int *reverse_map;
        int *splits;
        bool *mask;

        Memory *_memory;
    };
}

#endif //LMP_TENSORALLOY_VAP_H
