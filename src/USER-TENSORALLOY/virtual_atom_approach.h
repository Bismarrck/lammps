//
// Created by Xin Chen on 2019-07-02.
//

#ifndef LMP_TENSORALLOY_VAP_H
#define LMP_TENSORALLOY_VAP_H

#include <tensorflow/core/platform/default/integral_types.h>
#include "pair.h"
#include "memory.h"
#include "graph_model.h"

namespace LAMMPS_NS {

    using tensorflow::int32;

    class VirtualAtomMap {
    public:
        explicit VirtualAtomMap(Memory *);
        ~VirtualAtomMap();

        void build(const GraphModel&, int, const int *);

        const int32 *get_row_splits() const { return splits; }
        const int32 *get_index_map() const { return index_map; }
        const int32 *get_reverse_map() const { return reverse_map; }
        const float *get_atom_mask() const { return mask; }
        const int get_n_atoms_vap() const { return n_atoms_vap; }

        const double memory_usage() const { return total_bytes; };

    protected:
        // Internal variables
        int32 n_atoms_vap;
        int32 *element_map;
        int32 *offsets;
        int32 *index_map;
        int32 *reverse_map;
        int32 *splits;
        float *mask;

        Memory *_memory;

    private:
        double total_bytes;

    };
}

#endif //LMP_TENSORALLOY_VAP_H
