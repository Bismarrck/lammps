//
// Created by Xin Chen on 2019-07-02.
//

#include <string>
#include <vector>
#include <iostream>
#include "virtual_atom_approach.h"

using namespace LAMMPS_NS;

#define REAL_ATOM_START 1

/* ----------------------------------------------------------------------
   Initialization.
------------------------------------------------------------------------- */

VirtualAtomMap::VirtualAtomMap(Memory *memory)
{
    n_atoms_vap = 0;
    _memory = memory;

    mask = nullptr;
    element_map = nullptr;
    offsets = nullptr;
    index_map = nullptr;
    reverse_map = nullptr;
    splits = nullptr;

    total_bytes = 0.0;
}

/* ----------------------------------------------------------------------
   Build the virtual-atom map.
------------------------------------------------------------------------- */

void VirtualAtomMap::build(
        const GraphModel &graph_model,
        const int inum,
        const int *itypes)
{
    assert(graph_model.is_initialized());

    int i = 0;
    int local_index = 0;
    int atom_index = 0;
    int gsl_index = 0;
    int *delta = nullptr;
    int n_symbols_vap = graph_model.get_n_elements() + 1;

    // self.max_vap_n_atoms = sum(max_occurs.values()) + istart
    n_atoms_vap = REAL_ATOM_START;
    for (i = 1; i < n_symbols_vap; i++) {
        n_atoms_vap += graph_model.get_max_occur(i);
    }

    // mask = np.zeros(self.max_vap_n_atoms, dtype=bool)
    _memory->create(mask, n_atoms_vap, "pair:vap:mask");
    _memory->create(index_map, inum, "pair:vap:index_map");
    _memory->create(reverse_map, n_atoms_vap, "pair:vap:reverse_map");
    _memory->create(offsets, n_symbols_vap, "pair:vap:offsets");
    _memory->create(splits, n_symbols_vap, "pair:vap:splits");
    total_bytes = sizeof(int32) * (n_atoms_vap * 2 + inum + n_symbols_vap * 2);

    // Initialize `mask` to all zeros
    for (i = 0; i < n_atoms_vap; i++)
        mask[i] = 0.0;

    // Initialize `delta` to all zeros.
    _memory->create(delta, n_symbols_vap, "pair:vap:delta");
    for (i = 0; i < n_symbols_vap; i++) {
        delta[i] = 0;
    }

    // self.splits = np.array([1, ] + [max_occurs[e] for e in elements])
    splits[0] = 1;
    for (i = 1; i < n_symbols_vap; i++) {
        splits[i] = graph_model.get_max_occur(i);
    }

    // offsets = np.cumsum([max_occurs[e] for e in elements])[:-1]
    // offsets = np.insert(offsets, 0, 0)
    offsets[0] = 1;
    for (i = 1; i < n_symbols_vap; i++) {
        offsets[i] = offsets[i - 1] + graph_model.get_max_occur(i);
    }

    for (i = 0; i < inum; i++) {
        local_index = i;
        // `itypes` is a list and its values ranges from 1 to `inum`.
        // So we can use its values as indices directly.
        atom_index = itypes[i];
        gsl_index = offsets[atom_index - 1] + delta[atom_index];
        index_map[local_index] = gsl_index;
        delta[atom_index] += 1;
        mask[gsl_index] = 1.0;
    }

    // `delta` is no longer needed.
    _memory->destroy(delta);
    delete [] delta;

    // reverse_map = {v: k - 1 for k, v in index_map.items()}
    for (i = 0; i < n_atoms_vap; i++) {
        reverse_map[i] = -1;
    }
    for (i = 0; i < inum; i++) {
        reverse_map[index_map[i]] = i;
    }
}

/* ----------------------------------------------------------------------
   Deallocation.
------------------------------------------------------------------------- */

VirtualAtomMap::~VirtualAtomMap()
{
    _memory->destroy(element_map);
    delete [] element_map;

    _memory->destroy(offsets);
    delete [] offsets;

    _memory->destroy(index_map);
    delete [] index_map;

    _memory->destroy(splits);
    delete [] splits;

    _memory->destroy(reverse_map);
    delete [] reverse_map;

    _memory->destroy(mask);
    delete [] mask;
}
