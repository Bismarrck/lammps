//
// Created by Xin Chen on 2019-07-02.
//

#include <string>
#include <iostream>
#include "virtual_atom_approach.h"

using namespace LAMMPS_NS;

#define REAL_ATOM_START 1

/* ---------------------------------------------------------------------- */

VirtualAtomMap::VirtualAtomMap(Memory *memory, int n_elements,
                               int *max_occurs, int inum, int *itypes)
{
    int i = 0;
    int i_old = 0;
    int i_ele = 0;
    int i_gsl = 0;
    int *delta = nullptr;

    mask = nullptr;
    element_map = nullptr;
    offsets = nullptr;
    index_map = nullptr;
    reverse_map = nullptr;
    splits = nullptr;

    _n_elements = n_elements;
    _max_occurs = max_occurs;
    _itypes = itypes;
    _inum = inum;
    _memory = memory;

    // self.max_vap_n_atoms = sum(max_occurs.values()) + istart
    n_atoms_vap = REAL_ATOM_START;
    for (i = 0; i < n_elements; i++) {
        n_atoms_vap += max_occurs[i];
    }

    // mask = np.zeros(self.max_vap_n_atoms, dtype=bool)
    _memory->create(mask, n_atoms_vap, "pair:vap:mask");
    _memory->create(index_map, inum, "pair:vap:index_map");
    _memory->create(reverse_map, n_atoms_vap, "pair:vap:reverse_map");
    _memory->create(offsets, n_elements + 1, "pair:vap:offsets");
    _memory->create(splits, n_elements + 1, "pair:vap:splits");

    // Initialize `delta` to all zeros.
    _memory->create(delta, n_elements + 1, "pair:vap:delta");
    for (i = 0; i < n_elements + 1; i++) {
        delta[i] = 0;
    }

    // self.splits = np.array([1, ] + [max_occurs[e] for e in elements])
    splits[0] = 1;
    for (i = 1; i < n_elements + 1; i++) {
        splits[i] = max_occurs[i - 1];
    }

    // offsets = np.cumsum([max_occurs[e] for e in elements])[:-1]
    // offsets = np.insert(offsets, 0, 0)
    offsets[0] = 1;
    for (i = 1; i < n_elements + 1; i++) {
        offsets[i] = offsets[i - 1] + max_occurs[i - 1];
    }

    for (i = 0; i < inum; i++) {
        i_old = i;
        // `itypes` is a list and its values ranges from 1 to `inum`.
        // So we can use its values as indices directly.
        i_ele = itypes[i];
        i_gsl = offsets[i_ele - 1] + delta[i_ele];
        index_map[i_old] = i_gsl;
        delta[i_ele] += 1;
        mask[i_gsl] = true;
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

/* ---------------------------------------------------------------------- */

void print_int_array(const std::string& title, const int *array, const int num,
        int max_per_line)
{
    max_per_line = MAX(max_per_line, 1);

    std::cout << "* " << title << ":" << std::endl;
    for (int i = 0; i < num; i++) {
        std::cout << " " << array[i];
        if (i && i % max_per_line == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

void VirtualAtomMap::print()
{
    std::cout << "----------------" << std::endl;
    std::cout << "Virtual-Atom Map" << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << "N_atoms_vap: " << n_atoms_vap << std::endl;
    print_int_array("Splits", splits, _n_elements + 1, 20);
    print_int_array("Offsets", offsets, _n_elements + 1, 20);
    print_int_array("IndexMap", index_map, _inum, 20);
    print_int_array("ReverseMap", reverse_map, n_atoms_vap, 20);
    std::cout << std::endl;
}

/* ---------------------------------------------------------------------- */

double VirtualAtomMap::memory_usage()
{
    auto bytes = (double)sizeof(int);
    auto n = (n_atoms_vap * 2 + _inum + 2 * (_n_elements + 1));
    return bytes * n;
}

/* ---------------------------------------------------------------------- */

// The deallocation method.
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
