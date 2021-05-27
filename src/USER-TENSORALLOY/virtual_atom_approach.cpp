//
// Created by Xin Chen on 2019-07-02.
//

#include "virtual_atom_approach.h"
#include "memory.h"

using namespace LAMMPS_NS;

#define REAL_ATOM_START 1

/* ----------------------------------------------------------------------
   Initialization.
------------------------------------------------------------------------- */

VirtualAtomMap::VirtualAtomMap(LAMMPS *lmp, int num_elements) : Pointers(lmp) {
  n_atoms_vap = 0;
  n_elements = num_elements;

  atom_masks = nullptr;
  element_map = nullptr;
  offsets = nullptr;
  local_to_vap_map = nullptr;
  vap_to_local_map = nullptr;
  splits = nullptr;

  max_occurs = nullptr;
  curr_nlocal = 0;
  curr_itypes = nullptr;
  is_domain_changed = false;

  total_bytes = 0.0;
}

/* ----------------------------------------------------------------------
   Build the virtual-atom map.
------------------------------------------------------------------------- */

void VirtualAtomMap::build(const int nlocal, const int *itypes) {
  int i;
  int local_index;
  int atom_index;
  int gsl_index;
  int *delta = nullptr;
  int n_symbols_vap = n_elements + 1;

  // Build `max_occurs`
  is_domain_changed = build_max_occurs(nlocal, itypes);
  if (!is_domain_changed) {
    return;
  }

  // self.max_vap_n_atoms = sum(max_occurs.values()) + istart
  n_atoms_vap = REAL_ATOM_START;
  for (i = 1; i < n_symbols_vap; i++) {
    n_atoms_vap += max_occurs[i];
  }

  // mask = np.zeros(self.max_vap_n_atoms, dtype=bool)
  if (atom_masks) {
    memory->destroy(atom_masks);
    memory->destroy(local_to_vap_map);
    memory->destroy(vap_to_local_map);
    memory->destroy(offsets);
    memory->destroy(splits);
  }
  memory->create(atom_masks, n_atoms_vap, "tensoralloy:vap:mask");
  memory->create(local_to_vap_map, nlocal, "tensoralloy:vap:local2vap");
  memory->create(vap_to_local_map, n_atoms_vap,
                 "tensoralloy:vap:vap2local");
  memory->create(offsets, n_symbols_vap, "tensoralloy:vap:offsets");
  memory->create(splits, n_symbols_vap, "tensoralloy:vap:splits");
  total_bytes = static_cast<double>(sizeof(int32)) *
                (n_atoms_vap * 2 + nlocal * 2 + n_symbols_vap * 2);

  // Initialize `mask` to all zeros
  for (i = 0; i < n_atoms_vap; i++)
    atom_masks[i] = 0.0;

  // Initialize `delta` to all zeros.
  memory->create(delta, n_symbols_vap, "tensoralloy:vap:delta");
  for (i = 0; i < n_symbols_vap; i++) {
    delta[i] = 0;
  }

  // self.splits = np.array([1, ] + [max_occurs[e] for e in elements])
  splits[0] = 1;
  for (i = 1; i < n_symbols_vap; i++) {
    splits[i] = max_occurs[i];
  }

  // offsets = np.cumsum([max_occurs[e] for e in elements])[:-1]
  // offsets = np.insert(offsets, 0, 0)
  offsets[0] = 1;
  for (i = 1; i < n_symbols_vap; i++) {
    offsets[i] = offsets[i - 1] + max_occurs[i];
  }

  for (i = 0; i < nlocal; i++) {
    local_index = i;
    // `itypes` is a list and its values ranges from 1 to `inum`.
    // So we can use its values as indices directly.
    atom_index = itypes[i];
    gsl_index = offsets[atom_index - 1] + delta[atom_index];
    local_to_vap_map[local_index] = gsl_index;
    delta[atom_index] += 1;
    atom_masks[gsl_index] = 1.0;
  }

  // `delta` is no longer needed.
  memory->destroy(delta);

  // vap_to_local_map = {v: k - 1 for k, v in index_map.items()}
  for (i = 0; i < n_atoms_vap; i++) {
    vap_to_local_map[i] = -1;
  }
  for (i = 0; i < nlocal; i++) {
    vap_to_local_map[local_to_vap_map[i]] = i;
  }
}

/* ----------------------------------------------------------------------
   Compute `max_occurs`.
------------------------------------------------------------------------- */

bool VirtualAtomMap::build_max_occurs(int nlocal, const int *atom_types) {
  bool is_changed = false;
  if (curr_nlocal != nlocal) {
    is_changed = true;
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (curr_itypes[i] != atom_types[i]) {
        is_changed = true;
        break;
      }
    }
  }

  int size = n_elements + 1;
  if (max_occurs == nullptr) {
    memory->create(max_occurs, size, "tensoralloy:vap:max_occurs");
  }

  if (is_changed) {
    curr_nlocal = nlocal;
    curr_itypes =
        memory->grow(curr_itypes, nlocal, "tensoralloy:vap:curr_itypes");
    for (int i = 0; i < nlocal; i++) {
      curr_itypes[i] = atom_types[i];
    }
    for (int i = 0; i < size; i++) {
      max_occurs[i] = 0;
    }
    for (int i = 0; i < nlocal; i++) {
      max_occurs[atom_types[i]] += 1;
    }
    for (int i = 1; i < size; i++) {
      if (max_occurs[i] == 0) {
        max_occurs[i] = 1;
      }
    }
  }
  return is_changed;
}

/* ----------------------------------------------------------------------
   Deallocation.
------------------------------------------------------------------------- */

VirtualAtomMap::~VirtualAtomMap() {
  // All arrays were allocated with `memory->create`, so `delete` should
  // not be used here.
  memory->destroy(element_map);
  memory->destroy(offsets);
  memory->destroy(local_to_vap_map);
  memory->destroy(splits);
  memory->destroy(vap_to_local_map);
  memory->destroy(atom_masks);
  memory->destroy(curr_itypes);
  memory->destroy(max_occurs);
}
