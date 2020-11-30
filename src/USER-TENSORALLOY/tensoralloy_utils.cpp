//
// Created by Xin Chen on 2020/6/22.
//

#include "tensoralloy_utils.h"
#include <cstdlib>
#include <string>

using namespace LIBTENSORALLOY_NS;
using std::string;

#if !defined(LAMMPS_MEMALIGN) && !defined(_WIN32)
#define LAMMPS_MEMALIGN 64
#endif

/* ----------------------------------------------------------------------
   safe malloc
------------------------------------------------------------------------- */

void *Memory::smalloc(bigint nbytes, const char *name) {
  if (nbytes == 0)
    return nullptr;

#if defined(LAMMPS_MEMALIGN)
  void *ptr = nullptr;
  int retval = posix_memalign(&ptr, LAMMPS_MEMALIGN, nbytes);
  if (retval)
    ptr = nullptr;

#else
  void *ptr = malloc(nbytes);
#endif
  if (ptr == nullptr) {
    char buf[256];
    sprintf(buf, "Failed to alloc %lld bytes for array %s\n", nbytes, name);
    _logger(buf);
  }
  return ptr;
}

/* ----------------------------------------------------------------------
   safe realloc
------------------------------------------------------------------------- */

void *Memory::srealloc(void *ptr, bigint nbytes, const char *name) {
  if (nbytes == 0) {
    destroy(ptr);
    return nullptr;
  }

#if defined(LMP_USE_TBB_ALLOCATOR)
  ptr = scalable_aligned_realloc(ptr, nbytes, LAMMPS_MEMALIGN);
#elif defined(LMP_INTEL_NO_TBB) && defined(LAMMPS_MEMALIGN) &&                 \
    defined(__INTEL_COMPILER)

  ptr = realloc(ptr, nbytes);
  uintptr_t offset = ((uintptr_t)(const void *)(ptr)) % LAMMPS_MEMALIGN;
  if (offset) {
    void *optr = ptr;
    ptr = smalloc(nbytes, name);
#if defined(__APPLE__)
    memcpy(ptr, optr, MIN(nbytes, malloc_size(optr)));
#elif defined(_WIN32) || defined(__MINGW32__)
    memcpy(ptr, optr, MIN(nbytes, _msize(optr)));
#else
    memcpy(ptr, optr, MIN(nbytes, malloc_usable_size(optr)));
#endif
    free(optr);
  }
#else
  ptr = realloc(ptr, nbytes);
#endif
  if (ptr == nullptr) {
    char buf[256];
    sprintf(buf, "Failed to realloc %lld bytes for array %s\n", nbytes, name);
    _logger(buf);
  }
  return ptr;
}

/* ----------------------------------------------------------------------
   safe free
------------------------------------------------------------------------- */

void Memory::sfree(void *ptr) {
  if (ptr == nullptr)
    return;
#if defined(LMP_USE_TBB_ALLOCATOR)
  scalable_aligned_free(ptr);
#else
  free(ptr);
#endif
}

/* ----------------------------------------------------------------------
   erroneous usage of templated create/grow functions
------------------------------------------------------------------------- */

void Memory::fail(const char *name) {
  char buf[256];
  sprintf(buf, "Cannot create/grow a vector/array of pointers for %s\n", name);
  _logger(buf);
}
