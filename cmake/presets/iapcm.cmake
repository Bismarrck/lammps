# preset that turns on just a few, frequently used packages
# this will be compiled quickly and handle a lot of common inputs.

set(ALL_PACKAGES KSPACE MANYBODY SNAP USER-MISC SHOCK COMPRESS
    USER-TENSORALLOY)

foreach(PKG ${ALL_PACKAGES})
  set(PKG_${PKG} ON CACHE BOOL "" FORCE)
endforeach()
