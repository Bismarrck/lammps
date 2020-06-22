//
// Created by Xin Chen on 2020/6/22.
//

#include "tensoralloy_utils.h"

/*
 The szudzik pairing function for two scalars. This pairing function supports
 negative numbers.

 See Also
 --------
 https://gist.github.com/TheGreatRambler/048f4b38ca561e6566e0e0f6e71b7739
 */
int szudzik_pairing(const int x, const int y) {
    int xx = x >= 0 ? x * 2 : -2 * x - 1;
    int yy = y >= 0 ? y * 2 : -2 * y - 1;
    return xx >= yy ? xx * xx + xx + yy : yy * yy + xx;
}
