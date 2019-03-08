/** @file SRRender2DSingle_IFace.cpp
 *  @author Mark J. Olah (mjo\@cs.unm DOT edu)
 *  @date 2014-2019
 *  @brief The entry point for SRRender2DDouble_IFace mexFunction.
 *
 */
#include "SRRender_IFace.h"

SRRender2D_IFace<float,uint32_t> iface; /**< Global iface object provides a iface.mexFunction */

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
