/** @file SRRender.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration and inline and templated functions for SRRender.
 *
 * Rendering of SR emitter localizations
 */

#ifndef SRRENDER_SRRENDER_H
#define SRRENDER_SRRENDER_H

#include <BacktraceException/BacktraceException.h>
#include <armadillo>

namespace srrender {

using SRRenderError = backtrace_exception::BacktraceException;

/**
 * 
 * Points format.  Row-oriented each row is a point, each column is a property
 * 2D renderHist Columns: [I X Y]
 * 2D renderGauss Columns:[I X Y sigmaX sigmaY]
 * 2D renderHistMovie Columns: [I X Y Frame]  - Frame is 0-indexed
 * 2D renderGaussMovie Columns:[I X Y sigmaX sigmaY Frame] - Frame is 0-indexed
 * 
 * The 'size' parameter gives the size of the entire field of view to be rendered in units 
 * corresponding to the points format vectors.
 * 
 */
template<class FloatT=float, class IdxT=uint32_t>
class SRRender2D{
public:
    using IVecT = arma::Col<IdxT>;
    using VecT = arma::Col<FloatT>;
    using ImageT = arma::Mat<FloatT>;
    using MovieT = arma::Cube<FloatT>;
    using EmitterVecT = arma::Mat<FloatT>;
    static const FloatT DefaultSigmaAccuracy; //Default number of sigmas to render Gaussian at

    static void renderHist(const EmitterVecT &points, const VecT &roi, ImageT &im);
    static void renderGauss(const EmitterVecT &points, const VecT &roi, ImageT &im, FloatT sigmaAccuracy=DefaultSigmaAccuracy);
    static void renderHistMovie(const EmitterVecT &points, const VecT &roi, MovieT &im);
    static void renderGaussMovie(const EmitterVecT &points, const VecT &roi, MovieT &im, FloatT sigmaAccuracy=DefaultSigmaAccuracy);
//     static void checkPoints(const EmitterVecT &points);
private:
    static const FloatT normexp; // 1/sqrt(2);

    static void fill_stencil(IdxT size, FloatT x, FloatT sigma, VecT& stencil);
    static void renderHistSingle(const EmitterVecT &points, const VecT &roi, ImageT &im);
    static void renderHistParallel(const EmitterVecT &points, const VecT &roi, ImageT &im);
    static void renderGaussSingle(const EmitterVecT &points, const VecT &roi, ImageT &im, FloatT sigmaAccuracy);
    static void renderGaussParallel(const EmitterVecT &points, const VecT &roi, ImageT &im, FloatT sigmaAccuracy);
};

} /* namespace srrender */

#endif /* SRRENDER_SRRENDER_H */
