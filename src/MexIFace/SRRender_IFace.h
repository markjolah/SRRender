/** @file SRRender_IFace.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration and inline and templated functions for SRRender2D_IFace.
 */

#ifndef SRRENDER_SRRENDER_IFACE_H
#define SRRENDER_SRRENDER_IFACE_H

#include <cstdint>
#include <functional>
#include <thread>
#include <omp.h>

#include "MexIFace/MexIFace.h"
#include "SRRender/SRRender.h"

template<class FloatT=float, class IndexT=uint32_t>
class SRRender2D_IFace : public mexiface::MexIFace,
                         public mexiface::MexIFaceHandler<srrender::SRRender2D<FloatT,IndexT>>
{
public:
    SRRender2D_IFace();
private:
    using mexiface::MexIFaceHandler<srrender::SRRender2D<FloatT,IndexT>>::obj;

    //Constructor
    void objConstruct() override;

    //Exposed static methods
    void objRenderHist();
    void objRenderGauss();
    void objRenderHistMovie();
    void objRenderGaussMovie();
};

template<class FloatT, class IndexT>
SRRender2D_IFace<FloatT,IndexT>::SRRender2D_IFace()
{
    omp_set_num_threads(std::thread::hardware_concurrency());

    staticmethodmap["renderHist"] = std::bind(&SRRender2D_IFace::objRenderHist, this);
    staticmethodmap["renderGauss"] = std::bind(&SRRender2D_IFace::objRenderGauss, this);
    staticmethodmap["renderHistMovie"] = std::bind(&SRRender2D_IFace::objRenderHistMovie, this);
    staticmethodmap["renderGaussMovie"] = std::bind(&SRRender2D_IFace::objRenderGaussMovie, this);
}

template<class FloatT, class IndexT>
void SRRender2D_IFace<FloatT,IndexT>::objConstruct()
{
    checkNumArgs(1,0);
    this->outputHandle(new srrender::SRRender2D<FloatT,IndexT>());
}

template<class FloatT, class IndexT>
void SRRender2D_IFace<FloatT,IndexT>::objRenderHist()
{
    // [in] points: mat with n rows and 3 (or more) columns [I, x, y]
    // [in] roi=[xmin, xmax, ymin, ymax] - 4-element vector giving effective region of interest that
    //                                     for the images generated.  This should be exact to the boundaries of im.
    // [in/out] im: a image with arbitrary size but should match aspect ratio of the internal size.  This
    //          image will be modified in-place.

    checkNumArgs(0,3);
    auto points = getMat<FloatT>();
    auto roi = getVec<FloatT>();
    auto im = getMat<FloatT>();
    obj->renderHist(points,roi,im);
}

template<class FloatT, class IndexT>
void SRRender2D_IFace<FloatT,IndexT>::objRenderGauss()
{
    // [in] points: mat with n rows and 5 (or more) columns [I, x, y, sigma_x, sigma_y]
    // [in] roi=[xmin, xmax, ymin, ymax] - 4-element vector giving effective region of interest that
    //                                     for the images generated.  This should be exact to the boundaries of im.
    // [in] sigmaAccuracy: floating point >0.  Gives accuracy at which gaussians will be rendered
    // [in/out] im: a image with arbitrary size but should match aspect ratio of the internal size.  This
    //          image will be modified in-place.

    checkNumArgs(0,4);
    auto points = getMat<FloatT>();
    auto roi = getVec<FloatT>();
    auto sigmaAccuracy = getAsFloat<FloatT>();
    auto im = getMat<FloatT>();
    obj->renderGauss(points,roi,im, sigmaAccuracy);
}

template<class FloatT, class IndexT>
void SRRender2D_IFace<FloatT,IndexT>::objRenderHistMovie()
{
    // [in] points: mat with n rows and 6 (or more) columns [I, x, y, sigma_x, sigma_y, frameIdx]
    //              frame indexs are 0-based.  sigma_x and sigma_y are ignored but must be included,
    // [in] roi=[xmin, xmax, ymin, ymax] - 4-element vector giving effective region of interest that
    //                                     for the images generated.  This should be exact to the boundaries of im.
    // [in/out] im: a image sequence (movie) with arbitrary size but should match aspect ratio of the internal size.  This
    //          image will be modified in-place.  The number of frames should match the frame indexs from points

    checkNumArgs(0,3);
    auto points = getMat<FloatT>();
    auto roi = getVec<FloatT>();
    auto im = getCube<FloatT>();
    obj->renderHistMovie(points,roi,im);
}


template<class FloatT, class IndexT>
void SRRender2D_IFace<FloatT,IndexT>::objRenderGaussMovie()
{
    // [in] points: mat with n rows and 6 (or more) columns [I, x, y, sigma_x, sigma_y, frameIdx]
    //              frame indexs are 0-based.
    // [in] roi=[xmin, xmax, ymin, ymax] - 4-element vector giving effective region of interest that
    //                                     for the images generated.  This should be exact to the boundaries of im.
    // [in/out] im: a image sequence (movie) with arbitrary size but should match aspect ratio of the internal size.  This
    //          image will be modified in-place.  The number of frames should match the frame indexs from points

    checkNumArgs(0,3);
    auto points = getMat<FloatT>();
    auto roi = getVec<FloatT>();
    auto im = getCube<FloatT>();
    obj->renderGaussMovie(points,roi,im);
}

#endif /* SRRENDER_SRRENDER_IFACE_H */
