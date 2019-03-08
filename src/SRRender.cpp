/** @file SRRender.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2014-2019
 * @brief The class definition and template Specializations for SRRender.
 *
 * Rendering of SR emitter localizations
 */
#include <omp.h>
#include "SRRender/SRRender.h"

namespace srrender {

template<class FloatT, class IdxT>
const FloatT SRRender2D<FloatT,IdxT>::DefaultSigmaAccuracy = 5.;

template<class FloatT, class IdxT>
const FloatT SRRender2D<FloatT,IdxT>::normexp = 1/sqrt(2);


template<class FloatT, class IdxT>
void SRRender2D<FloatT,IdxT>::renderHist(const EmitterVecT &points, const VecT &roi, ImageT &im)
{
//     #ifdef DEBUG
//     checkPoints(points);
//     #endif
    if (points.n_rows >= sqrt(im.n_elem)) {
        renderHistParallel(points,roi,im);
    } else {
        renderHistSingle(points,roi,im);
    }
}

template<class FloatT, class IdxT>
void SRRender2D<FloatT,IdxT>::renderHistSingle(const EmitterVecT &points, const VecT &roi, ImageT &im)
{
    IdxT pixelsX =  static_cast<IdxT>(im.n_cols); //number of output pixels in the X direction (across rows)
    IdxT pixelsY =  static_cast<IdxT>(im.n_rows); //number of output pixels in the Y direction (down columns)
    FloatT xmin = roi(0);
    FloatT ymin = roi(2);
    FloatT sizeRatioX = static_cast<FloatT>(pixelsX) / (roi(1)-roi(0));
    FloatT sizeRatioY = static_cast<FloatT>(pixelsY) / (roi(3)-roi(2));
    for(unsigned n=0; n<points.n_rows; n++){
        IdxT ix = static_cast<IdxT>((points(n,1)-xmin)*sizeRatioX);
        IdxT iy = static_cast<IdxT>((points(n,2)-ymin)*sizeRatioY);
        if(0<=ix && ix<pixelsX && 0<=iy && iy<pixelsY) im(iy,ix) += points(n,0);
    }
}

template<class FloatT, class IdxT>
void SRRender2D<FloatT,IdxT>::renderHistParallel(const EmitterVecT &points, const VecT &roi, ImageT &im)
{
    IdxT pixelsX =  static_cast<IdxT>(im.n_cols); //number of output pixels in the X direction (across rows)
    IdxT pixelsY =  static_cast<IdxT>(im.n_rows); //number of output pixels in the Y direction (down columns)
    FloatT xmin = roi(0);
    FloatT ymin = roi(2);
    FloatT sizeRatioX = static_cast<FloatT>(pixelsX) / (roi(1)-roi(0));
    FloatT sizeRatioY = static_cast<FloatT>(pixelsY) / (roi(3)-roi(2));

    IdxT max_threads = omp_get_max_threads();
    arma::field<ImageT> histF(max_threads);
    IdxT num_threads;
    #pragma omp parallel
    {
        ImageT hist(im.n_rows,im.n_cols);
        hist.zeros();
        num_threads=omp_get_num_threads(); //Save number of threads actually run
        #pragma omp for
        for(unsigned n=0; n<points.n_rows; n++){
            IdxT ix = static_cast<IdxT>((points(n,1)-xmin)*sizeRatioX);
            IdxT iy = static_cast<IdxT>((points(n,2)-ymin)*sizeRatioY);
            if(0<=ix && ix<pixelsX && 0<=iy && iy<pixelsY) hist(iy,ix) += points(n,0); //intensity
        }
        histF(omp_get_thread_num()) = hist;
    }
    //Parellelize sum of individual historgrams over columns
    #pragma omp parallel for
    for(IdxT x=0; x<pixelsX; x++) for(IdxT y=0; y<pixelsY; y++) {
        FloatT sum = 0.0;
        for(IdxT n=0;n<num_threads; n++) sum += histF(n)(y,x);
        im(y,x) = sum;
    }
}

template<class FloatT, class IdxT>
void SRRender2D<FloatT,IdxT>::renderHistMovie(const EmitterVecT &points, const VecT &roi, MovieT &im)
{
    IdxT pixelsX =  static_cast<IdxT>(im.n_cols); //number of output pixels in the X direction (across rows)
    IdxT pixelsY =  static_cast<IdxT>(im.n_rows); //number of output pixels in the Y direction (down columns)
    FloatT xmin = roi(0);
    FloatT ymin = roi(2);
    FloatT sizeRatioX = static_cast<FloatT>(pixelsX) / (roi(1)-roi(0));
    FloatT sizeRatioY = static_cast<FloatT>(pixelsY) / (roi(3)-roi(2));

    #pragma omp parallel
    {
        IdxT num_threads = omp_get_num_threads(); //Number we actually created my be less than max
        IdxT tid = omp_get_thread_num();
        for(unsigned n=0; n<points.n_rows; n++){
            IdxT frame = points(n,5);
            if (frame%num_threads != tid) continue;
            IdxT ix = static_cast<IdxT>((points(n,1)-xmin)*sizeRatioX);
            IdxT iy = static_cast<IdxT>((points(n,2)-ymin)*sizeRatioY);
            if(0<=ix && ix<pixelsX && 0<=iy && iy<pixelsY) im(iy,ix,frame) += points(n,0);
        }
    }
}

template<class FloatT, class IdxT>
void SRRender2D<FloatT,IdxT>::renderGauss(const EmitterVecT &points, const VecT &roi, ImageT &im, FloatT sigmaAccuracy)
{
//     #ifdef DEBUG
//     checkPoints(points);
//     #endif
    if (points.n_rows >= sqrt(im.n_elem)) {
        renderGaussParallel(points,roi,im,sigmaAccuracy);
    } else {
        renderGaussSingle(points,roi,im,sigmaAccuracy);
    }
}

template<class FloatT, class IdxT>
void SRRender2D<FloatT,IdxT>::renderGaussSingle(const EmitterVecT &points, const VecT &roi, ImageT &im, FloatT sigmaAccuracy)
{
    IdxT pixelsX =  static_cast<IdxT>(im.n_cols); //number of output pixels in the X direction (across rows)
    IdxT pixelsY =  static_cast<IdxT>(im.n_rows); //number of output pixels in the Y direction (down columns)
    FloatT imageXmin = roi(0);
    FloatT imageYmin = roi(2);
    FloatT sizeRatioX = static_cast<FloatT>(pixelsX) / (roi(1)-roi(0));
    FloatT sizeRatioY = static_cast<FloatT>(pixelsY) / (roi(3)-roi(2));
    IdxT N = static_cast<IdxT>(points.n_rows);
    VecT xStencil(pixelsX), yStencil(pixelsY);
    im.zeros();
    for(IdxT n=0; n<N; n++) {
        FloatT X = (points(n,1)-imageXmin)*sizeRatioX;
        FloatT sigmaX = points(n,3)*sizeRatioX;
        IdxT xp = static_cast<IdxT>(X); // 0 <= xp < pixelsX
        IdxT xhw = static_cast<IdxT>(0.5+sigmaAccuracy*sigmaX); //halfwidth for gaussian X
        IdxT xmin = xp <= xhw ? 0 : xp-xhw;
        IdxT xmax = std::min(pixelsX-1,xp+xhw);
        IdxT xspn = xmax-xmin+1;

        FloatT Y = (points(n,2)-imageYmin)*sizeRatioY;
        FloatT sigmaY = points(n,4)*sizeRatioY;
        IdxT yp = static_cast<IdxT>(Y); // 0 <= yp < pixelsY
        IdxT yhw = static_cast<IdxT>(0.5+sigmaAccuracy*sigmaY); //halfwidth for gauissian Y
        IdxT ymin = yp <= yhw ? 0 : yp-yhw;
        IdxT ymax = std::min(pixelsY-1,yp+yhw);
        IdxT yspn = ymax-ymin+1;

        if(xspn<=0 || yspn<=0) continue;
        fill_stencil(xspn, X-xmin, sigmaX, xStencil);
        fill_stencil(yspn, Y-ymin, sigmaY, yStencil);
        FloatT I = points(n,0);
        for(IdxT x=0; x<xspn; x++) xStencil(x)*=I; //Pre-multiply by I;
        //Copy in image of new gaussian
        for(IdxT x=xmin; x<=xmax; x++) for(IdxT y=ymin; y<=ymax; y++)
            im(y,x) += xStencil(x-xmin) * yStencil(y-ymin);
    }
}

template<class FloatT, class IdxT>
void SRRender2D<FloatT,IdxT>::renderGaussParallel(const EmitterVecT &points, const VecT &roi, ImageT &final_image, FloatT sigmaAccuracy)
{
    IdxT pixelsX =  static_cast<IdxT>(final_image.n_cols); //number of output pixels in the X direction (across rows)
    IdxT pixelsY =  static_cast<IdxT>(final_image.n_rows); //number of output pixels in the Y direction (down columns)
    FloatT imageXmin = roi(0);
    FloatT imageYmin = roi(2);
    FloatT sizeRatioX = static_cast<FloatT>(pixelsX) / (roi(1)-roi(0));
    FloatT sizeRatioY = static_cast<FloatT>(pixelsY) / (roi(3)-roi(2));
    IdxT N = static_cast<IdxT>(points.n_rows);
    IdxT max_threads = omp_get_max_threads();
    IdxT num_threads;
    arma::field<ImageT> imStack(max_threads);
    #pragma omp parallel
    {
        ImageT im(pixelsY,pixelsX,arma::fill::zeros);
        num_threads = omp_get_num_threads(); //Number we actually created my be less than max
        VecT xStencil(pixelsX), yStencil(pixelsY);
        num_threads = omp_get_num_threads(); //Number we actually created my be less than max
        #pragma omp for
        for(IdxT n=0; n<N; n++) {
            FloatT X = (points(n,1)-imageXmin)*sizeRatioX;
            FloatT sigmaX = points(n,3)*sizeRatioX;
            IdxT xp = static_cast<IdxT>(X); // 0 <= xp < pixelsX
            IdxT xhw = static_cast<IdxT>(0.5+sigmaAccuracy*sigmaX); //halfwidth for gaussian X
            IdxT xmin = xp <= xhw ? 0 : xp-xhw;
            IdxT xmax = std::min(pixelsX-1,xp+xhw);
            IdxT xspn = xmax-xmin+1;

            FloatT Y = (points(n,2)-imageYmin)*sizeRatioY;
            FloatT sigmaY = points(n,4)*sizeRatioY;
            IdxT yp = static_cast<IdxT>(Y); // 0 <= yp < pixelsY
            IdxT yhw = static_cast<IdxT>(0.5+sigmaAccuracy*sigmaY); //halfwidth for gauissian Y
            IdxT ymin = yp <= yhw ? 0 : yp-yhw;
            IdxT ymax = std::min(pixelsY-1,yp+yhw);
            IdxT yspn = ymax-ymin+1;

            if(xspn<=0 || yspn<=0) continue;
            fill_stencil(xspn, X-xmin, sigmaX, xStencil);
            fill_stencil(yspn, Y-ymin, sigmaY, yStencil);
            FloatT I = points(n,0);
            for(IdxT x=0; x<xspn; x++) xStencil(x)*=I; //Pre-multiply by I;
            //Copy in image of new gaussian
            for(IdxT x=xmin; x<=xmax; x++) for(IdxT y=ymin; y<=ymax; y++)
                im(y,x) += xStencil(x-xmin) * yStencil(y-ymin);
        }
        imStack(omp_get_thread_num()) = im;
    }
    //Parellelize sum of individual historgrams over columns
    #pragma omp parallel for
    for(IdxT x=0; x<pixelsX; x++) for(IdxT y=0; y<pixelsY; y++) {
        FloatT sum = 0.0;
        for(IdxT n=0;n<num_threads; n++) sum += imStack(n)(y,x);
        final_image(y,x) = sum;
    }
}

template<class FloatT, class IdxT>
void SRRender2D<FloatT,IdxT>::renderGaussMovie(const EmitterVecT &points, const VecT &roi, MovieT &im, FloatT sigmaAccuracy)
{
    IdxT pixelsX =  static_cast<IdxT>(im.n_cols); //number of output pixels in the X direction (across rows)
    IdxT pixelsY =  static_cast<IdxT>(im.n_rows); //number of output pixels in the Y direction (down columns)
    FloatT imageXmin = roi(0);
    FloatT imageYmin = roi(2);
    FloatT sizeRatioX = static_cast<FloatT>(pixelsX) / (roi(1)-roi(0));
    FloatT sizeRatioY = static_cast<FloatT>(pixelsY) / (roi(3)-roi(2));
    IdxT N = static_cast<IdxT>(points.n_rows);
    #pragma omp parallel
    {
        VecT xStencil(pixelsX), yStencil(pixelsY);
        IdxT num_threads = omp_get_num_threads(); //Number we actually created my be less than max
        IdxT tid = omp_get_thread_num();
        for(IdxT n=0; n<N; n++) {
            IdxT frame = points(n,5);
            if (frame%num_threads != tid) continue;
            FloatT X = (points(n,1)-imageXmin)*sizeRatioX;
            FloatT sigmaX = points(n,3)*sizeRatioX;
            IdxT xp = static_cast<IdxT>(X); // 0 <= xp < pixelsX
            IdxT xhw = static_cast<IdxT>(0.5+sigmaAccuracy*sigmaX); //halfwidth for gaussian X
            IdxT xmin = xp <= xhw ? 0 : xp-xhw;
            IdxT xmax = std::min(pixelsX-1,xp+xhw);
            IdxT xspn = xmax-xmin+1;

            FloatT Y = (points(n,2)-imageYmin)*sizeRatioY;
            FloatT sigmaY = points(n,4)*sizeRatioY;
            IdxT yp = static_cast<IdxT>(Y); // 0 <= yp < pixelsY
            IdxT yhw = static_cast<IdxT>(0.5+sigmaAccuracy*sigmaY); //halfwidth for gauissian Y
            IdxT ymin = yp <= yhw ? 0 : yp-yhw;
            IdxT ymax = std::min(pixelsY-1,yp+yhw);
            IdxT yspn = ymax-ymin+1;

            if(xspn<=0 || yspn<=0) continue;
            fill_stencil(xspn, X-xmin, sigmaX, xStencil);
            fill_stencil(yspn, Y-ymin, sigmaY, yStencil);
            FloatT I = points(n,0);
            for(IdxT x=0; x<xspn; x++) xStencil(x)*=I; //Pre-multiply by I;
            //Copy in image of new gaussian
            for(IdxT x=xmin; x<=xmax; x++) for(IdxT y=ymin; y<=ymax; y++)
                im(y,x,frame) += xStencil(x-xmin) * yStencil(y-ymin);
        }
    }
}

template<class FloatT, class IdxT>
void SRRender2D<FloatT,IdxT>::fill_stencil(IdxT size, FloatT x, FloatT sigma, VecT& stencil)
{
    FloatT norm = normexp/sigma;
    FloatT derf = erf(-norm*x);
    for(IdxT i=0;i<size;i++) {
        FloatT last_derf = derf;
        derf = erf(norm*((i+1)-x));
        stencil(i) = 0.5*(derf-last_derf);
    }
}
/*
template<class FloatT, class IdxT>
void SRRender2D<FloatT,IdxT>::checkPoints(const EmitterVecT &points) const
{
    IdxT nPoints = points.n_rows;
    for(IdxT n=0; n<nPoints; n++){
        assert(points(n,0)>0);
        assert(points(n,1)>=0 && points(n,1)<size(0));
        assert(points(n,2)>=0 && points(n,2)<size(1));
        assert(points(n,3)>0);
        assert(points(n,4)>0);
    }
}*/

/* Explicit Template Instantiation */
template class SRRender2D<float>;
template class SRRender2D<double>;


/*
template<class FloatT, class IdxT>
SRRenderHS<FloatT>::SRRenderHS(const VecT &_size) : size(_size) {}



template<class FloatT, class IdxT>
void SRRenderHS<FloatT>::renderHist(const EmitterVecT &points, ImageT &im)
{
    for(unsigned n=0; n<points.n_cols; n++) im(points(n,2), points(n,3), points(n,4)) += points(n,1);
}*/

} /* namespace srrender */
