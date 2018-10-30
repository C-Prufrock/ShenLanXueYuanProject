/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <fstream>
#include <stdexcept>

#include "util/timer.h"
#include "math/functions.h"
#include "math/matrix.h"
#include "math/matrix_tools.h"
#include "core/image_io.h"
#include "core/image_tools.h"
#include "features/sift.h"

FEATURES_NAMESPACE_BEGIN

Sift::Sift (Options const& options)
    : options(options)
{
    if (this->options.min_octave < -1
        || this->options.min_octave > this->options.max_octave)
        throw std::invalid_argument("Invalid octave range");

    if (this->options.contrast_threshold < 0.0f)
        this->options.contrast_threshold = 0.02f
            / static_cast<float>(this->options.num_samples_per_octave);

    if (this->options.debug_output)
        this->options.verbose_output = true;
}

/* ---------------------------------------------------------------- */

void
Sift::process (void)   //start to detect keypoint and descriptors;
{
    util::ClockTimer timer, total_timer;

    /*
     * Creates the scale space representation of the image by
     * sampling the scale space and computing the DoG images.
     * See Section 3, 3.2 and 3.3 in SIFT article.
     */
    if (this->options.verbose_output)
    {
        std::cout << "SIFT: Creating "
            << (this->options.max_octave - this->options.min_octave)
            << " octaves (" << this->options.min_octave << " to "
            << this->options.max_octave << ")..." << std::endl;
    }
    timer.reset();
    this->create_octaves();   //生成图像金子塔,其中主要调用了add-octave函数，即对于每一个octave应如何建立；
    if (this->options.debug_output)
    {
        std::cout << "SIFT: Creating octaves took "
            << timer.get_elapsed() << "ms." << std::endl;
    }

    /*
     * Detects local extrema in the DoG function as described in Section 3.1.
     */
    if (this->options.debug_output)
    {
        std::cout << "SIFT: Detecting local extrema..." << std::endl;
    }
    timer.reset();
    this->extrema_detection(); //对特征点进行粗定位；
    if (this->options.debug_output)
    {
        std::cout << "SIFT: Detected " << this->keypoints.size()
            << " keypoints, took " << timer.get_elapsed() << "ms." << std::endl;
    }

    /*
     * Accurate keypoint localization and filtering.
     * According to Section 4 in SIFT article.
     */
    if (this->options.debug_output)
    {
        std::cout << "SIFT: Localizing and filtering keypoints..." << std::endl;
    }
    timer.reset();
    this->keypoint_localization(); //对图像进行亚像素定位；
    if (this->options.debug_output)
    {
        std::cout << "SIFT: Retained " << this->keypoints.size() << " stable "
            << "keypoints, took " << timer.get_elapsed() << "ms." << std::endl;
    }

    /*
     * Difference of Gaussian images are not needed anymore.
     */
    for (std::size_t i = 0; i < this->octaves.size(); ++i)
        this->octaves[i].dog.clear();

    /*
     * Generate the list of keypoint descriptors.
     * See Section 5 and 6 in the SIFT article.
     * This list can in general be larger than the amount of keypoints,
     * since for each keypoint several descriptors may be created.
     */
    if (this->options.verbose_output)
    {
        std::cout << "SIFT: Generating keypoint descriptors..." << std::endl;
    }
    timer.reset();
    this->descriptor_generation();
    if (this->options.debug_output)
    {
        std::cout << "SIFT: Generated " << this->descriptors.size()
            << " descriptors, took " << timer.get_elapsed() << "ms."
            << std::endl;
    }
    if (this->options.verbose_output)
    {
        std::cout << "SIFT: Generated " << this->descriptors.size()
            << " descriptors from " << this->keypoints.size() << " keypoints,"
            << " took " << total_timer.get_elapsed() << "ms." << std::endl;
    }

    /* Free memory. */
    this->octaves.clear();
}

/* ---------------------------------------------------------------- */

void
Sift::set_image (core::ByteImage::ConstPtr img)
{
    if (img->channels() != 1 && img->channels() != 3)
        throw std::invalid_argument("Gray or color image expected");

    // 将图像转化成灰度图
    this->orig = core::image::byte_to_float_image(img);
    if (img->channels() == 3) {
        this->orig = core::image::desaturate<float>
            (this->orig, core::image::DESATURATE_AVERAGE);
    }
}

/* ---------------------------------------------------------------- */

void
Sift::set_float_image (core::FloatImage::ConstPtr img)
{
    if (img->channels() != 1 && img->channels() != 3)
        throw std::invalid_argument("Gray or color image expected");

    if (img->channels() == 3)
    {
        this->orig = core::image::desaturate<float>
            (img, core::image::DESATURATE_AVERAGE);
    }
    else
    {
        this->orig = img->duplicate();
    }
}

/* ---------------------------------------------------------------- */

void
Sift::create_octaves (void)
{
    this->octaves.clear();

    /*
     * Create octave -1. The original image is assumed to have blur
     * sigma = 0.5. The double size image therefore has sigma = 1.
     */
    if (this->options.min_octave < 0)
    {
        //std::cout << "Creating octave -1..." << std::endl;
        core::FloatImage::Ptr img
            = core::image::rescale_double_size_supersample<float>(this->orig);///对图像进行上采样；
        this->add_octave(img, this->options.inherent_blur_sigma * 2.0f,
            this->options.base_blur_sigma);                  //建立图像的-1层金字塔；
    }

    /*
     * Prepare image for the first positive octave by downsampling.
     * This code is executed only if min_octave > 0.
     */
    core::FloatImage::ConstPtr img = this->orig;// 获得最初的输入图像；
    for (int i = 0; i < this->options.min_octave; ++i)
        img = core::image::rescale_half_size_gaussian<float>(img);  //获得-1层图像；

    /*
     * Create new octave from 'img', then subsample octave image where
     * sigma is doubled to get a new base image for the next octave.
     */
    float img_sigma = this->options.inherent_blur_sigma; //获得图像的最初高斯模糊  系数为0.5；
    for (int i = std::max(0, this->options.min_octave);
        i <= this->options.max_octave; ++i)   //如果min_octave是0，则进行下采样了；
    {
        //std::cout << "Creating octave " << i << "..." << std::endl;
        this->add_octave(img, img_sigma, this->options.base_blur_sigma);           //对每一层图像进行高斯模糊；

        core::FloatImage::ConstPtr pre_base = octaves[octaves.size()-1].img[0];   //获得下一个octave的base图像，即上一个octave的base图像进行下采样；
        img = core::image::rescale_half_size_gaussian<float>(pre_base);          //对上一个octave的base图像进行下采样；

        img_sigma = this->options.base_blur_sigma;
    }
}

/* ---------------------------------------------------------------- */

void
Sift::add_octave (core::FloatImage::ConstPtr image,
        float has_sigma, float target_sigma)   //添加图像金字塔的octave；输入为图像,已经进行的高斯模糊系数，目标高斯模糊；
{
    /*
     * First, bring the provided image to the target blur.  //首先对目标图像进行一定程度的模糊；
     * Since L * g(sigma1) * g(sigma2) = L * g(sqrt(sigma1^2 + sigma2^2)),//给予该公式获得每次相乘的系数以获得目标高斯模糊的图像；
     * we need to blur with sigma = sqrt(target_sigma^2 - has_sigma^2).
     */
    float sigma = std::sqrt(MATH_POW2(target_sigma) - MATH_POW2(has_sigma)); //获得每次应相乘的系数；
    //std::cout << "Pre-blurring image to sigma " << target_sigma << " (has "
    //    << has_sigma << ", blur = " << sigma << ")..." << std::endl;
    core::FloatImage::Ptr base = (target_sigma > has_sigma
        ? core::image::blur_gaussian<float>(image, sigma)
        : image->duplicate()); // 判断将图像进行上采样 还是 进行 进一步的高斯模糊；

    /* Create the new octave and add initial image. */
    this->octaves.push_back(Octave());
    Octave& oct = this->octaves.back();   //back() return the quote of the last element in the vector;
    oct.img.push_back(base);

    /* 'k' is the constant factor between the scales in scale space. */
    float const k = std::pow(2.0f, 1.0f / this->options.num_samples_per_octave); //设置K=2的1/s；
    sigma = target_sigma;

    /* Create other (s+2) samples of the octave to get a total of (s+3). */ ///创建其他s+2个图像金字塔；
    for (int i = 1; i < this->options.num_samples_per_octave + 3; ++i)
    {
        /* Calculate the blur sigma the image will get. */
        float sigmak = sigma * k;     // sigmak即为 图像金字塔目标的高斯滤波系数；
        float blur_sigma = std::sqrt(MATH_POW2(sigmak) - MATH_POW2(sigma));///获得要乘上的高斯滤波系数；

        /* Blur the image to create a new scale space sample. */
        //std::cout << "Blurring image to sigma " << sigmak << " (has " << sigma
        //    << ", blur = " << blur_sigma << ")..." << std::endl;
        core::FloatImage::Ptr img = core::image::blur_gaussian<float>
            (base, blur_sigma);   //对图像进行高斯滤波；
        oct.img.push_back(img);

        /* Create the Difference of Gaussian image (DoG). */
        //计算差分拉普拉斯 // todo revised by sway
        core::FloatImage::Ptr dog = core::image::subtract<float>(img, base);
        oct.dog.push_back(dog);  //获得高斯查分图像；并存入dog；

        /* Update previous image and sigma for next round. */
        base = img;
        sigma = sigmak;
    }
}

/* ---------------------------------------------------------------- */

void
Sift::extrema_detection (void)
{
    /* Delete previous keypoints. */
    this->keypoints.clear();

    /* Detect keypoints in each octave... */
    for (std::size_t i = 0; i < this->octaves.size(); ++i)  //遍历金字塔层数
    {
        Octave const& oct(this->octaves[i]);
        /* In each octave, take three subsequent DoG images and detect. */
        for (int s = 0; s < (int)oct.dog.size() - 2; ++s)   //对于每个octave的金字塔层数
        {
            core::FloatImage::ConstPtr samples[3] =
            { oct.dog[s + 0], oct.dog[s + 1], oct.dog[s + 2] };
            this->extrema_detection(samples, static_cast<int>(i)
                + this->options.min_octave, s);  //对于每3层dog图像，进行特性点的比较； extrema_detection重构函数实现了该功能；
        }
    }
}

/* ---------------------------------------------------------------- */

std::size_t
Sift::extrema_detection (core::FloatImage::ConstPtr s[3], int oi, int si)
{
    int const w = s[1]->width();   //每行元素数目；共多少列；
    int const h = s[1]->height();  //每列元素数目；共多少行；

    /* Offsets for the 9-neighborhood w.r.t. center pixel. */
    int noff[9] = { -1 - w, 0 - w, 1 - w, -1, 0, 1, -1 + w, 0 + w, 1 + w }; //对于要判断是否为特征点的点，其比较的9个对象；

    /*
     * Iterate over all pixels in s[1], and check if pixel is maximum
     * (or minumum) in its 27-neighborhood.
     */
    int detected = 0;
    int off = w;
    for (int y = 1; y < h - 1; ++y, off += w)  //行遍历；
        for (int x = 1; x < w - 1; ++x)  //列遍历； 对图像缩小一圈遍历比较；
        {
            int idx = off + x;    //idx是将图像视为一列，排列顺序为依照行顺序排开；

            bool largest = true;
            bool smallest = true;
            float center_value = s[1]->at(idx);  //取得要进行比较的值；
            for (int l = 0; (largest || smallest) && l < 3; ++l)  //三幅图像遍历比较；
                for (int i = 0; (largest || smallest) && i < 9; ++i) //9个邻域进行比较
                {
                    if (l == 1 && i == 4) // Skip center pixel  //不需要和自己进行比较；
                        continue;
                    if (s[l]->at(idx + noff[i]) >= center_value)
                        largest = false;
                    if (s[l]->at(idx + noff[i]) <= center_value)
                        smallest = false;      //不需要关心是极大值，或是极小值而作两次循环比较，而是将二者放入同一个比较的体系中两个极大极小标志中。
                }

            /* Skip non-maximum values. */
            if (!smallest && !largest)
                continue;    //如果不是最大也不是最小，则不需要进行放入kp的vector容器中。

            /* Yummy. Add detected scale space extremum. */
            Keypoint kp;     //加入尺度空间极值点；
            kp.octave = oi;   //那一层金字塔；
            kp.x = static_cast<float>(x);
            kp.y = static_cast<float>(y);
            kp.sample = static_cast<float>(si);  //金字塔中的哪一个级别的高斯滤波；
            this->keypoints.push_back(kp);
            detected += 1;  //探测到的极值点数目；
        }

    return detected;
}

/* ---------------------------------------------------------------- */

void
Sift::keypoint_localization (void)   //对关键点进一步进行定位；
{
    /*
     * Iterate over all keypoints, accurately localize minima and maxima
     * in the DoG function by fitting a quadratic Taylor polynomial
     * around the keypoint.
     */

    int num_singular = 0;
    int num_keypoints = 0; // Write iterator
    for (std::size_t i = 0; i < this->keypoints.size(); ++i)   //遍历未精确定位的特征点；
    {
        /* Copy keypoint. */
        Keypoint kp(this->keypoints[i]);//在extrema_detection函数中，kp为保存的特征点,需要进一步定位；

        /* Get corresponding octave and DoG images. */ //获得对应的octave,以及DOG图像；
        Octave const& oct(this->octaves[kp.octave - this->options.min_octave]); //哪一层的octave；
        int sample = static_cast<int>(kp.sample);  //sample为金字塔某一层的高斯滤波；
        core::FloatImage::ConstPtr dogs[3] = { oct.dog[sample + 0], oct.dog[sample + 1], oct.dog[sample + 2] }; //比较的三层高斯差分图；

        /* Shorthand for image width and height. */
        int const w = dogs[0]->width();  //获得差分图的宽度
        int const h = dogs[0]->height(); //获得差分图的高度；
        /* The integer and floating point location of the keypoints. */
        int ix = static_cast<int>(kp.x);   //获得关键点的x坐标
        int iy = static_cast<int>(kp.y);  //获得关键点的y坐标
        int is = static_cast<int>(kp.sample); //获得关键点所处的高斯滤波层级；
        float delta_x, delta_y, delta_s;   //声明x偏置,y偏置,s偏置.
        /* The first and second order derivatives. */
        float Dx, Dy, Ds;   //获得一阶偏导；
        float Dxx, Dyy, Dss; //获得二阶偏导；
        float Dxy, Dxs, Dys;  //二阶偏导；

        /*
         * Locate the keypoint using second order Taylor approximation. 利用二阶泰勒展开公式对关键定进行定位；
         * The procedure might get iterated around a neighboring pixel if
         * the accurate keypoint is off by >0.6 from the center pixel.如果精确地关键点位置偏离中心像素超过0.5个像素，该过程即在附近的元素进行迭代；
         */
#       define AT(S,OFF) (dogs[S]->at(px + OFF))  //AT（S,OFF）的含义是差分图S在px的偏置位置；
        for (int j = 0; j < 5; ++j)
        {
            std::size_t px = iy * w + ix;  //px为  w为图宽度,iy为关键点坐标；

            /* Compute first and second derivatives. */ //计算一阶、二阶偏导；
            Dx = (AT(1,1) - AT(1,-1)) * 0.5f;   //-1 +1为x处的  或同行内的偏导；
            Dy = (AT(1,w) - AT(1,-w)) * 0.5f;   //-w,+w为y处的  或同列的偏导；
            Ds = (AT(2,0) - AT(0,0))  * 0.5f;  //2,是不同高斯图 之间的值偏导；

            Dxx = AT(1,1) + AT(1,-1) - 2.0f * AT(1,0);   // x方向二阶偏导
            Dyy = AT(1,w) + AT(1,-w) - 2.0f * AT(1,0);   //y方向二阶偏导；
            Dss = AT(2,0) + AT(0,0)  - 2.0f * AT(1,0);  //不同高斯模糊之间的偏导；

            Dxy = (AT(1,1+w) + AT(1,-1-w) - AT(1,-1+w) - AT(1,1-w)) * 0.25f;   // 交叉偏导；
            Dxs = (AT(2,1)   + AT(0,-1)   - AT(2,-1)   - AT(0,1))   * 0.25f;
            Dys = (AT(2,w)   + AT(0,-w)   - AT(2,-w)   - AT(0,w))   * 0.25f;

            /* Setup the Hessian matrix. */
            math::Matrix3f H;
            /****************************task-1-0  构造Hessian矩阵 ******************************/
            /*
             * 参考第32页slide的Hessian矩阵构造方式填充H矩阵，其中dx=dy=d_sigma=1, 其中A矩阵按照行顺序存储，即
             * H=[H[0], H[1], H[2]]
             *   [H[3], H[4], H[5]]
             *   [H[6], H[7], H[8]]
             */
            H[0]=Dxx;H[1]=Dxy;H[2]=Dxs;
            H[3]=Dxy;H[4]=Dyy;H[5]=Dys;
            H[6]=Dxs;H[7]=Dys;H[8]=Dss;

            /**********************************************************************************/
//            H[0] = Dxx; H[1] = Dxy; H[2] = Dxs;
//            H[3] = Dxy; H[4] = Dyy; H[5] = Dys;
//            H[6] = Dxs; H[7] = Dys; H[8] = Dss;


            /* Compute determinant to detect singular matrix. */
            float detH = math::matrix_determinant(H);  //获取H矩阵的行列式； traceH的平方初一DetH 如果小于一定阈值，则视为边界点；
            if (MATH_EPSILON_EQ(detH, 0.0f, 1e-15f))   //该行列式的值为0 或 过小 该矩阵此时不可逆；
            {
                num_singular += 1;  //则此时 为奇异矩阵；
                delta_x = delta_y = delta_s = 0.0f; // FIXME: Handle this case? //则对该情况不进行进一步处理；
                break;
            }
            /* Invert the matrix to get the accurate keypoint. */
            math::Matrix3f H_inv = math::matrix_inverse(H, detH);  //获得该矩阵的逆；
            math::Vec3f b(-Dx, -Dy, -Ds);


            //math::Vec3f delta;
            /****************************task-1-1  求解偏移量deta ******************************/

             /* 参考第30页slide delta_x的求解方式 delta_x = inv(H)*b
             * 请在此处给出delta的表达式
             */
                     /*                  */
                     /*    此处添加代码    */
                     /*                  */

            /**********************************************************************************/
            math::Vec3f delta = H_inv * b;  //此处即求出了delta；


            delta_x = delta[0];           //获得x偏置量；
            delta_y = delta[1];           //获得y偏置量；
            delta_s = delta[2];           //获得高斯模糊偏置量；
            std::vector<float>deltaTHis={delta_x,delta_y,delta_s};

            /* Check if accurate location is far away from pixel center. */ //检查偏移位置是否距离中心像素超过0.5个像素；
            // dx =0 表示|dx|>0.6f
            int dx = (delta_x > 0.6f && ix < w-2) * 1 + (delta_x < -0.6f && ix > 1) * (-1);  //ix<w-2表示划定图像边界；-2的含义在于在图像边缘处无法进行微分计算；
            int dy = (delta_y > 0.6f && iy < h-2) * 1 + (delta_y < -0.6f && iy > 1) * (-1);

            /* If the accurate location is closer to another pixel,
             * repeat localization around the other pixel. */  //如果精确位置距离另外一个像素更近，则对另外的像素进行迭代计算；
            if (dx != 0 || dy != 0)  //如果dx dy不为0，则比为1或-1；
            {
                ix += dx;
                iy += dy;
                continue;  //继续迭代计算；
            }
            /* Accurate location looks good. */
            break;
        }


        /* Calcualte function value D(x) at accurate keypoint x. *///计算准确位置的极值点的值；
        /*****************************task1-2求解极值点处的DoG值val ***************************/
         /*
          * 参考第30页slides的机极值点f(x)的求解公式f(x) = f(x0) + 0.5* delta.dot(D)
          * 其中
          * f(x0)--表示插值点(ix, iy, is) 处的DoG值，可通过dogs[1]->at(ix, iy, 0)获取
          * delta--为上述求得的delta=[delta_x, delta_y, delta_s]
          * D--为一阶导数，表示为(Dx, Dy, Ds)
          * 请给出求解val的代码
          */
        //float val = 0.0;
        /*                  */
        /*    此处添加代码    */
        /*                  */

        float val = dogs[1]->at(ix,iy,0)+(Dx * delta_x + Dy * delta_y + Ds * delta_s);//一阶展开，二阶展开式子不知道如何相乘；
        /************************************************************************************/
        //float val = dogs[1]->at(ix, iy, 0) + 0.5f * (Dx * delta_x + Dy * delta_y + Ds * delta_s);
        /* Calcualte edge response score Tr(H)^2 / Det(H), see Section 4.1. */
         /**************************去除边缘点，参考第33页slide 仔细阅读代码 ****************************/
        float hessian_trace = Dxx + Dyy;                      //获得DOG的trace；
        float hessian_det = Dxx * Dyy - MATH_POW2(Dxy);      //获得DOG特征点的det；
        float hessian_score = MATH_POW2(hessian_trace) / hessian_det;   //获得起harris检测分数；
        float score_thres = MATH_POW2(this->options.edge_ratio_threshold + 1.0f)
            / this->options.edge_ratio_threshold;         //其边缘阈值为（r+1）的平方/r；
        /********************************************************************************/

        /*
         * Set accurate final keypoint location.
         */
        kp.x = (float)ix + delta_x;
        kp.y = (float)iy + delta_y;
        kp.sample = (float)is + delta_s;

        /*
         * Discard keypoints with:
         * 1. low contrast (value of DoG function at keypoint),   //去除DOG函数过低的关键点；
         * 2. negative hessian determinant (curvatures with different sign),  //去除二阶矩阵为负的关键点；
         *    Note that negative score implies negative determinant.
         * 3. large edge response (large hessian score),  //较大的边缘响应值；
         * 4. unstable keypoint accurate locations,    //不稳定的关键点位置；
         * 5. keypoints beyond the scale space boundary.  //超出尺度空间边界的关键点；
         */
        if (std::abs(val) < this->options.contrast_threshold
            || hessian_score < 0.0f || hessian_score > score_thres
            || std::abs(delta_x) > 1.5f || std::abs(delta_y) > 1.5f || std::abs(delta_s) > 1.0f
            || kp.sample < -1.0f
            || kp.sample > (float)this->options.num_samples_per_octave
            || kp.x < 0.0f || kp.x > (float)(w - 1)
            || kp.y < 0.0f || kp.y > (float)(h - 1))
        {
            //std::cout << " REJECTED!" << std::endl;
            continue;
        }

        /* Keypoint is accepted, copy to write iter and advance. */
        this->keypoints[num_keypoints] = kp;
        num_keypoints += 1;
    }

    /* Limit vector size to number of accepted keypoints. */
    this->keypoints.resize(num_keypoints);

    if (this->options.debug_output && num_singular > 0)
    {
        std::cout << "SIFT: Warning: " << num_singular
            << " singular matrices detected!" << std::endl;
    }
}

/* ---------------------------------------------------------------- */
//生成描述子函数；
void
Sift::descriptor_generation (void)
{
    if (this->octaves.empty())
        throw std::runtime_error("Octaves not available!");
    if (this->keypoints.empty())
        return;

    this->descriptors.clear();  //清理描述子；
    this->descriptors.reserve(this->keypoints.size() * 3 / 2);  //分配关键点1.5倍的描述子空间；

    /*
     * Keep a buffer of S+3 gradient and orientation images for the current
     * octave. Once the octave is changed, these images are recomputed.
     * To ensure efficiency, the octave index must always increase, never
     * decrease, which is enforced during the algorithm.
     */
    int octave_index = this->keypoints[0].octave;
    Octave* octave = &this->octaves[octave_index - this->options.min_octave];

    // todo 计算每个octave中所有图像的梯度值和方向，具体得, octave::grad存储图像的梯度响应值，octave::ori存储梯度方向
    this->generate_grad_ori_images(octave);

    /* Walk over all keypoints and compute descriptors. */
    for (std::size_t i = 0; i < this->keypoints.size(); ++i)
    {
        Keypoint const& kp(this->keypoints[i]);

        /* Generate new gradient and orientation images if octave changed. */
        if (kp.octave > octave_index)
        {
            /* Clear old octave gradient and orientation images. */
            if (octave)
            {
                octave->grad.clear();
                octave->ori.clear();
            }
            /* Setup new octave gradient and orientation images. */
            octave_index = kp.octave;
            octave = &this->octaves[octave_index - this->options.min_octave];
            this->generate_grad_ori_images(octave);
        }
        else if (kp.octave < octave_index)
        {
            throw std::runtime_error("Decreasing octave index!");
        }

        /* Orientation assignment. This returns multiple orientations. */
        /* todo 统计直方图找到特征点主方向,找到几个主方向*/
        std::vector<float> orientations;
        orientations.reserve(8);
        this->orientation_assignment(kp, octave, orientations);

        /* todo 生成特征向量,同一个特征点可能有多个描述子，为了提升匹配的稳定性*/
        /* Feature vector extraction. */
        for (std::size_t j = 0; j < orientations.size(); ++j)
        {
            Descriptor desc;
            float const scale_factor = std::pow(2.0f, kp.octave);
            desc.x = scale_factor * (kp.x + 0.5f) - 0.5f;
            desc.y = scale_factor * (kp.y + 0.5f) - 0.5f;
            desc.scale = this->keypoint_absolute_scale(kp);
            desc.orientation = orientations[j];
            if (this->descriptor_assignment(kp, desc, octave))
                this->descriptors.push_back(desc);
        }
    }
}

/* ---------------------------------------------------------------- */

void
Sift::generate_grad_ori_images (Octave* octave)   //获得图像的主方向；
{
    octave->grad.clear();
    octave->grad.reserve(octave->img.size());   //共有s+3幅梯度图；
    octave->ori.clear();
    octave->ori.reserve(octave->img.size());    //共有s+3幅主方向图；

    int const width = octave->img[0]->width();   //获得该层金字塔图像的宽度和高度
    int const height = octave->img[0]->height();

    //std::cout << "Generating gradient and orientation images..." << std::endl;//生成图像的梯度图和主方向图；
    for (std::size_t i = 0; i < octave->img.size(); ++i)  // 遍历该层金字塔的图像；
    {
        core::FloatImage::ConstPtr img = octave->img[i];   //获得当前图像的首地址；
        core::FloatImage::Ptr grad = core::FloatImage::create(width, height, 1); //初始化梯度图的大小；1为通道数； 1个说明是灰度图呀；
        core::FloatImage::Ptr ori = core::FloatImage::create(width, height, 1);  //初始化方向图的大小；

        int image_iter = width + 1;   //设置图像迭代；
        for (int y = 1; y < height - 1; ++y, image_iter += 2)//遍历行；行换+2;边界的影响；
            for (int x = 1; x < width - 1; ++x, ++image_iter) //遍历列；列换+1；
            {
                float m1x = img->at(image_iter - 1);
                float p1x = img->at(image_iter + 1);
                float m1y = img->at(image_iter - width);
                float p1y = img->at(image_iter + width);
                float dx = 0.5f * (p1x - m1x);
                float dy = 0.5f * (p1y - m1y);

                float atan2f = std::atan2(dy, dx);
                grad->at(image_iter) = std::sqrt(dx * dx + dy * dy);  //获得梯度响应值；
                ori->at(image_iter) = atan2f < 0.0f
                    ? atan2f + MATH_PI * 2.0f : atan2f;  //获得梯度的主方向；如果大于0，为其自身，否则+pi 设置其的主方向；
            }
        octave->grad.push_back(grad);
        octave->ori.push_back(ori);
    }
}

/* ---------------------------------------------------------------- */

void
Sift::orientation_assignment (Keypoint const& kp,
    Octave const* octave, std::vector<float>& orientations)  //方向分配；
{
    int const nbins = 36;  //设置36个度数直方图,每个直方图为10度；
    float const nbinsf = static_cast<float>(nbins); //改变数据类型；

    /* Prepare 36-bin histogram. */   //准备36个二进制直方图
    float hist[nbins];
    std::fill(hist, hist + nbins, 0.0f);//初始化为0；

    /* Integral x and y coordinates and closest scale sample. */
    int const ix = static_cast<int>(kp.x + 0.5f);   //x大于0.5则值为1，否则为0；
    int const iy = static_cast<int>(kp.y + 0.5f);   //同上；
    int const is = static_cast<int>(math::round(kp.sample)); //ocaave中的层数，由于进行了精确定位，因此也float化了；
    float const sigma = this->keypoint_relative_scale(kp); //计算kp的相对尺度，即在一个octave中的尺度；

    /* Images with its dimension for the keypoint. */
    core::FloatImage::ConstPtr grad(octave->grad[is + 1]); //设置grad的首地址为,+1原因在于dog图从+1开始； 差分导致的边界的存在
    core::FloatImage::ConstPtr ori(octave->ori[is + 1]);   //设置主方向的首地址，同理；
    int const width = grad->width();
    int const height = grad->height();

    /*
     * Compute window size 'win', the full window has  2 * win + 1  pixel.
     * The factor 3 makes the window large enough such that the gaussian
     * has very little weight beyond the window. The value 1.5 is from
     * the SIFT paper. If the window goes beyond the image boundaries,
     * the keypoint is discarded.
     */
    float const sigma_factor = 1.5f;
    int win = static_cast<int>(sigma * sigma_factor * 3.0f);  //win指的是对关键点进行描述的描述，需要在一个窗口里面进行描述；//尺度不同，设置的window大小也不同；
    if (ix < win || ix + win >= width || iy < win || iy + win >= height)
        return;   //保证边界不被计算；

    /* Center of keypoint index. */
    int center = iy * width + ix;   //计算出关键点的中心；
    float const dxf = kp.x - static_cast<float>(ix);  //离散值与连续值的偏置量；
    float const dyf = kp.y - static_cast<float>(iy);  //y方向；
    float const maxdist = static_cast<float>(win*win) + 0.5f;//距离中心位置最远的位置距离；

    /* Populate histogram over window, intersected with (1,1), (w-2,h-2). */
    for (int dy = -win; dy <= win; ++dy)      //变量窗口内的元素；
    {
        int const yoff = dy * width;  //计算单列化偏置；
        for (int dx = -win; dx <= win; ++dx)  //遍历行内元素；
        {
            /* Limit to circular window (centered at accurate keypoint). */
            float const dist = MATH_POW2(dx-dxf) + MATH_POW2(dy-dyf); //限制在圆形窗口内；
            if (dist > maxdist)  //如果dist超过了 maxdist 则不予计算；
                continue;

            float gm = grad->at(center + yoff + dx); // gradient magnitude  //当前位置的梯度值；
            float go = ori->at(center + yoff + dx); // gradient orientation  //当前位置梯度方向；
            float weight = math::gaussian_xx(dist, sigma * sigma_factor);  //分配高斯权重；
            int bin = static_cast<int>(nbinsf * go / (2.0f * MATH_PI));    //计算处于哪一个量化等级；
            bin = math::clamp(bin, 0, nbins - 1);  //clamp函数含义：如果bin大于max则为max，小于min则为min，如果处于二者中间则为其本身；
            hist[bin] += gm * weight;  //每个主方向bin的值为其距离的高斯分布权重×梯度值后相加；而不是把每一个都看成是1；
        }
    }

    /* Smooth histogram. *///对直方图进行smooth；
    for (int i = 0; i < 6; ++i)  //i的意义是什么？  6 次平滑的循環？
    {
        float first = hist[0]; //直方图第一个值设置为first；
        float prev = hist[nbins - 1];  //直方图最后一个地址的值设置为prev；
        for (int j = 0; j < nbins - 1; ++j)
        {
            float current = hist[j];  //current为当前遍历到的直方图的值；
            hist[j] = (prev + current + hist[j + 1]) / 3.0f; //设置当前直方图的值为相邻三个值的均值；进行平滑smooth；
            prev = current;
        }
        hist[nbins - 1] = (prev + hist[nbins - 1] + first) / 3.0f; //设置hist最后一个元素的计算方法；
    }

    /* Find maximum element. */
    float maxh = *std::max_element(hist, hist + nbins);  //寻找到直方图中元素值最大的bin；

    /* Find peaks within 80% of max element. */  //选择在最大值的80%以上的元素；
    for (int i = 0; i < nbins; ++i)
    {
        float h0 = hist[(i + nbins - 1) % nbins];  //取余数；是i-1呀 有什么卵用？
        float h1 = hist[i];
        float h2 = hist[(i + 1) % nbins]; //取余数？ 还是i+1呀 有什么作用？

        /* These peaks must be a local maximum! */
        if (h1 <= 0.8f * maxh || h1 <= h0 || h1 <= h2)  //h1 must be larger than 0.8maxh and h1 and h2;
            continue;

        /*
         * Quadratic interpolation to find accurate maximum.
         * f(x) = ax^2 + bx + c, f(-1) = h0, f(0) = h1, f(1) = h2
         * --> a = 1/2 (h0 - 2h1 + h2), b = 1/2 (h2 - h0), c = h1.
         * x = f'(x) = 2ax + b = 0 --> x = -1/2 * (h2 - h0) / (h0 - 2h1 + h2)
         */
        float x = -0.5f * (h2 - h0) / (h0 - 2.0f * h1 + h2); //find accurate maximum through quaratic interpolation.
        float o =  2.0f * MATH_PI * (x + (float)i + 0.5f) / nbinsf; //angle to float.
        orientations.push_back(o);
    }
}

/* ---------------------------------------------------------------- */

bool
Sift::descriptor_assignment (Keypoint const& kp, Descriptor& desc,
    Octave const* octave)
{
    /*
     * The final feature vector has size PXB * PXB * OHB.
     * The following constants should not be changed yet, as the
     * (PXB^2 * OHB = 128) element feature vector is still hard-coded.
     */
    //int const PIX = 16; // Descriptor region with 16x16 pixel  分配種子點；16*16个像素；一个种子点的描述子有其4*4的矩阵描述；
    int const PXB = 4; // Pixel bins with 4x4 bins    //每个种子点描述4*4个像素的梯度分布；
    int const OHB = 8; // Orientation histogram with 8 bins  //每一个种子点描述由8个方向描述，每个方向关联45度；

    /* Integral x and y coordinates and closest scale sample. */
    int const ix = static_cast<int>(kp.x + 0.5f);  //对x,y进行四舍五入；
    int const iy = static_cast<int>(kp.y + 0.5f);
    int const is = static_cast<int>(math::round(kp.sample));
    float const dxf = kp.x - static_cast<float>(ix); //四舍五入的量；
    float const dyf = kp.y - static_cast<float>(iy);
    float const sigma = this->keypoint_relative_scale(kp);  //获得关键点的相对尺度；

    /* Images with its dimension for the keypoint. */
    core::FloatImage::ConstPtr grad(octave->grad[is + 1]);  //get gradient of the octave;
    core::FloatImage::ConstPtr ori(octave->ori[is + 1]);    //get the orientation of the octave;
    int const width = grad->width();//获得梯度图的宽度；
    int const height = grad->height(); //获得梯度图的高度；

    /* Clear feature vector. */
    desc.data.fill(0.0f);   //清理特征矢量；

    /* Rotation constants given by descriptor orientation. */
    float const sino = std::sin(desc.orientation);  //rotation constant given by the descriptor orientation;
    float const coso = std::cos(desc.orientation);   //根据梯度的主方向,调整关键点的描述子；

    /*
     * Compute window size.
     * Each spacial bin has an extension of 3 * sigma (sigma is the scale
     * of the keypoint). For interpolation we need another half bin at
     * both ends in each dimension. And since the window can be arbitrarily
     * rotated, we need to multiply with sqrt(2). The window size is:
     * 2W = sqrt(2) * 3 * sigma * (PXB + 1).
     */
    float const binsize = 3.0f * sigma;  //octave的相对系数；
    int win = MATH_SQRT2 * binsize * (float)(PXB + 1) * 0.5f;
    if (ix < win || ix + win >= width || iy < win || iy + win >= height)
        return false;

    /*
     * Iterate over the window, intersected with the image region
     * from (1,1) to (w-2, h-2) since gradients/orientations are
     * not defined at the boundary pixels. Add all samples to the
     * corresponding bin.
     */
    int const center = iy * width + ix; // Center pixel at KP location
    for (int dy = -win; dy <= win; ++dy)
    {
        int const yoff = dy * width;
        for (int dx = -win; dx <= win; ++dx)
        {
            /* Get pixel gradient magnitude and orientation. */
            float const mod = grad->at(center + yoff + dx);
            float const angle = ori->at(center + yoff + dx);
            float theta = angle - desc.orientation;
            if (theta < 0.0f)
                theta += 2.0f * MATH_PI;

            /* Compute fractional coordinates w.r.t. the window. */
            float const winx = (float)dx - dxf;  //计算winx
            float const winy = (float)dy - dyf;

            /*
             * Compute normalized coordinates w.r.t. bins. The window
             * coordinates are rotated around the keypoint. The bins are
             * chosen such that 0 is the coordinate of the first bins center
             * in each dimension. In other words, (0,0,0) is the coordinate
             * of the first bin center in the three dimensional histogram.
             */
            float binoff = (float)(PXB - 1) / 2.0f;
            float binx = (coso * winx + sino * winy) / binsize + binoff;
            float biny = (-sino * winx + coso * winy) / binsize + binoff;
            float bint = theta * (float)OHB / (2.0f * MATH_PI) - 0.5f;

            /* Compute circular window weight for the sample. */
            float gaussian_sigma = 0.5f * (float)PXB;
            float gaussian_weight = math::gaussian_xx
                (MATH_POW2(binx - binoff) + MATH_POW2(biny - binoff),
                gaussian_sigma);

            /* Total contribution of the sample in the histogram is now: */
            float contrib = mod * gaussian_weight;

            /*
             * Distribute values into bins (using trilinear interpolation).
             * Each sample is inserted into 8 bins. Some of these bins may
             * not exist, because the sample is outside the keypoint window.
             */
            int bxi[2] = { (int)std::floor(binx), (int)std::floor(binx) + 1 };
            int byi[2] = { (int)std::floor(biny), (int)std::floor(biny) + 1 };
            int bti[2] = { (int)std::floor(bint), (int)std::floor(bint) + 1 };

            float weights[3][2] = {
                { (float)bxi[1] - binx, 1.0f - ((float)bxi[1] - binx) },
                { (float)byi[1] - biny, 1.0f - ((float)byi[1] - biny) },
                { (float)bti[1] - bint, 1.0f - ((float)bti[1] - bint) }
            };

            // Wrap around orientation histogram
            if (bti[0] < 0)
                bti[0] += OHB;
            if (bti[1] >= OHB)
                bti[1] -= OHB;

            /* Iterate the 8 bins and add weighted contrib to each. */
            int const xstride = OHB;
            int const ystride = OHB * PXB;
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x)
                    for (int t = 0; t < 2; ++t)
                    {
                        if (bxi[x] < 0 || bxi[x] >= PXB
                            || byi[y] < 0 || byi[y] >= PXB)
                            continue;

                        int idx = bti[t] + bxi[x] * xstride + byi[y] * ystride;
                        desc.data[idx] += contrib * weights[0][x]
                            * weights[1][y] * weights[2][t];
                    }
        }
    }

    /* Normalize the feature vector. */
    desc.data.normalize();

    /* Truncate descriptor values to 0.2. */
    for (int i = 0; i < PXB * PXB * OHB; ++i)
        desc.data[i] = std::min(desc.data[i], 0.2f);

    /* Normalize once again. */
    desc.data.normalize();

    return true;
}

/* ---------------------------------------------------------------- */

/*
 * The scale of a keypoint is: scale = sigma0 * 2^(octave + (s+1)/S).
 * sigma0 is the initial blur (1.6), octave the octave index of the
 * keypoint (-1, 0, 1, ...) and scale space sample s in [-1,S+1] where
 * S is the amount of samples per octave. Since the initial blur 1.6
 * corresponds to scale space sample -1, we add 1 to the scale index.
 */

float
Sift::keypoint_relative_scale (Keypoint const& kp)    //相对尺度，指的是在一个octave中的高斯模糊成都
{
    return this->options.base_blur_sigma * std::pow(2.0f,
        (kp.sample + 1.0f) / this->options.num_samples_per_octave);
}

float
Sift::keypoint_absolute_scale (Keypoint const& kp)  //绝对尺度包括了第几层层octave的影响，1为原图像进行上采样的尺度；
{
    return this->options.base_blur_sigma * std::pow(2.0f,
        kp.octave + (kp.sample + 1.0f) / this->options.num_samples_per_octave);
}

/* ---------------------------------------------------------------- */

void
Sift::load_lowe_descriptors (std::string const& filename, Descriptors* result)
{
    std::ifstream in(filename.c_str());
    if (!in.good())
        throw std::runtime_error("Cannot open descriptor file");

    int num_descriptors;
    int num_dimensions;
    in >> num_descriptors >> num_dimensions;
    if (num_descriptors > 100000 || num_dimensions != 128)
    {
        in.close();
        throw std::runtime_error("Invalid number of descriptors/dimensions");
    }
    result->clear();
    result->reserve(num_descriptors);
    for (int i = 0; i < num_descriptors; ++i)
    {
        Sift::Descriptor descriptor;
        in >> descriptor.y >> descriptor.x
            >> descriptor.scale >> descriptor.orientation;
        for (int j = 0; j < 128; ++j)
            in >> descriptor.data[j];
        descriptor.data.normalize();
        result->push_back(descriptor);
    }

    if (!in.good())
    {
        result->clear();
        in.close();
        throw std::runtime_error("Error while reading descriptors");
    }

    in.close();
}

FEATURES_NAMESPACE_END
