#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "hdr_common.hpp"
from distutils.command.build import build
from PIL import Image
import glob
import os
import numpy as np
import cv2

class AlignMTBImpl:
    def __init__(self, path, out_path, exclusion_range=1):
        self.out_path = out_path
        self.exclusion_range = exclusion_range

        image_fns = sorted(glob.glob(os.path.join(path, '*.JPG')))
        if len(image_fns) == 0:
            image_fns = sorted(glob.glob(os.path.join(path, '*.png')))
        if len(image_fns) == 0:
            image_fns = sorted(glob.glob(os.path.join(path, '*.jpg')))

        print(image_fns)
        self.P = len(image_fns) # number of images

        self.rgb_images = np.asarray([np.asarray(Image.open(fn).convert("RGB")) for fn in image_fns])
        self.raw_images = np.asarray([np.asarray(Image.open(fn).convert("L")) for fn in image_fns])

        # print(self.raw_images[2])
        # img = Image.fromarray(self.raw_images[2])
        # img.save('greyscale.png')

    def process(self):
        print("Aligning images")
        img = Image.fromarray(self.rgb_images[0])
        img.save(os.path.join(self.out_path, "aligned_{:02d}".format(0) + '.png'))
        for p in range(1, self.P):
            shift = self.calculateShift(self.raw_images[0], self.raw_images[p])
            print("Image ", str(p), ":", shift)

            self.rgb_images[p] = self.shiftMat(self.rgb_images[p] , shift)
            img = Image.fromarray(self.rgb_images[p])
            img.save(os.path.join(self.out_path, "aligned_{:02d}".format(p) + '.png'))
        del self.rgb_images
        del self.raw_images
        # TODO
        #     CV_INSTRUMENT_REGION();
        #     std::vector<Mat> src;
        #     _src.getMatVector(src);

        #     checkImageDimensions(src);
        #     dst.resize(src.size());

        #     size_t pivot = src.size() / 2;
        #     dst[pivot] = src[pivot];
        #     Mat gray_base;
        #     cvtColor(src[pivot], gray_base, COLOR_RGB2GRAY);
        #     std::vector<Point> shifts;

        #     for(size_t i = 0; i < src.size(); i++) {
        #         if(i == pivot) {
        #             shifts.push_back(Point(0, 0));
        #             continue;
        #         }
        #         Mat gray;
        #         cvtColor(src[i], gray, COLOR_RGB2GRAY);
        #         Point shift = calculateShift(gray_base, gray);
        #         shifts.push_back(shift);
        #         shiftMat(src[i], dst[i], shift);
        #     }
        #     if(cut) {
        #         Point max(0, 0), min(0, 0);
        #         for(size_t i = 0; i < shifts.size(); i++) {
        #             if(shifts[i].x > max.x) {
        #                 max.x = shifts[i].x;
        #             }
        #             if(shifts[i].y > max.y) {
        #                 max.y = shifts[i].y;
        #             }
        #             if(shifts[i].x < min.x) {
        #                 min.x = shifts[i].x;
        #             }
        #             if(shifts[i].y < min.y) {
        #                 min.y = shifts[i].y;
        #             }
        #         }
        #         Point size = dst[0].size();
        #         for(size_t i = 0; i < dst.size(); i++) {
        #             dst[i] = dst[i](Rect(max, min + size));
        #         }
        #     }
        # }


    def calculateShift(self, img0, img1):
        maxlevel = int(np.log2(min(img0.shape[0], img0.shape[1]))) - 2
        pyr0 = self.buildPyr(img0, maxlevel)
        pyr1 = self.buildPyr(img1, maxlevel)

        shift = np.array([0,0])
        for level in range(maxlevel, 0, -1):
            shift = shift * 2
            tb0, eb0 = self.computeBitmaps(pyr0[level])
            tb1, eb1 = self.computeBitmaps(pyr1[level])
            min_err = pyr0[level].shape[0] * pyr0[level].shape[1]
            for v in (-1, 0, 1):
                for h in (-1, 0, 1):
                    test_shift = shift + np.array([v, h])
                    shifted_tb1 = self.shiftMat(tb1, test_shift)
                    shifted_eb1 = self.shiftMat(eb1, test_shift)

                    diff = np.logical_xor(tb0, shifted_tb1)
                    # diff = np.logical_and(diff, eb0)
                    # diff = np.logical_and(diff, shifted_eb1)

                    # img = Image.fromarray(diff)
                    # img.save('diff_'+ str(v)+ "_"+ str(h)+ '.png')

                    err = np.sum(diff)
                    # print(v, " ", h, " " , err)
                    if err < min_err:
                        new_shift = test_shift
                        min_err = err
            shift = new_shift
        return shift

        # TODO
        # CV_INSTRUMENT_REGION();

        # Mat img0 = _img0.getMat();
        # Mat img1 = _img1.getMat();
        # CV_Assert(img0.channels() == 1 && img0.type() == img1.type());
        # CV_Assert(img0.size() == img1.size());

        # int maxlevel = static_cast<int>(log((double)max(img0.rows, img0.cols)) / log(2.0)) - 1;
        # maxlevel = min(maxlevel, max_bits - 1);

        # std::vector<Mat> pyr0;
        # std::vector<Mat> pyr1;
        # buildPyr(img0, pyr0, maxlevel);
        # buildPyr(img1, pyr1, maxlevel);

        # Point shift(0, 0);
        # for(int level = maxlevel; level >= 0; level--) {

        #     shift *= 2;
        #     Mat tb1, tb2, eb1, eb2;
        #     computeBitmaps(pyr0[level], tb1, eb1);
        #     computeBitmaps(pyr1[level], tb2, eb2);

        #     int min_err = (int)pyr0[level].total();
        #     Point new_shift(shift);
        #     for(int i = -1; i <= 1; i++) {
        #         for(int j = -1; j <= 1; j++) {
        #             Point test_shift = shift + Point(i, j);
        #             Mat shifted_tb2, shifted_eb2, diff;
        #             shiftMat(tb2, shifted_tb2, test_shift);
        #             shiftMat(eb2, shifted_eb2, test_shift);
        #             bitwise_xor(tb1, shifted_tb2, diff);
        #             bitwise_and(diff, eb1, diff);
        #             bitwise_and(diff, shifted_eb2, diff);
        #             int err = countNonZero(diff);
        #             if(err < min_err) {
        #                 new_shift = test_shift;
        #                 min_err = err;
        #             }
        #         }
        #     }
        #     shift = new_shift;
        # }
        # return shift;


    def shiftMat(self, src, shift):
        if len(src.shape) <= 2:
            v_pad = np.zeros((abs(shift[0]), src.shape[1]), dtype=np.bool8)
            h_pad = np.zeros((src.shape[0], abs(shift[1])), dtype=np.bool8)
            if shift[0] > 0:
                src = np.concatenate((v_pad, src[:-shift[0], :]), axis=0)
            elif shift[0] < 0:
                src = np.concatenate((src[-shift[0]:, :], v_pad), axis=0)
            else:
                pass

            if shift[1] > 0:
                src = np.concatenate((h_pad, src[:, :-shift[1]]), axis=1)
            elif shift[1] < 0:
                src = np.concatenate((src[:, -shift[1]:], h_pad), axis=1)
            else:
                pass
        else:
            v_pad = np.zeros((abs(shift[0]), src.shape[1], 3), dtype=np.bool8)
            h_pad = np.zeros((src.shape[0], abs(shift[1]), 3), dtype=np.bool8)
        
            if shift[0] > 0:
                src = np.concatenate((v_pad, src[:-shift[0], :,:]), axis=0)
            elif shift[0] < 0:
                src = np.concatenate((src[-shift[0]:, :,:], v_pad), axis=0)
            else:
                pass

            if shift[1] > 0:
                src = np.concatenate((h_pad, src[:, :-shift[1],:]), axis=1)
            elif shift[1] < 0:
                src = np.concatenate((src[:, -shift[1]:,:], h_pad), axis=1)
            else:
                pass

        return src
        
        # TODO
        # CV_INSTRUMENT_REGION();

        # Mat src = _src.getMat();
        # _dst.create(src.size(), src.type());
        # Mat dst = _dst.getMat();

        # Mat res = Mat::zeros(src.size(), src.type());
        # int width = src.cols - abs(shift.x);
        # int height = src.rows - abs(shift.y);
        # Rect dst_rect(max(shift.x, 0), max(shift.y, 0), width, height);
        # Rect src_rect(max(-shift.x, 0), max(-shift.y, 0), width, height);
        # src(src_rect).copyTo(res(dst_rect));
        # res.copyTo(dst);
    
    def computeBitmaps(self, img):
        self.bitmap = np.zeros_like(img, dtype=np.bool8)
        self.ex_bitmap = np.zeros_like(img, dtype=np.bool8)

        median = np.median(img, axis=None)
        bitmap = np.where(img > median, True, False)
        ex_bitmap = np.where(np.abs(img - median) > self.exclusion_range, True, False)

        return bitmap, ex_bitmap
        # TODO

        # CV_INSTRUMENT_REGION();

        # Mat img = _img.getMat();
        # _tb.create(img.size(), CV_8U);
        # _eb.create(img.size(), CV_8U);
        # Mat tb = _tb.getMat(), eb = _eb.getMat();
        # int median = getMedian(img);
        # compare(img, median, tb, CMP_GT);
        # compare(abs(img - median), exclude_range, eb, CMP_GT);


    def downsample(self, src):
        return cv2.resize(src, (src.shape[1]//2, src.shape[0]//2), interpolation=cv2.INTER_AREA)
        # TODO
        # dst = Mat(src.rows / 2, src.cols / 2, CV_8UC1);

        # int offset = src.cols * 2;
        # uchar *src_ptr = src.ptr();
        # uchar *dst_ptr = dst.ptr();
        # for(int y = 0; y < dst.rows; y ++) {
        #     uchar *ptr = src_ptr;
        #     for(int x = 0; x < dst.cols; x++) {
        #         dst_ptr[0] = ptr[0];
        #         dst_ptr++;
        #         ptr += 2;
        #     }
        #     src_ptr += offset;
        # }
    



    def buildPyr(self, img, maxlevel):
        pyr = [img,]
        for l in range(maxlevel):
            pyr.append(self.downsample(pyr[l]))
        return pyr
        # TODO
        # pyr.resize(maxlevel + 1);
        # pyr[0] = img.clone();
        # for(int level = 0; level < maxlevel; level++) {
        #     downsample(pyr[level], pyr[level + 1]);
        # }

    def getMedian(self, img):
        pass
        # TODO
        # int channels = 0;
        # Mat hist;
        # int hist_size = LDR_SIZE;
        # float range[] = {0, LDR_SIZE} ;
        # const float* ranges[] = {range};
        # calcHist(&img, 1, &channels, Mat(), hist, 1, &hist_size, ranges);
        # float *ptr = hist.ptr<float>();
        # int median = 0, sum = 0;
        # int thresh = (int)img.total() / 2;
        # while(sum < thresh && median < LDR_SIZE) {
        #     sum += static_cast<int>(ptr[median]);
        #     median++;
        # }
        # return median;

# Ptr<AlignMTB> createAlignMTB(int max_bits, int exclude_range, bool cut)
# {
#     return makePtr<AlignMTBImpl>(max_bits, exclude_range, cut);
# }

# }


if __name__ == '__main__':
    aligner = AlignMTBImpl('hdr_photos')
    aligner.process()