// ImageIO-backed image decode fallback: stb_image covers PNG/JPEG but not
// WebP (which is what browser clients commonly produce, e.g. dsh re-encodes
// uploads to WebP). This decoder handles every format macOS ImageIO knows
// (WebP, HEIC, GIF, TIFF, ...) and returns interleaved RGB888 rows.
//
// Used by qwen38_preprocess.preprocessImageBytes when stbi_load_from_memory
// returns null.
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#include <stdlib.h>

// Returns malloc'd RGB888 buffer (w*h*3 bytes) or NULL on failure.
// Caller frees with free().
uint8_t* dmlx_imageio_decode_rgb(const uint8_t* bytes, size_t len, int* out_w, int* out_h) {
    @autoreleasepool {
        NSData* data = [NSData dataWithBytesNoCopy:(void*)bytes length:len freeWhenDone:NO];
        CGImageSourceRef src = CGImageSourceCreateWithData((__bridge CFDataRef)data, NULL);
        if (!src) return NULL;
        CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
        CFRelease(src);
        if (!img) return NULL;

        const size_t w = CGImageGetWidth(img);
        const size_t h = CGImageGetHeight(img);
        if (w == 0 || h == 0 || w > 16384 || h > 16384) {
            CGImageRelease(img);
            return NULL;
        }
        // Draw into a 32bpp RGBX context (the widely-supported fast path),
        // then compact to RGB888.
        uint8_t* rgba = (uint8_t*)malloc(w * h * 4);
        if (!rgba) {
            CGImageRelease(img);
            return NULL;
        }
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(rgba, w, h, 8, w * 4, cs,
                                                 kCGImageAlphaNoneSkipLast);
        CGColorSpaceRelease(cs);
        if (!ctx) {
            free(rgba);
            CGImageRelease(img);
            return NULL;
        }
        CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
        CGContextRelease(ctx);
        CGImageRelease(img);
        uint8_t* rgb = (uint8_t*)malloc(w * h * 3);
        if (!rgb) {
            free(rgba);
            return NULL;
        }
        for (size_t i = 0; i < w * h; i++) {
            rgb[i * 3 + 0] = rgba[i * 4 + 0];
            rgb[i * 3 + 1] = rgba[i * 4 + 1];
            rgb[i * 3 + 2] = rgba[i * 4 + 2];
        }
        free(rgba);
        *out_w = (int)w;
        *out_h = (int)h;
        return rgb;
    }
}
