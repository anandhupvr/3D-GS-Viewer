#include "framebuffer.h"

#include <vector>

// change it for gpu RGBA, cpu it was using RGB, this will break cpu
FrameBuffer::FrameBuffer(int width, int height)
    : width_(width), height_(height), pixels_(width * height * 4, 0) {}

FrameBuffer::~FrameBuffer() {}

void FrameBuffer::clear(int r, int g, int b) {
    for (size_t i = 0; i < pixels_.size(); i += 3) {
        pixels_[i] = r;
        pixels_[i + 1] = g;
        pixels_[i + 2] = b;
    }
}

void FrameBuffer::set_pixel(int x, int y, int r, int g, int b) {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        return;  // Out of bounds
    }
    size_t index = (y * width_ + x) * 3;
    pixels_[index] = r;
    pixels_[index + 1] = g;
    pixels_[index + 2] = b;
}

std::vector<int> FrameBuffer::get_pixel(int x, int y) {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        return {0, 0, 0};  // Out of bounds, return black
    }

    size_t index = (y * width_ + x) * 3;
    return {pixels_[index], pixels_[index + 1], pixels_[index + 2]};
}

const uint8_t* FrameBuffer::data() const {
    return pixels_.data();
}