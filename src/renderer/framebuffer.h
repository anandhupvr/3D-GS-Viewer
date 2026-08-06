#pragma once
#include <cstdint>
#include <vector>
class FrameBuffer {
   public:
    FrameBuffer(int width, int height);
    ~FrameBuffer();

    void clear(int r, int g, int b);
    void set_pixel(int x, int y, int r, int g, int b);
    std::vector<int> get_pixel(int x, int y);
    const uint8_t* data() const;

    uint8_t* mutable_data() {
        return pixels_.data();
    }

    const int width() const {
        return width_;
    }
    const int height() const {
        return height_;
    }

   private:
    int width_, height_;
    std::vector<uint8_t> pixels_;
};