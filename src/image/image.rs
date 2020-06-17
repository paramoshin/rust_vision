extern crate image;

use std::cmp;
use std::u8;
use std::path::Path;

#[derive(Clone)]
pub struct Image {
    pub w: u32,
    pub h: u32,
    pub c: u32,
    pub data: Vec<u8>,
}

impl Image {
    pub fn new(w: u32, h: u32, c: u32) -> Self {
        Image {
            w: w,
            h: h,
            c: c,
            data: vec![0; (h * w * c) as usize],
        }
    }

    fn get_idx(&self, x: u32, y: u32, c: u32) -> usize {
        (x + y * self.w + (c * self.w * self.h)) as usize
    }

    pub fn load_image(path: &Path) -> Result<Self, image::ImageError> {

        let im = image::open(path)?.to_rgb();

        let (w, h) = im.dimensions();
        let c = 3;
        let mut img = Image::new(w, h, c);

        for (x, y, p) in im.enumerate_pixels() {
            for i in 0..c {
                let idx = img.get_idx(x, y, i);
                img.data[idx] = p[i as usize];
            }
        }

        Ok(img)
    }

    pub fn get_pixel(&self, x: u32, y: u32, c: u32) -> u8 {
        let idx = self.get_idx(x, y, c);
        self.data[idx]
    }

    pub fn write(&self, path: &Path) -> Result<(), image::ImageError> {
        if self.c == 1{
            let mut img = image::GrayImage::new(self.w, self.h);
            for y in 0..self.h {
                for x in 0..self.w {
                    let pixels: [u8; 1] = [self.get_pixel(x, y, 0)]; 
                    img.put_pixel(x, y, image::Luma(pixels));
                }
            }
        img.save(path)?;
        } else {
            let mut img = image::RgbImage::new(self.w, self.h);
            let mut pixels: [u8; 3] = [0; 3];
            for y in 0..self.h {
                for x in 0..self.w {
                    for i in 0..self.c {
                        pixels[i as usize] = self.get_pixel(x, y, i);
                    }
                img.put_pixel(x, y, image::Rgb(pixels));
                }
            }
        img.save(path)?;
        }
        Ok(())
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, c: u32, v: u8) {
        let idx = self.get_idx(x, y, c);
        self.data[idx] = v;
    }

    pub fn rgb_to_grayscale(&mut self) {
        let mut grayscale_data: Vec<u8> = vec![0; (self.h * self.w) as usize];
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = self.get_idx(x, y, 0);
                grayscale_data[idx] = (
                    0.299 * self.get_pixel(x, y, 0) as f64 + 0.587 * self.get_pixel(x, y, 1) as f64 + 0.114 * self.get_pixel(x, y, 2) as f64
                ) as u8;
            }
        }
        self.c = 1;
        self.data = grayscale_data;
    }

    pub fn shift_color_rgb(&mut self, c: u32, v: f64) {
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = self.get_idx(x, y, c);
                let new_col = self.data[idx] as f64 * v;
                if new_col > u8::MAX as f64 {
                    self.data[idx] = u8::MAX;
                    continue;
                }
                if new_col < u8::MIN as f64 {
                    self.data[idx] = u8::MIN;
                    continue;
                }
                self.data[idx] = (self.data[idx] as f64 * v) as u8;
            }
        }
    }

    pub fn shift_color_hsv(&self, hsv: &mut Vec<f64>, c: u32, v: f64) {
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = self.get_idx(x, y, c);
                hsv[idx] = hsv[idx] * v;
            }
        }
    }

    pub fn rgb_to_hsv(&self) -> Vec<f64> {
        let mut hsv_data: Vec<f64> = vec![0.0; (self.h * self.w * self.c) as usize];
        for y in 0..self.h {
            for x in 0..self.w {
                let r_idx = self.get_idx(x, y, 0);
                let g_idx = self.get_idx(x, y, 1);
                let b_idx = self.get_idx(x, y, 2);

                let r = self.data[r_idx];
                let g = self.data[g_idx];
                let b = self.data[b_idx];

                let v = cmp::max(r, cmp::max(g, b)) as f64 / 255.0;
                let m = cmp::min(r, cmp::min(g, b)) as f64 / 255.0;
                let c = v - m;

                let s: f64;
                if v == 0.0 {
                    s = 0.0;
                } else {
                    s = c / v;
                }

                hsv_data[g_idx] = s;
                hsv_data[b_idx] = v;

                let mut h: f64 = 0.0;
                if c == 0.0 {
                    h = 0.0;
                    hsv_data[r_idx] = h;
                    continue;
                }

                let r = r as f64 / 255.0;
                let g = g as f64 / 255.0;
                let b = b as f64 / 255.0;

                if v == r {
                    h = (g - b) / c;
                }
                if v == g {
                    h = ((b - r) / c) + 2.0;
                }
                if v == b {
                    h = ((r - g) / c) + 4.0;
                }
                if h < 0.0 {
                    h = (h / 6.0) + 1.0;
                } else {
                    h = h / 6.0;
                }
                hsv_data[b_idx] = h;
            }
        }
        hsv_data
    }

    pub fn hsv_to_rgb(&mut self, hsv: Vec<f64>) {
        for y in 0..self.h {
            for x in 0..self.w {
                let r_idx = self.get_idx(x, y, 0);
                let g_idx = self.get_idx(x, y, 1);
                let b_idx = self.get_idx(x, y, 2);

                let h = hsv[r_idx];
                let s = hsv[g_idx];
                let v = hsv[b_idx];
         
                let h_i = (h / 6.0).floor() as usize % 6;
                let v_min = (1.0 - s) * v;
                let a = (v - v_min) * (((h as usize % 6) + 6) / 6) as f64;
                let v_inc = v_min + a;
                let v_dec = v - a;

                let mut r = v;
                let mut g = v_inc;
                let mut b = v_min;

                if h_i == 1 {
                    r = v_dec;
                    g = v;
                    b = v_min
                }
                if h_i == 2 {
                    r = v_min;
                    g = v;
                    b = v_inc;
                }
                if h_i == 3 {
                    r = v_min;
                    g = v_dec;
                    b = v;
                }
                if h_i == 4 {
                    r = v_inc;
                    g = v_min;
                    b = v;
                }
                if h_i == 5 {
                    r = v;
                    g = v_min;
                    b = v_dec;
                }

                self.data[r_idx] = (r * 255.0) as u8;
                self.data[g_idx] = (g * 255.0) as u8;
                self.data[b_idx] = (b * 255.0) as u8;
            }
        }
    }
}