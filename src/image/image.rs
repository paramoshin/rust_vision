extern crate image;

use std::cmp;
use std::u8;
use std::f64;
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
                let new_col = self.data[idx] as f64 + (v * 255.0);
                let new_col = Image::float_to_rgb(new_col);
                self.data[idx] = new_col as u8;
            }
        }
    }

    pub fn shift_color_hsv(&self, hsv: &mut Vec<f64>, c: u32, v: f64) {
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = self.get_idx(x, y, c);
                if c < 2 {
                    hsv[idx] = hsv[idx] + v;
                } else {
                    hsv[idx] = hsv[idx] + v * 255.0;
                }
            }
        }
    }

    pub fn scale_color_rgb(&mut self, c: u32, v: f64) {
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = self.get_idx(x, y, c);
                self.data[idx] = Image::float_to_rgb(v * self.data[idx] as f64);
            }
        }
    }

    pub fn scale_color_hsv(&mut self, hsv: &mut Vec<f64>, c: u32, v: f64) {
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = self.get_idx(x, y, c);
                    hsv[idx] *= v;
            }
        }
    }

    fn rgb_pix_to_hsv_pix(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
        let mut h: f64;
        let s: f64;
        let v: f64;

        let cmin = cmp::min(r, cmp::min(g, b));
        let cmax = cmp::max(r, cmp::max(g, b));

        v = cmax as f64;
        let delta = (cmax - cmin) as f64;
        if delta < 0.00001 {
            s = 0.0;
            h = 0.0;
            return (h, s, v);
        }
        if cmax == 0 {
            s = 0.0;
            h = 0.0;
            return (h, s, v);
        } else {
            s = delta / cmax as f64;
        }
        h = match cmax {
            max if r >= max => (g as f64 - b as f64) / delta,
            max if g >= max => 2.0 + (b as f64 - r as f64) / delta,
            _               => 4.0 + (r as f64 - g as f64) / delta,
        };
        h *= 60.0;
        if h < 0.0 {
            h += 360.0
        }
        (h, s, v)
    }

    fn float_to_rgb(f: f64) -> u8 {
        if f > u8::MAX as f64 {
            u8::MAX
        } else if f < u8::MIN as f64 {
            u8::MIN
        } else {
            f as u8
        }
    }

    fn hsv_pix_to_rgb_pix(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
        let r: u8;
        let g: u8;
        let b: u8;

        let hh = if h >= 360.0 {
            0.0
        } else {
            h / 60.0
        };
        
        let i = hh as u8;
        let ff = hh - i as f64;
        let p = v * (1.0 - s);
        let q = v * (1.0 - (s * ff));
        let t = v * (1.0 - (s * (1.0 - ff)));

        let v = Image::float_to_rgb(v);
        let p = Image::float_to_rgb(p);
        let q = Image::float_to_rgb(q);
        let t = Image::float_to_rgb(t);

        match i {
            0 => {
                r = v;
                g = t;
                b = p;
            },
            1 => {
                r = q;
                g = v;
                b = p;
            },
            2 => {
                r = p;
                g = v;
                b = t;
            },
            3 => {
                r = p;
                g = q;
                b = v;
            },
            4 => {
                r = t;
                g = p;
                b = v;
            },
            _ => {
                r = v;
                g = p;
                b = q;
            },
        }

        (r, g, b)
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

                let(h, s, v) = Image::rgb_pix_to_hsv_pix(r, g, b);

                hsv_data[r_idx] = h;
                hsv_data[g_idx] = s;
                hsv_data[b_idx] = v;
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

                let (r, g, b) = Image::hsv_pix_to_rgb_pix(h, s, v);
                
                self.data[r_idx] = r;
                self.data[g_idx] = g;
                self.data[b_idx] = b;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_hsv_to_rgb() {
        let r: u8 = 0;
        let g: u8 = 0;
        let b: u8 = 0;
        let (h, s, v) = Image::rgb_pix_to_hsv_pix(r, g, b);
        assert_eq!((0.0, 0.0, 0.0), Image::rgb_pix_to_hsv_pix(r, g, b));
        assert_eq!(Image::hsv_pix_to_rgb_pix(h, s, v), (r, g, b));

        let r: u8 = 255;
        let g: u8 = 0;
        let b: u8 = 0;
        let (h, s, v) = Image::rgb_pix_to_hsv_pix(r, g, b);
        assert_eq!((0.0, 1.0, 255.0), (h, s, v));
        assert_eq!(Image::hsv_pix_to_rgb_pix(h, s, v), (r, g, b));

        let r: u8 = 0;
        let g: u8 = 255;
        let b: u8 = 0;
        let (h, s, v) = Image::rgb_pix_to_hsv_pix(r, g, b);
        assert_eq!((120.0, 1.0, 255.0), (h, s, v));
        assert_eq!(Image::hsv_pix_to_rgb_pix(h, s, v), (r, g, b));
        
        let r: u8 = 0;
        let g: u8 = 0;
        let b: u8 = 255;
        let (h, s, v) = Image::rgb_pix_to_hsv_pix(r, g, b);
        assert_eq!((240.0, 1.0, 255.0), (h, s, v));
        assert_eq!(Image::hsv_pix_to_rgb_pix(h, s, v), (r, g, b));

        let r: u8 = 255;
        let g: u8 = 255;
        let b: u8 = 255;
        let (h, s, v) = Image::rgb_pix_to_hsv_pix(r, g, b);
        assert_eq!((0.0, 0.0, 255.0), (h, s, v));
        assert_eq!(Image::hsv_pix_to_rgb_pix(h, s, v), (r, g, b));

        let r: u8 = 120;
        let g: u8 = 200;
        let b: u8 = 50;
        let (h, s, v) = Image::rgb_pix_to_hsv_pix(r, g, b);
        assert_eq!((92.0, 0.75, 200.0), (h, s, v));
        assert_eq!(Image::hsv_pix_to_rgb_pix(h, s, v), (119, g, b));
    }
}