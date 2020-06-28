extern crate image;

use std::cmp;
use std::u8;
use std::f64;
use std::path::Path;
use std::ops;

#[derive(Clone)]
pub struct Image {
    pub w: u32,
    pub h: u32,
    pub c: u32,
    pub data: Vec<u8>,
}

pub struct SqaureBoxFilter {
    pub w: u32,
    pub data: Vec<f64>,
}

impl SqaureBoxFilter {
    pub fn new(w: u32) -> Self {
        let data = vec![1.0 / (w * w) as f64; (w * w) as usize];
        SqaureBoxFilter {
            w: w,
            data: data,
        }
    }

    pub fn new_highpass_filter() -> Self {
        let data: Vec<f64> = vec![0.0, -1.0, 0.0, 
                                  -1.0, 4.0, -1.0,
                                  0.0, -1.0, 0.0];
        SqaureBoxFilter {
            w: 3,
            data: data
        }
    }

    pub fn new_sharpen_filter() -> Self {
        let data: Vec<f64> = vec![0.0, -0.5, 0.0,
                                  -0.5, 3.0, -0.5, 
                                  0.0, -0.5, 0.0];
        SqaureBoxFilter {
            w: 3,
            data: data
        }
    }

    pub fn new_emboss_filter() -> Self {
        let data: Vec<f64> = vec![-2.0, -1.0, 0.0, 
                                  -1.0, 1.0, 1.0, 
                                  0.0, 1.0, 2.0];
        SqaureBoxFilter {
            w: 3,
            data: data
        }
    }

    pub fn new_sobel_filter_x() -> Self {
        let data: Vec<f64> = vec![1.0, 0.0, -1.0, 
                                  2.0, 0.0, -2.0, 
                                  1.0, 0.0, -1.0];
        SqaureBoxFilter {
            w: 3,
            data: data,
        }
    }

    pub fn new_sobel_filter_y() -> Self {
        let data: Vec<f64> = vec![1.0, 2.0, 1.0, 
                                  0.0, 0.0, 0.0, 
                                  -1.0, -2.0, -1.0];
        SqaureBoxFilter {
            w: 3,
            data: data,
        }
    }

    pub fn new_gaussian_filter(sigma: f64) -> Self {
        let w: u32 = if (sigma * 6.0).round() as u32 % 2 == 0 {
            (sigma * 6.0).round() as u32 + 1
        } else {
            (sigma * 6.0).round() as u32
        };
        let mut data: Vec<f64> = vec![0.0; (w * w) as usize];
        for i in 0..w {
            for j in 0..w {
                let x: f64 = (i as i32 - (w / 2) as i32) as f64;
                let y: f64 = (j as i32 - (w / 2) as i32) as f64;
                data[(i + j * w) as usize] = (-(x.powi(2) + y.powi(2)) / (2.0 * sigma.powi(2))).exp() / (2.0 * f64::consts::PI * sigma.powi(2));
            }
        }
        SqaureBoxFilter {
            w: w,
            data: data,
        }
    }

    pub fn from_data(data: Vec<f64>) -> Self {
        SqaureBoxFilter {
            w: (data.len() as f64).sqrt() as u32,
            data: data,
        }
    }
}

impl ops::Add<&Image> for &Image {
    type Output = Image;

    fn add(self, other: &Image) -> Image {
        assert_eq!(self.w, other.w);
        assert_eq!(self.h, other.h);
        assert_eq!(self.c, other.c);
        let mut img = Image::new(self.w, self.h, self.c);
        for y in 0..self.h {
            for x in 0..self.w {
                for c in 0..self.c {
                    let v: u8 = Image::float_to_rgb(
                        self.get_pixel(x, y, c) as f64 + other.get_pixel(x, y, c) as f64
                    );
                    img.set_pixel(x, y, c, v);
                }
            }
        }
        img
    }
}

impl ops::Sub<&Image> for &Image {
    type Output = Image;

    fn sub(self, other: &Image) -> Image {
        assert_eq!(self.w, other.w);
        assert_eq!(self.h, other.h);
        assert_eq!(self.c, other.c);
        let mut img = Image::new(self.w, self.h, self.c);
        for y in 0..self.h {
            for x in 0..self.w {
                for c in 0..self.c {
                    let v: u8 = Image::float_to_rgb(
                        self.get_pixel(x, y, c) as f64 - other.get_pixel(x, y, c) as f64
                    );
                    img.set_pixel(x, y, c, v);
                }
            }
        }
        img
    }
}

impl ops::Mul<&Image> for &Image {
    type Output = Image;

    fn mul(self, other: &Image) -> Image {
        assert_eq!(self.w, other.w);
        assert_eq!(self.h, other.h);
        assert_eq!(self.c, other.c);
        let mut img = Image::new(self.w, self.h, self.c);
        for y in 0..self.h {
            for x in 0..self.w {
                for c in 0..self.c {
                    let v: u8 = Image::float_to_rgb(
                        self.get_pixel(x, y, c) as f64 * other.get_pixel(x, y, c) as f64
                    );
                    img.set_pixel(x, y, c, v);
                }
            }
        }
        img
    }
}

impl ops::Add<&Image> for Image {
    type Output = Image;

    fn add(self, other: &Image) -> Image {
        assert_eq!(self.w, other.w);
        assert_eq!(self.h, other.h);
        assert_eq!(self.c, other.c);
        let mut img = Image::new(self.w, self.h, self.c);
        for y in 0..self.h {
            for x in 0..self.w {
                for c in 0..self.c {
                    let v: u8 = Image::float_to_rgb(
                        self.get_pixel(x, y, c) as f64 + other.get_pixel(x, y, c) as f64
                    );
                    img.set_pixel(x, y, c, v);
                }
            }
        }
        img
    }
}

impl ops::Sub<&Image> for Image {
    type Output = Image;

    fn sub(self, other: &Image) -> Image {
        assert_eq!(self.w, other.w);
        assert_eq!(self.h, other.h);
        assert_eq!(self.c, other.c);
        let mut img = Image::new(self.w, self.h, self.c);
        for y in 0..self.h {
            for x in 0..self.w {
                for c in 0..self.c {
                    let v: u8 = Image::float_to_rgb(
                        self.get_pixel(x, y, c) as f64 - other.get_pixel(x, y, c) as f64
                    );
                    img.set_pixel(x, y, c, v);
                }
            }
        }
        img
    }
}

impl ops::Mul<&Image> for Image {
    type Output = Image;

    fn mul(self, other: &Image) -> Image {
        assert_eq!(self.w, other.w);
        assert_eq!(self.h, other.h);
        assert_eq!(self.c, other.c);
        let mut img = Image::new(self.w, self.h, self.c);
        for y in 0..self.h {
            for x in 0..self.w {
                for c in 0..self.c {
                    let v: u8 = Image::float_to_rgb(
                        self.get_pixel(x, y, c) as f64 * other.get_pixel(x, y, c) as f64
                    );
                    img.set_pixel(x, y, c, v);
                }
            }
        }
        img
    }
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
    
    fn nn_interpolate(&self, x: f64, y: f64, c: u32) -> u8 {
        let nn_x: u32 = if x < 0.0 {
            0
        } else if x.round() as u32 >= self.w {
            self.w - 1
        } else {
            x.round() as u32
        };
        let nn_y: u32 = if y < 0.0 {
            0
        } else if y.round() as u32 >= self.h {
            self.h - 1
        } else {
            y.round() as u32
        };
        self.get_pixel(nn_x, nn_y, c)
    }

    pub fn nn_resize(&self, w: u32, h: u32) -> Image {
        let mut img = Image::new(w, h, self.c);
        let x_step: f64 = self.w as f64 / w as f64;
        let y_step: f64 = self.h as f64 / h as f64;
        for y in 0..img.h {
            for x in 0..img.w {
                for i in 0..img.c {
                    let idx = img.get_idx(x, y, i);
                    img.data[idx] = self.nn_interpolate(x as f64 * x_step, y as f64 * y_step, i);
                }
            }
        }
        img
    }

    fn bilinear_interpolate(&self, x: f64, y: f64, c: u32) -> u8 {
        let get_smaller_idx = |f: f64, u: u32| -> u32 {
            if f < 0.0 {
                0
            } else if f as u32 >= u {
                u - 1
            } else {
                f as u32
            }
        };
        let get_bigger_idx = |f: f64, u: u32| -> u32 {
            if f < 0.0 {
                0
            } else if f as u32 >= u - 1 {
                u - 1
            } else {
                f as u32 + 1
            }
        };
        let x1 = get_smaller_idx(x, self.w);
        let x2 = get_bigger_idx(x, self.w);
        let y1 = get_smaller_idx(y, self.h);
        let y2 = get_bigger_idx(y, self.h);

        let p1: u8 = self.get_pixel(x1, y1, c);
        let p2: u8 = self.get_pixel(x2, y1, c);
        let p3: u8 = self.get_pixel(x1, y2, c);
        let p4: u8 = self.get_pixel(x2, y2, c);

        let a1: f64 = (x2 as f64 - x) * (y2 as f64 - y);
        let a2: f64 = (x - x1 as f64) * (y2 as f64 - y);
        let a3: f64 = (x2 as f64 - x) * (y - y1 as f64);
        let a4: f64 = (x - x1 as f64) * (y - y1 as f64);

        (a1 * p1 as f64 + a2 * p2 as f64 + a3 * p3 as f64 + a4 * p4 as f64) as u8
    }

    pub fn bilinear_resize(&self, w: u32, h: u32) -> Image {
        let mut img = Image::new(w, h, self.c);
        let x_step: f64 = self.w as f64 / w as f64;
        let y_step: f64 = self.h as f64 / h as f64;
        for y in 0..img.h {
            for x in 0..img.w {
                for i in 0..img.c {
                    let idx = img.get_idx(x, y, i);
                    img.data[idx] = self.bilinear_interpolate(x as f64 * x_step, y as f64 * y_step, i);
                }
            }
        }
        img
    }

    fn conv_pix(&self, x: u32, y: u32, c: u32, f: &SqaureBoxFilter) -> f64 {
        let mut v: f64 = 0.0;
        for i in 0..f.w {
            for j in 0..f.w {
                let x_idx: u32 = if x < (f.w / 2) + j {
                    0
                } else if (x as i32 - (f.w / 2) as i32 + j as i32) >= self.w as i32 {
                    self.w - 1
                } else {
                    x - (f.w / 2) + j
                };
                let y_idx: u32 = if y < (f.w / 2) + i {
                    0
                } else if (y as i32 - (f.w / 2) as i32 + i as i32) >= self.h as i32 {
                    self.h - 1
                } else {
                    y - (f.w / 2) + i
                };
                v += self.get_pixel(x_idx, y_idx, c) as f64 * f.data[(j + i * f.w) as usize];
            }
        }
        v
    }

    pub fn convolve(&self, f: &SqaureBoxFilter, preserve: bool) -> Self {

        let mut img = if preserve {
            Image::new(self.w, self.h, self.c)
        } else {
            Image::new(self.w, self.h, 1)
        };
        
        for y in 0..self.h {
            for x in 0..self.w {
                let mut v: f64 = 0.0;
                for c in 0..self.c {
                    if preserve {
                        let v = self.conv_pix(x, y, c, f);
                        let v = Image::float_to_rgb(v);
                        img.set_pixel(x, y, c, v);
                    } else {
                        v += self.conv_pix(x, y, c, f);
                    }
                }
                if !preserve {
                    let v = Image::float_to_rgb(v);
                    let idx = self.get_idx(x, y, 0);
                    img.data[idx] = v;
                }
            }
        }
        img
    }

    pub fn min_max_normalize(data: &Vec<i32>, w: u32, h: u32) -> Vec<u8> {
        let mut norm_data: Vec<u8> = vec![0; (w * h) as usize];

        let min_: i32 =  match data.iter().min() {
            Some(min) => *min,
            None      => 0
        };
        let max_: i32 = match data.iter().max() {
            Some(max) => *max,
            None      => 0
        };
        let delta = max_ - min_;

        for y in 0..h {
            for x in 0..w {
                let idx = (x + y * w) as usize;
                if delta == 0 {
                    norm_data[idx] = 0
                } else {
                    let v: f64 = u8::MIN as f64 + (((data[idx] - min_) as f64 * (u8::MAX as f64 - u8::MIN as f64)) / (max_ - min_) as f64);
                    norm_data[idx] = v as u8;
                }
            }
        }
        norm_data
    }

    pub fn sobel(&self, sobel_x: &SqaureBoxFilter, sobel_y: &SqaureBoxFilter) -> (Self, Self) {
        let g_x: Image = self.convolve(sobel_x, false);
        let g_y: Image = self.convolve(sobel_y, false);

        let mut g_data: Vec<i32> = vec![0; (self.w * self.h) as usize];
        let mut g: Image = Image::new(self.w, self.h, 1);
        let mut dir = Image::new(self.w, self.h, 1);

        for y in 0..self.h {
            for x in 0..self.w {
                let g_x_pix: f64 = g_x.get_pixel(x, y, 0) as f64;
                let g_y_pix: f64 = g_y.get_pixel(x, y, 0) as f64;
                g_data[g.get_idx(x, y, 0)] = (g_x_pix.powi(2) + g_y_pix.powi(2)).sqrt() as i32;
                dir.set_pixel(x, y, 0, Image::float_to_rgb((g_y_pix / g_x_pix).atan()));
            }
        }
        g.data = Image::min_max_normalize(&g_data, self.w, self.h);

        (g, dir)
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