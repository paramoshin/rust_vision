extern crate image;

use std::path::Path;

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
            data: Vec::new(),
        }
    }

    pub fn load_image(path: &Path) -> Result<Self, image::ImageError> {

        let image = image::open(path)?.to_rgb();

        let (w, h) = image.dimensions();
        let c = 3;
        let mut img = Image::new(w, h, c);

        for i in 0..c {
            for y in 0..h {
                for x in 0..w {
                    // let idx = ((y * w + x) * c) as usize;
                    img.data.push(image.get_pixel(x, y)[i as usize]);
                    // img.data[idx] = image.get_pixel(x, y)[i as usize];
                }
            }
        }
        Ok(img)
    }

    pub fn write_image(&self, path: &Path) -> Result<(), image::ImageError> {
        let mut image = image::RgbImage::new(self.w, self.h);
        for y in 0..self.h {
            for x in 0..self.w {
                image.put_pixel(x, y, image::Rgb([self.data[(x + y * self.w) as usize], self.data[(x + y * self.w + self.w * self.h) as usize], self.data[(x + y * self.w + 2 * self.w * self.h) as usize]]))
            }
        }
        image.save(path)?;
        // let buf: &[u8] = &self.data;
        // image::save_buffer(path, buf, self.w, self.h, image::ColorType::Rgb8)?;
        Ok(())
    }
}