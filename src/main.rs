use std::path::Path;

use image::ImageError;

use rust_vision::{ Image, SqaureBoxFilter };

fn main() -> Result<(), ImageError> {
    let mut img = Image::load_image(Path::new("data/dog.jpg"))?;
    let f = SqaureBoxFilter::new_gaussian_filger(2.0);
    img.convolve(f, true);
    img.write(Path::new("data/dog_gauss_2.jpg"))?;
    Ok(())
} 