use std::path::Path;

use image::ImageError;

use rust_vision::{ Image, SqaureBoxFilter };

fn main() -> Result<(), ImageError> {
    let img = Image::load_image(Path::new("data/dog.jpg"))?;
    let gauss = SqaureBoxFilter::new_gaussian_filter(1.0);
    let img = img.convolve(&gauss, true);
    let sobel_x = SqaureBoxFilter::new_sobel_filter_x();
    let sobel_y = SqaureBoxFilter::new_sobel_filter_y();
    let (g, _) = img.sobel(&sobel_x, &sobel_y);
    g.write(Path::new("data/magnitude.jpg"))?;
    Ok(())
} 