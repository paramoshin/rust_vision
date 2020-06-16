use std::path::Path;

use image::ImageError;

use rust_vision::image::image::Image;

fn main() -> Result<(), ImageError> {
    let img = Image::load_image(Path::new("data/dog.jpg"))?;
    img.write_image(Path::new("data/dog_test.jpg"))?;
    Ok(())
}
