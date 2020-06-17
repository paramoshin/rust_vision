use std::path::Path;

use image::ImageError;

use rust_vision::image::image::Image;

fn main() -> Result<(), ImageError> {
    let mut img = Image::load_image(Path::new("data/dog.jpg"))?;
    let mut hsv = img.rgb_to_hsv();
    img.scale_color_hsv(&mut hsv, 1, 2.0);
    img.hsv_to_rgb(hsv);
    img.write(Path::new("data/dog_scale_saturated.jpg"))?;
    Ok(())
}
