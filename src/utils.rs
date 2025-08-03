use ndarray::{s, Array3};
use image::{GrayImage, Luma};

pub fn load_mnist() -> (Vec<Array3<f32>>, Vec<usize>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let images = (0..100)
        .map(|_| Array3::from_elem((28, 28, 1), rng.gen_range(0.0..1.0)))
        .collect();
    let labels = (0..100).map(|_| rng.gen_range(0..10)).collect();
    (images, labels)
}

pub fn save_feature_map(map: &Array3<f32>, filename_prefix: &str) {
    let (h, w, channels) = map.dim();
    for c in 0..channels {
        let mut img = GrayImage::new(w as u32, h as u32);
        let slice = map.slice(s![.., .., c]);

        let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = if max - min == 0.0 { 1.0 } else { 255.0 / (max - min) };

        for y in 0..h {
            for x in 0..w {
                let val = ((slice[[y, x]] - min) * scale).clamp(0.0, 255.0) as u8;
                img.put_pixel(x as u32, y as u32, Luma([val]));
            }
        }

        let filename = format!("{}_channel_{}.png", filename_prefix, c);
        img.save(&filename).unwrap();
    }
}
