use ndarray::{s, Array3};

pub struct MaxPool2D {
    pub size: usize,
}

impl MaxPool2D {
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (h, w, d) = input.dim();
        let out_h = h / self.size;
        let out_w = w / self.size;
        let mut output = Array3::zeros((out_h, out_w, d));

        for z in 0..d {
            for i in 0..out_h {
                for j in 0..out_w {
                    let patch = input.slice(s![i * self.size..(i + 1) * self.size, j * self.size..(j + 1) * self.size, z]);
                    output[[i, j, z]] = patch.iter().cloned().fold(f32::MIN, f32::max);
                }
            }
        }
        output
    }

    pub fn backward(&self, grad_output: &Array3<f32>) -> Array3<f32> {
        grad_output.mapv(|x| x) // dummy pass-through
    }
}
