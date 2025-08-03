use ndarray::Array3;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        input.mapv(|x| x.max(0.0))
    }

    pub fn backward(&self, grad_output: &Array3<f32>) -> Array3<f32> {
        grad_output.mapv(|g| if g > 0.0 { g } else { 0.0 })
    }
}