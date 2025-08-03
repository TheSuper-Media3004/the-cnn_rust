use minifb::{Key, Scale, Window, WindowOptions};
use mnist::{Mnist, MnistBuilder};
use ndarray::{s, Array, Array1, Array2, ArrayView1, ArrayView2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

struct Conv2D {
    kernel: Array2<f32>,
    input_cache: Array2<f32>,
}

impl Conv2D {
    fn new(kernel_size: (usize, usize)) -> Self {
        let kernel = Array::random(kernel_size, Uniform::new(-0.5, 0.5));
        Self {
            kernel,
            input_cache: Array2::zeros((0, 0)),
        }
    }

    fn forward(&mut self, input: &ArrayView2<f32>) -> Array2<f32> {
        self.input_cache = input.to_owned();
        let (h, w) = input.dim();
        let (kh, kw) = self.kernel.dim();
        let oh = h - kh + 1;
        let ow = w - kw + 1;
        let mut output = Array2::<f32>::zeros((oh, ow));
        for i in 0..oh {
            for j in 0..ow {
                let window = input.slice(s![i..i + kh, j..j + kw]);
                output[[i, j]] = (&window * &self.kernel).sum();
            }
        }
        output
    }

    fn backward(&self, d_output: &ArrayView2<f32>) -> (Array2<f32>, Array2<f32>) {
        let (kh, kw) = self.kernel.dim();
        let mut d_kernel = Array2::<f32>::zeros((kh, kw));
        let (oh, ow) = d_output.dim();
        for i in 0..oh {
            for j in 0..ow {
                let window = self.input_cache.slice(s![i..i + kh, j..j + kw]);
                d_kernel = d_kernel + &(&window * d_output[[i, j]]);
            }
        }
        (d_kernel, Array2::zeros((0,0)))
    }

    fn update(&mut self, d_kernel: &ArrayView2<f32>, learning_rate: f32) {
        self.kernel = &self.kernel - &(d_kernel * learning_rate);
    }
}

struct Dense {
    weights: Array2<f32>,
    biases: Array1<f32>,
    input_cache: Array1<f32>,
}

impl Dense {
    fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Array::random((output_size, input_size), Uniform::new(-0.5, 0.5)),
            biases: Array::random(output_size, Uniform::new(-0.5, 0.5)),
            input_cache: Array1::zeros(0),
        }
    }

    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.input_cache = input.to_owned();
        self.weights.dot(input) + &self.biases
    }

    fn backward(&self, d_output: &ArrayView1<f32>) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
        let d_weights = d_output.to_shape((d_output.len(), 1)).unwrap().dot(&self.input_cache.to_shape((1, self.input_cache.len())).unwrap());
        let d_biases = d_output.to_owned();
        let d_input = self.weights.t().dot(d_output);
        (d_weights, d_biases, d_input)
    }

    fn update(&mut self, d_weights: &ArrayView2<f32>, d_biases: &ArrayView1<f32>, learning_rate: f32) {
        self.weights = &self.weights - &(d_weights * learning_rate);
        self.biases = &self.biases - &(d_biases * learning_rate);
    }
}


fn relu(input: &ArrayView2<f32>) -> Array2<f32> {
    input.mapv(|x| x.max(0.0))
}

fn relu_backward(d_output: &ArrayView2<f32>, original_input: &ArrayView2<f32>) -> Array2<f32> {
    d_output * &original_input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

fn max_pool2d(input: &ArrayView2<f32>, pool_size: usize) -> Array2<f32> {
    let (h, w) = input.dim();
    let out_h = h / pool_size;
    let out_w = w / pool_size;
    let mut output = Array2::<f32>::zeros((out_h, out_w));
    for i in 0..out_h {
        for j in 0..out_w {
            let window = input.slice(s![i * pool_size..(i + 1) * pool_size, j * pool_size..(j + 1) * pool_size]);
            output[[i, j]] = window.iter().fold(f32::MIN, |a, &b| a.max(b));
        }
    }
    output
}

fn max_pool2d_backward(d_output: &ArrayView2<f32>, original_input: &ArrayView2<f32>, pool_size: usize) -> Array2<f32> {
    let mut d_input = Array2::<f32>::zeros(original_input.raw_dim());
    let (out_h, out_w) = d_output.dim();
    for i in 0..out_h {
        for j in 0..out_w {
            let window = original_input.slice(s![i * pool_size..(i + 1) * pool_size, j * pool_size..(j + 1) * pool_size]);
            let max_val = window.iter().fold(f32::MIN, |a, &b| a.max(b));
            for (r, row) in window.indexed_iter() {
                if *row == max_val {
                    let (x, y) = r;
                    d_input[[i * pool_size + x, j * pool_size + y]] = d_output[[i, j]];
                    break;
                }
            }
        }
    }
    d_input
}

fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits.mapv(|x| (x - max_logit).exp());
    let sum_exp = exp.sum();
    exp / sum_exp
}


fn main() {

    let learning_rate = 0.005;
    let epochs = 500; 
    const IMG_WIDTH: usize = 28;
    const IMG_HEIGHT: usize = 28;

    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .base_path("data")
        .label_format_digit()
        .training_set_length(1)
        .finalize();

    let image_for_training = Array2::from_shape_vec((IMG_HEIGHT, IMG_WIDTH), trn_img.clone())
        .expect("Reshape failed")
        .mapv(|x| x as f32 / 255.0);
    let label = trn_lbl[0];

    let mut conv = Conv2D::new((3, 3));
    let pooled_dim = (IMG_HEIGHT - conv.kernel.dim().0 + 1) / 2;
    let mut dense = Dense::new(pooled_dim * pooled_dim, 10);

    let display_buffer: Vec<u32> = trn_img.iter().map(|&pixel| {
        let p = pixel as u32;
        (p << 16) | (p << 8) | p
    }).collect();
    
    let window_title = format!("Training on Image - True Label: {} (Press ESC to exit)", label);
    let mut window = Window::new(
        &window_title,
        IMG_WIDTH,
        IMG_HEIGHT,
        WindowOptions {
            scale: Scale::X16,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");

    println!("Training on a single image with True Label: {}", label);
    println!("--- Press ESC in the image window to stop training ---");

    for epoch in 0..epochs {
      
        if !window.is_open() || window.is_key_down(Key::Escape) {
            println!("Exiting.");
            break;
        }

        let conv_out = conv.forward(&image_for_training.view());
        let relu_out = relu(&conv_out.view());
        let pooled_out = max_pool2d(&relu_out.view(), 2);
        let flat = Array1::from(pooled_out.into_raw_vec());
        let logits = dense.forward(&flat);
        let probs = softmax(&logits);
        let loss = -probs[label as usize].ln();

        if epoch % 5 == 0 {
            let predicted_label = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            println!("Epoch: {:<3} Loss: {:.4} | Predicted: {} | Confidence: {:.2}%", 
                     epoch, loss, predicted_label, probs[predicted_label] * 100.0);
        }

        let mut d_logits = probs.clone();
        d_logits[label as usize] -= 1.0;
        let (d_weights, d_biases, d_flat) = dense.backward(&d_logits.view());
        let d_pooled = d_flat.into_shape((pooled_dim, pooled_dim)).unwrap();
        let d_relu = max_pool2d_backward(&d_pooled.view(), &relu_out.view(), 2);
        let d_conv = relu_backward(&d_relu.view(), &conv_out.view());
        let (d_kernel, _) = conv.backward(&d_conv.view());
        
        conv.update(&d_kernel.view(), learning_rate);
        dense.update(&d_weights.view(), &d_biases.view(), learning_rate);

        window.update_with_buffer(&display_buffer, IMG_WIDTH, IMG_HEIGHT).unwrap();
    }
}