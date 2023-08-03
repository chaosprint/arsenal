use arsenal::*;

// cargo run -r --example plot-fft
fn main() {
    let data = read_wav_file("sin440.wav").unwrap();
    let r = spectral_analysis(data[0].clone(), 1024);
    plot_spectrum(r.0).unwrap();
}
