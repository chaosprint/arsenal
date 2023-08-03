use arsenal::*;
use std::thread;

// cargo run -r --example plot-waveform
fn main() {
    let data = read_wav_file("dun_dun_dun.wav").unwrap();
    let play_data = data.clone();
    thread::spawn(move || {
        play_audio(play_data).unwrap();
    });
    plot(data.to_vec()).unwrap();
}
