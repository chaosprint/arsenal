use arsenal::*;

fn main() {
    let data = read_wav_file("amen_break.wav").unwrap();
    play_audio(data).unwrap();
}
