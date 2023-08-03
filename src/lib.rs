use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use eframe::egui;
#[allow(unused_imports)]
use egui::plot::{AxisBools, GridInput, GridMark, PlotResponse};
use egui::*;
use hound::WavReader;
#[allow(unused_imports)]
use plot::{
    Arrows, Bar, BarChart, BoxElem, BoxPlot, BoxSpread, CoordinatesFormatter, Corner, HLine,
    Legend, Line, LineStyle, MarkerShape, Plot, PlotImage, PlotPoint, PlotPoints, Points, Polygon,
    Text, VLine,
};
use smallvec::{smallvec as svec, SmallVec};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rustfft::algorithm::Radix4;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::Fft;
use rustfft::FftDirection;

pub type AudioData = SmallVec<[Vec<f32>; 2]>;

pub fn read_wav_file<P: AsRef<Path>>(path: P) -> Result<AudioData, hound::Error> {
    let mut reader = WavReader::open(path).expect("Failed to open WAV file");
    let spec = reader.spec();
    println!("_________Sample spec: {:?}", spec);
    let num_channels = spec.channels as usize;
    let mut data: AudioData = svec![];
    for _ in 0..num_channels {
        data.push(Vec::new());
    }

    let mut sample_count = 0;

    match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => {
                for result in reader.samples::<i16>() {
                    let sample = result? as f32 / i16::MAX as f32;
                    let channel = sample_count % num_channels;
                    data[channel].push(sample);
                    sample_count += 1;
                }
            }

            24 => {
                for result in reader.samples::<i32>() {
                    let sample = result?;
                    let sample = if sample & (1 << 23) != 0 {
                        (sample | !0xff_ffff) as f32
                    } else {
                        sample as f32
                    };
                    let sample = sample / (1 << 23) as f32;
                    let channel = sample_count % num_channels;
                    data[channel].push(sample as f32);
                    sample_count += 1;
                }
            }

            32 => {
                for result in reader.samples::<i32>() {
                    let sample = result? as f32 / i32::MAX as f32;
                    let channel = sample_count % num_channels;
                    data[channel].push(sample);
                    sample_count += 1;
                }
            }
            _ => return Err(hound::Error::Unsupported),
        },
        hound::SampleFormat::Float => {
            for result in reader.samples::<f32>() {
                let sample = result?;
                let channel = sample_count % num_channels;
                data[channel].push(sample);
                sample_count += 1;
            }
        }
    }

    Ok(data)
}

pub fn play_audio(data: AudioData) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("Failed to get default output device");
    let config = device
        .default_output_config()
        .expect("Failed to get default output config");

    match config.sample_format() {
        cpal::SampleFormat::F32 => play::<f32>(&device, &config.into(), data),
        cpal::SampleFormat::I16 => play::<i16>(&device, &config.into(), data),
        cpal::SampleFormat::U16 => play::<u16>(&device, &config.into(), data),
        _ => panic!("unhandled sample format"),
    }
}

fn play<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    data: AudioData,
) -> anyhow::Result<()>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    assert!(channels == data.len(), "Mismatch in number of channels");

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let position = AtomicUsize::new(0);
    let data = Arc::new(Mutex::new(data));

    let stream = device.build_output_stream(
        config,
        move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
            let data = data.lock().unwrap();
            for frame in output.chunks_mut(channels) {
                let pos = position.fetch_add(1, Ordering::Relaxed);
                for (sample, channel_data) in frame.iter_mut().zip(data.iter()) {
                    let value: T = T::from_sample(channel_data[pos % channel_data.len()]);
                    *sample = value;
                }
            }
        },
        err_fn,
        None,
    )?;

    stream.play()?;
    std::thread::sleep(std::time::Duration::from_secs(3000));
    Ok(())
}

pub fn plot(data: Vec<Vec<f32>>) -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(800.0, 600.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Arsenal",
        options,
        Box::new(|_cc| Box::new(PlotApp::new(data))),
    )
}

struct PlotApp {
    audio_data: Vec<Vec<f32>>,
}

impl PlotApp {
    fn new(data: Vec<Vec<f32>>) -> Self {
        Self { audio_data: data }
    }
}

impl eframe::App for PlotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Audio data plotter");
            let plot = Plot::new("audio").legend(Legend::default());
            plot.show(ui, |plot_ui| {
                for channel in self.audio_data.iter() {
                    plot_ui.line(Line::new(PlotPoints::from_ys_f32(channel)));
                }
            })
            .response;
        });
    }
}

pub fn spectral_analysis(samples: Vec<f32>, window_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let fft = Radix4::new(window_size, FftDirection::Forward);
    let mut input: Vec<Complex<f32>> = vec![Complex::zero(); window_size];
    let mut magnitudes: Vec<Vec<f32>> = Vec::new();
    let mut phases: Vec<Vec<f32>> = Vec::new();
    for window in samples.chunks(window_size) {
        if window.len() == window_size {
            for (sample, input) in window.iter().zip(input.iter_mut()) {
                *input = Complex::new(*sample, 0.0);
            }
            fft.process(&mut input);
            let magnitude: Vec<f32> = input.iter().map(|c| c.norm()).collect();
            let phase: Vec<f32> = input.iter().map(|c| c.arg()).collect();
            magnitudes.push(magnitude);
            phases.push(phase);
        }
    }

    (magnitudes, phases)
}

pub fn plot_spectrum(data: Vec<Vec<f32>>) -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(800.0, 600.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Arsenal",
        options,
        Box::new(|_cc| Box::new(PlotSpectrumApp::new(data))),
    )
}

struct PlotSpectrumApp {
    spectrum_data: Vec<Vec<f32>>,
}

impl PlotSpectrumApp {
    fn new(data: Vec<Vec<f32>>) -> Self {
        Self {
            spectrum_data: data,
        }
    }
    fn markers(&self) -> Vec<Points> {
        let mut markers = vec![];
        for (x, bin) in self.spectrum_data.iter().enumerate() {
            let half = &bin[0..512];
            for (y, depth) in half.iter().enumerate() {
                let points = Points::new(vec![[x as f64, y as f64 * 42.]])
                    .shape(MarkerShape::Square)
                    .radius(3.)
                    .color(Color32::from_gray((depth / 50.0 * 255.0) as u8));
                markers.push(points);
            }
        }
        markers
    }
}

impl eframe::App for PlotSpectrumApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Audio spectrum plotter");
            let plot = Plot::new("audio").legend(Legend::default());
            plot.show(ui, |plot_ui| {
                // let markers = ;
                for m in self.markers() {
                    plot_ui.points(m);
                }
            })
            .response
        });
    }
}
