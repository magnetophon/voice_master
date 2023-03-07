use atomic_float::AtomicF32;

use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
// use pitch_detection::detector::mcleod::McLeodDetector;
use pyin::PadMode::Constant;
use pyin::{Framing, PYINExecutor};

use irapt::{Irapt, Parameters};
use std::collections::VecDeque;

use simple_eq::design::Curve;
use simple_eq::*;
use std::sync::Arc;

use rubato::{FftFixedInOut,Resampler};

use pitch::detect;
use pitch_detection::detector::mcleod::McLeodDetector;

mod editor;
mod mc_pitch;

// TODO:
//
// remove the 32 copies and just use one writepointer and 32 read pointers
// make than nr variable
// make that variable dependent on detector-size, so we  always get the same nr of pitches per second
//
// compare pitch trackers:
// https://docs.rs/pyin/1.0.2/pyin/
// https://crates.io/crates/irapt
// https://crates.io/crates/pitch
// https://crates.io/crates/pitch-detector
// https://crates.io/crates/pitch-detection
//
// https://github.com/sevagh/pitch-detection/issues/63
// https://github.com/marl/crepe
// https://github.com/sevagh/pitch-detection/tree/master/misc/probabilistic-mcleod
// https://www.eecs.qmul.ac.uk/~simond/pub/2014/MauchDixon-PYIN-ICASSP2014.pdf

// add pitch override options:
//
// override kind param:
//   - no override
//   - use last pitch
//   - use midi input
//
// for the vibrato during hold, add an LFO that get's it's pitch from
// a separte pitchtracker that works as follows:
//  - plot the ratio between pitch and avg pitch, track the pitch of that plot
//  - avg pitch is calculated:
//    - make a ringbuf for each note we want to track
//    - for each note:
//    - if ( abs_diff(i) < 1  )
//        add note to ringbuf[i]
//
//
// parallel detectors at different bufsizes::
// one at 64 samples, one at 256 and one at 2048
// when we go unvoiced, remember the octave we where in
// when there is a pitch, use that,
// optionaly in the last octave we where in, if it's lower than it's octave.
// when a slower pitch-detector comes with a lower pitch, use that
// find the fastest detector that has the current pitch-class and use it's pitches, switched to the correct octave
//
// possibly use a different samplerate for each detector

/// The time it takes for the peak meter to decay by 12 dB after switching to complete silence.
const PEAK_METER_DECAY_MS: f64 = 150.0;

/// Blocksize of the detector, determines the lowest pitch that can be detected at a given samplerate.
/// 2^6 = 64 samples
/// 44100/64 = 689 Hz at a samplerate of 44.1k
/// for when you want to track your picolo flute with really low latency!
const MIN_DETECTOR_SIZE_POWER: usize = 6;
// const MIN_DETECTOR_SIZE_POWER: usize = 11;
/// 2^13 = 8192
/// 192000/8192 = 23.4Hz at a samplerate of 192k
/// for when you want to play 6 string bassguitar at 192k
const MAX_DETECTOR_SIZE_POWER: usize = 13;
// const MAX_DETECTOR_SIZE_POWER: usize = 11;
/// the number of detectors we need, one for each size
const NR_OF_DETECTORS: usize = MAX_DETECTOR_SIZE_POWER - MIN_DETECTOR_SIZE_POWER + 1;
const MAX_SIZE: usize = 2_usize.pow(MAX_DETECTOR_SIZE_POWER as u32);
/// the maximum nr of times the detector is updated each 2048 samples
const MAX_OVERLAP: usize = 128;
/// The median is taken from at max this nr of pitches
const MAX_MEDIAN_NR: usize = 32;
const MEDIAN_NR_DEFAULT: i32 = 1;
const DOWNSAMPLE_RATIO: usize = 8;
const DOWNSAMPLED_RATE: usize = 6000;

/// This is mostly identical to the gain example, minus some fluff, and with a GUI.
pub struct VoiceMaster {
    params: Arc<VoiceMasterParams>,

    /// Needed to normalize the peak meter's response based on the sample rate.
    peak_meter_decay_weight: f32,
    /// The current data for the peak meter. This is stored as an [`Arc`] so we can share it between
    /// the GUI and the audio processing parts. If you have more state to share, then it's a good
    /// idea to put all of that in a struct behind a single `Arc`.
    ///
    /// This is stored as voltage gain.
    peak_meter: Arc<AtomicF32>,
    /// sample rate
    sample_rate: f32,
    /// the delay-line to use for the latency compensation
    delay_line: Vec<f32>,
    /// the signal to be used in the pitchtrackers
    signal: Vec<f32>,
    /// the sample index of the above signal
    signal_index: usize,
    /// the curent pitch and clarity
    pitch_val: [f32; 2],
    /// vector of pitches to pick the median from
    pitches: Vec<f32>,
    /// index into "pitches", to use it as a ringbuffer
    median_index: usize,
    /// previous value of the output saw, to calculate the new one from
    previous_saw: f32,
    /// previous value of the raw pitch detector, to calculate the change rate from
    previous_pitch: f32,
    /// the final pitch that we are using
    final_pitch: f32,
    /// an array of pitch detectors, one for each size:
    detectors: [McLeodDetector<f32>; NR_OF_DETECTORS],
    pyin_exec: [PYINExecutor<f32>; NR_OF_DETECTORS],
    irapt: Irapt,
    eq: Equalizer<f32>,
    resampler: FftFixedInOut<f32>,
}

#[derive(Params)]
struct VoiceMasterParams {
    /// The parameter's ID is used to identify the parameter in the wrappred plugin API. As long as
    /// these IDs remain constant, you can rename and reorder these fields as you wish. The
    /// parameters are exposed to the host in the same order they were defined. In this case, this
    /// gain parameter is stored as linear gain while the values are displayed in decibels.
    /// The editor state, saved together with the parameter state so the custom scaling can be
    /// restored.
    #[persist = "editor-state"]
    editor_state: Arc<ViziaState>,

    #[id = "gain"]
    pub gain: FloatParam,

    #[id = "detector_size"]
    pub detector_size: IntParam,

    #[id = "overlap"]
    pub overlap: IntParam,
    // pub(crate) fn pitch(sample_rate: f32, signal: &Vec<f32>) -> [f32; 2] {
    // Include only notes that exceed a power threshold which relates to the
    // amplitude of frequencies in the signal. Use the suggested default
    // value of 5.0 from the library.
    #[id = "power_threshold"]
    pub power_threshold: FloatParam,
    // The clarity measure describes how coherent the sound of a note is. For
    // example, the background sound in a crowded room would typically be would
    // have low clarity and a ringing tuning fork would have high clarity.
    // This threshold is used to accept detect notes that are clear enough
    // (valid values are in the range 0-1).
    #[id = "clarity_threshold"]
    pub clarity_threshold: FloatParam,
    // https://github.com/alesgenova/pitch-detection/issues/23#issue-1354799855
    #[id = "pick_threshold"]
    pub pick_threshold: FloatParam,
    // the number of values to pick the median from
    #[id = "median_nr"]
    pub median_nr: IntParam,
    // TODO: use note names in addition to frequencies
    // TODO: add learn function?
    #[id = "min_pitch"]
    pub min_pitch: FloatParam,
    #[id = "max_pitch"]
    pub max_pitch: FloatParam,
    #[id = "hp_freq"]
    pub hp_freq: FloatParam,
    #[id = "lp_freq"]
    pub lp_freq: FloatParam,
    #[id = "ok_change"]
    pub ok_change: FloatParam,
    #[id = "max_change"]
    pub max_change: FloatParam,
    #[id = "change_compression"]
    pub change_compression: FloatParam,
    #[id = "latency"]
    pub latency: BoolParam,
}

impl Default for VoiceMaster {
    fn default() -> Self {
        Self {
            params: Arc::new(VoiceMasterParams::default()),

            peak_meter_decay_weight: 1.0,
            peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            sample_rate: 0.0,
            delay_line: vec![0.0; MAX_SIZE],
            signal: vec![0.0; MAX_SIZE],
            signal_index: 0,
            pitch_val: [-1.0, 0.0],
            pitches: vec![0.0; MAX_MEDIAN_NR],
            median_index: 0,
            previous_saw: 0.0,
            previous_pitch: -1.0,
            final_pitch: 0.0,
            // they wil get the real size later
            detectors: [
                McLeodDetector::new(2, 1),
                McLeodDetector::new(2, 1),
                McLeodDetector::new(2, 1),
                McLeodDetector::new(2, 1),
                McLeodDetector::new(2, 1),
                McLeodDetector::new(2, 1),
                McLeodDetector::new(2, 1),
                McLeodDetector::new(2, 1),
            ],
            pyin_exec: [
                PYINExecutor::new(750.0, 1350.0, 48000, 64, None, None, None),
                PYINExecutor::new(375.0, 1350.0, 48000, 128, None, None, None),
                PYINExecutor::new(187.5, 1350.0, 48000, 256, None, None, None),
                PYINExecutor::new(93.75, 1350.0, 48000, 512, None, None, None),
                PYINExecutor::new(46.875, 1350.0, 48000, 1024, None, None, None),
                PYINExecutor::new(23.4375, 1350.0, 48000, 2048, None, None, None),
                PYINExecutor::new(11.71875, 1350.0, 48000, 4096, None, None, None),
                PYINExecutor::new(5.859375, 1350.0, 48000, 8192, None, None, None),
            ],
            eq: Equalizer::new(48000.0),

            // let parameters = Parameters::default();

            // irapt:  Irapt::new(Parameters::default().clone()),
            // irapt:  Irapt { parameters: val, estimator: val, candidate_generator: val, candidate_selector: val },

            // irapt: Irapt::new(irapt::Parameters {
            // sample_rate: downsampled_rate,
            // pitch_range: PITCH_RANGE,
            // ..<_>::default()
            // })
            irapt: Irapt::new(Parameters::default().clone())
                .expect("the default parameters should be valid"),
            resampler: FftFixedInOut::<f32>::new(
                48000,
                DOWNSAMPLED_RATE,
                2048,
                1,

            ).unwrap(),
        }
    }
}

//at 48k, these freq ranges work:
// 107 - 114
// 173 - 236
// 291 - 292
// 352 - 471
// 536 - 536
// 716 - 1181  -> reports one octave too low
// 1331 - 1779  -> reports 1/3 of the actual freq
// 1927 - 3997  -> reports 1/4 of the actual freq
//
impl Default for VoiceMasterParams {
    fn default() -> Self {
        Self {
            // This gain is stored as linear gain. NIH-plug comes with useful conversion functions
            // to treat these kinds of parameters as if we were dealing with decibels. Storing this
            // as decibels is easier to work with, but requires a conversion for every sample.
            editor_state: editor::default_state(),

            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
                .with_smoother(SmoothingStyle::Logarithmic(50.0))
                .with_unit(" dB")
                .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
                .with_string_to_value(formatters::s2v_f32_gain_to_db()),

            detector_size: IntParam::new(
                "Detector Size",
                11,
                IntRange::Linear {
                    min: MIN_DETECTOR_SIZE_POWER as i32,
                    max: MAX_DETECTOR_SIZE_POWER as i32,
                },
            )
                .with_unit(" samples")
                .with_value_to_string(formatters::v2s_i32_power_of_two())
                .with_string_to_value(formatters::s2v_i32_power_of_two()),

            overlap: IntParam::new(
                "Overlap",
                1,
                IntRange::Linear {
                    min: 1,
                    max: MAX_OVERLAP as i32,
                },
            )
                .with_unit(" times/2048"),

            power_threshold: FloatParam::new(
                "Power Threshold",
                1.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 10.0,
                },
            ),
            clarity_threshold: FloatParam::new(
                "Clarity Threshold",
                0.55,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            pick_threshold: FloatParam::new(
                "Pick Threshold",
                0.999,
                FloatRange::Skewed {
                    min: 0.8,
                    max: 1.0,
                    factor: FloatRange::skew_factor(2.0),
                },
            ),
            median_nr: IntParam::new(
                "Median Number",
                MEDIAN_NR_DEFAULT,
                IntRange::Linear {
                    min: 1 as i32,
                    max: MAX_MEDIAN_NR as i32,
                },
            ),

            min_pitch: FloatParam::new(
                "Minimum Pitch",
                // E2, min male vocal pitch
                82.407,
                FloatRange::Skewed {
                    // A0:
                    min: 27.5,
                    // D5, min of picolo flute
                    max: 587.33,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
                .with_unit(" Hz"),
            max_pitch: FloatParam::new(
                "Maximum Pitch",
                // F6, max pitch of Freddy Mercury
                1396.91,
                FloatRange::Skewed {
                    // A2
                    min: 82.407,
                    // C8, max of picolo flute
                    max: 4186.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
                .with_unit(" Hz"),

            hp_freq: FloatParam::new(
                "High Pass Frequency",
                // E1, an octave below the min male vocal pitch
                41.203,
                FloatRange::Skewed {
                    // A0:
                    min: 10.0,
                    // D5, min of picolo flute
                    max: 587.33,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
                .with_unit(" Hz"),
            lp_freq: FloatParam::new(
                "Low Pass Frequency",
                // A4, max male vocal pitch
                4400.0,
                FloatRange::Skewed {
                    // A2
                    min: 82.407,
                    // 22k
                    max: 2.2e4,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
                .with_unit(" Hz"),

            ok_change: FloatParam::new(
                "OK Change Rate",
                // 0.001,
                2.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 2.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            ),
            max_change: FloatParam::new(
                "max Change Rate",
                2.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 2.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            ),
            change_compression: FloatParam::new(
                "Change Compression",
                // 0.042,
                1.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            ),
            latency: BoolParam::new("Latency", true),
        }
    }
}

impl Plugin for VoiceMaster {
    const NAME: &'static str = "Voice Master";
    const VENDOR: &'static str = "Bart Brouns";
    const URL: &'static str = "magnetophon.nl";
    const EMAIL: &'static str = "bart@magnetophon.nl";

    const VERSION: &'static str = "0.0.1";

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(3),
            ..AudioIOLayout::const_default()
        },
    ];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type BackgroundTask = ();
    type SysExMessage = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.peak_meter.clone(),
            self.params.editor_state.clone(),
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // After `PEAK_METER_DECAY_MS` milliseconds of pure silence, the peak meter's value should
        // have dropped by 12 dB
        self.peak_meter_decay_weight = 0.25f64
                                        .powf((buffer_config.sample_rate as f64 * PEAK_METER_DECAY_MS / 1000.0).recip())
                                        as f32;
        self.sample_rate = buffer_config.sample_rate;

        // create an EQ for a given sample rate
        self.eq = Equalizer::new(self.sample_rate as f32);

        // pYin: None to use default values
        let sr = self.sample_rate as u32; // sampling rate of audio data in Hz
        // let sr = (self.sample_rate / DOWNSAMPLE_RATIO as f32) as u32; // sampling rate of audio data in Hz
        let fmax = 1350.0f64; // maximum frequency in Hz
        // let frame_length = 2048usize; // frame length in samples
        // let pad_mode = PadMode::Constant(0.); // Zero-padding is applied on both sides of the signal. (only if cetner is true)

        for i in 0..NR_OF_DETECTORS {
            // let size = 2^i;
            let size = 2_usize.pow((i + MIN_DETECTOR_SIZE_POWER) as u32);
            let (win_length, hop_length, resolution) = (None, Some(size), None);
            let fmin = (sr / size as u32) as f64; // minimum frequency in Hz
            let padding = size / 2;
            self.detectors[i] = McLeodDetector::new(size, padding);
            // println!("i: {}, pow: {}, size: {}",i, i+MIN_DETECTOR_SIZE_POWER, size);
            // let min_period = ((sr as f64 / fmax).floor() as usize).max(1);
            // let max_period = ((sr as f64 / fmin).ceil() as usize).min(frame_length - win_length - 1);
            // if max_period - min_period < 2 {
            // panic!("min(ceil(sr / fmin), (frame_length - win_length - 1)) + 2 < floor(sr / fmax) should be satisfied!");

            // println!("fmin: {}, fmax: {}, sr: {}",fmin, fmax, sr);
            self.pyin_exec[i] =
                PYINExecutor::new(fmin, fmax, sr, size, win_length, hop_length, resolution);
            // PYINExecutor::new(60.0, 6000.0, sr, size, win_length, hop_length, resolution);

            let _parameters = Parameters::default();
            self.irapt =
            // Irapt::new(parameters.clone()).expect("the default parameters should be valid");
                Irapt::new(irapt::Parameters {
                    // sample_rate: 48000.0,
                    sample_rate: DOWNSAMPLED_RATE as f64,
                    // pitch_range: PITCH_RANGE,
                    ..<_>::default()
                }).expect("the default parameters should be valid");
            self.resampler = FftFixedInOut::<f32>::new(
                self.sample_rate as usize,
                DOWNSAMPLED_RATE,
                2048,
                1,

            ).unwrap();
        }

        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let mut channel_counter = 0;
        let size = 2_usize.pow((self.params.detector_size.value() as usize) as u32);
        let overlap = self.params.overlap.value() as usize * size / 2048;

        // set the latency, cannot do that from a callback
        // if self.params.latency.value() {
        // context.set_latency_samples(size as u32);
        // } else {
        // context.set_latency_samples(0);
        // }

        // if there is a new median_nr value
        if self.pitches.len() != (self.params.median_nr.value() as usize) {
            // resize the pitches vector
            self.pitches
                .resize(self.params.median_nr.value() as usize, 440.0);
            // reset the median index
            self.median_index = self.median_index % self.params.median_nr.value() as usize;
        }

        // set the filter frequencies
        // self.eq
        // .set(0, Curve::Highpass, self.params.hp_freq.value(), 1.0, 0.0);
        // self.eq
        // .set(1, Curve::Lowpass, self.params.lp_freq.value(), 1.0, 0.0);

        for channel_samples in buffer.iter_samples() {
            let mut amplitude = 0.0;
            let num_samples = channel_samples.len();

            let gain = self.params.gain.smoothed.next();

            for sample in channel_samples {
                // which of the 3 channels are we on
                match channel_counter {
                    // audio:
                    0 => {
                        *sample *= gain;
                        amplitude += *sample;

                        // fill the delay line
                        self.delay_line[self.signal_index] = *sample;
                        // apply the filters to a copy of the sample
                        // we don't want to filter the main audio output, just the pitch detector input
                        // let mut sample_filtered = *sample;
                        // let mut sample_filtered = self.eq.process(*sample);

                        // let mut downsampling_index = self.signal_index % DOWNSAMPLE_RATIO;
                        // copy our filtered sample to signal
                        // if downsampling_index == 0 {
                        self.signal[self.signal_index] = *sample;
                        // self.signal[self.signal_index] = sample_filtered as f32;
                        // }
                        // if the user chooses to sync up the audio with the pitch
                        // if self.params.latency.value() {
                        //     // delay our sample
                        //     *sample = self.delay_line[(signal_index - size) % MAX_SIZE];
                        // }

                        // update the index
                        self.signal_index = (self.signal_index + 1) % MAX_SIZE;


                        // do overlap nr of times:
                        for i in 0..overlap {
                            // if index[i] == 0
                            // so IOW: when the buffer is full
                            if staggered_index(i, self.signal_index, size, overlap) == 0
                            // && (downsampling_index == 0)
                            {
                                // [ &v[..3], &v[l - 3..]].concat()
                                // check if the end wraps around:
                                let index_plus_size = (self.signal_index + size) % MAX_SIZE;
                                let mut slice = vec![0.0; MAX_SIZE];
                                // if no wrap around:
                                if (self.signal_index) < index_plus_size {
                                    slice =
                                        self.signal[self.signal_index..index_plus_size].to_vec();
                                    // if we do have a wrap around:
                                } else {
                                    slice = [
                                        &self.signal[self.signal_index..],
                                        &self.signal[..index_plus_size],
                                    ]
                                        .concat()
                                        .to_vec();
                                };

                                // resample:
                                // let resampled = self.resampler.process(&vec![slice;1],None).unwrap()[0].clone();
                                // let mut sample_buffer = VecDeque::from(resampled);
                                // don't resample:
                                // let mut sample_buffer = VecDeque::from(slice);


                                // call the pitchtracker
                                self.pitch_val = mc_pitch::pitch(
                                    self.sample_rate,
                                    &slice,
                                    &mut self.detectors[(self.params.detector_size.value()
                                                         as usize
                                                         - MIN_DETECTOR_SIZE_POWER)],
                                    self.params.power_threshold.value(),
                                    // clarity_threshold: use 0.0, so all pitch values are let trough
                                    0.0,
                                    self.params.pick_threshold.value(),
                                );
                                // let wav: CowArray<f64, Ix1> = ...;
                                // let _fill_unvoiced = 0.0f32;
                                // let framing: Framing<f32> = Framing::Valid;
                                // let _framing: Framing<f32> = Framing::Center(Constant(0.0));
                                // let center = true; // If true, the first sample in wav becomes the center of the first frame.
                                // let pad_mode = PadMode::Constant(0.); // Zero-padding is applied on both sides of the signal. (only if cetner is true)
                                // let array = CowArray::from(Array::from_vec(slice));
                                // let mut sample_buffer = VecDeque::from(slice);

                                // f0 (Array1<f64>) contains the pitch estimate in Hz. (NAN if unvoiced)
                                // voiced_flag (Array1<bool>) contains whether the frame is voiced or not.
                                // voiced_prob (Array1<f64>) contains the probability of the frame is voiced.

                                // let (f0, voiced_flag, voiced_prob) =
                                // self.pyin_exec[(self.params.detector_size.value() as usize
                                // - MIN_DETECTOR_SIZE_POWER)]
                                // .pyin(array, fill_unvoiced, framing);

                                // call the pitchtracker
                                let (hz, amplitude) = pitch::detect(&slice.as_slice().iter().map(|&x| x as f64).collect::<Vec<f64>>());


                                // if clarity is high enough
                                if self.pitch_val[1] > self.params.clarity_threshold.value()
                                // and the pitch isn't too low or too high
                                    && self.pitch_val[0] > self.params.min_pitch.value()
                                    && self.pitch_val[0] < self.params.max_pitch.value()
                                    && (hz as f32) > self.params.min_pitch.value()
                                    && (hz as f32) < self.params.max_pitch.value()
                                {
                                    let mut diff = 0.0;
                                    if (hz as f32) < self.pitch_val[0]
                                    {
                                        diff = (1.0-(hz as f32/self.pitch_val[0])).abs();
                                    }
                                    else
                                    {
                                        diff = (1.0-(self.pitch_val[0]/hz as f32)).abs();
                                    }
                                    if diff < self.params.change_compression.value()
                                    // if (1.0-(hz as f32/self.pitch_val[0])).abs() < 0.79
                                    {
                                        self.final_pitch = self.pitch_val[0];
                                    }
                                    else
                                    {
                                        println!(
                                            "mc_pitch: {}, hz: {}, diff: {}, change {}",
                                            self.pitch_val[0],
                                            hz,
                                            diff,
                                            self.params.change_compression.value()
                                        );
                                    };
                                };

                                // let mut sample_index = 0;
                                // while let (initial_sample_buffer_len, Some(output)) =
                                // (size, self.irapt.process(&mut sample_buffer))
                                // {
                                // let estimated_pitch =
                                // output.pitch_estimates().final_estimate();
                                // if estimated_pitch.energy as f32
                                // > self.params.clarity_threshold.value()
                                // {
                                // let estimated_pitch_index = (sample_index as isize
                                // + estimated_pitch.offset)
                                // as usize;
                                // let estimated_pitch_time =
                                // estimated_pitch_index as f32 / self.sample_rate;
                                // println!(
                                // "estimated pitch at {:0.3}: {}Hz with energy {}",
                                // estimated_pitch_time,
                                // estimated_pitch.frequency,
                                // estimated_pitch.energy
                                // );
                                // sample_index +=
                                // initial_sample_buffer_len - sample_buffer.len();
                                // self.pitch_val = [
                                // estimated_pitch.frequency as f32,
                                // estimated_pitch.energy as f32,
                                // ];
                                // }
                                // }

                                // if
                                // voiced_prob.to_vec()[0] > 0.0
                                        // && f0.to_vec()[0] > (self.sample_rate/size as f32)
                                        // {
                                        // println!(
                                        // "clarity: {},   pitch: {}, voiced_flag: {}",
                                        // voiced_prob.to_vec()[0],
                                        // f0.to_vec()[0],
                                        // voiced_flag.to_vec()[0]
                                        // );
                                        // };

                                        // if voiced_prob.to_vec()[0] > 0.0
                                        // && f0.to_vec()[0] > (self.sample_rate/size as f32)
                                        // {
                                        // self.pitch_val = [f0.to_vec()[0], voiced_prob.to_vec()[0]];
                                        // }
                                        // println!("clarity: {},   pitch: {}", self.pitch_val[0],self.pitch_val[1]);
                                        //-> (Array1<A>, Array1<bool>, Array1<A>)
                                        // if clarity is high enough
                                        if self.pitch_val[1] > self.params.clarity_threshold.value()
                                        // and the pitch isn't too low or too high
                                    && self.pitch_val[0] > self.params.min_pitch.value()
                                    && self.pitch_val[0] < self.params.max_pitch.value()
                                {
                                    // let ratio = self.previous_pitch / self.pitch_val[0];
                                    // let change = (ratio - 1.0).abs();
                                    // let prev_change =
                                    // ((self.pitches[self.median_index] / self.pitch_val[0]) - 1.0).abs();
                                    // let sign = if ratio > 1.0 { 1.0 } else { -1.0 };
                                    // let sign = ratio > 1.0;
                                    // let sp = ((change - self.params.ok_change.value())
                                    // * self.params.change_compression.value() as f32)
                                    // + self.params.ok_change.value();
                                    // let ratioo = if sign {
                                    // 1.0 + sp
                                    // (1.0 + sp).min(self.params.max_change.value())
                                    // } else {
                                    // 1.0 - sp
                                    // (1.0 - sp).max(0.0-self.params.max_change.value())
                                    // };

                                    // if change > self.params.ok_change.value() {
                                    // update the pitches

                                    // self.pitches[self.median_index] =
                                    // (ratioo) * self.pitch_val[0];
                                    // self.previous_pitch / ratioo;
                                    // update the ringbuf pointer
                                    // self.median_index = (self.median_index + 1)
                                    // % (self.params.median_nr.value() as usize);
                                    // self.previous_pitch = self.pitch_val[0];
                                    // if (ratio - ratioo).abs() > 0.05 {
                                    // println!(
                                        // "ratio: {} change: {} change-ok: {} sign: {} sp: {} ratioo: {}",
                                        // ratio,
                                        // change,
                                        // change - self.params.ok_change.value()
                                        // ,
                                        // sign,
                                        // sp,
                                        // ratioo,
                                        // );
                                    };
                                // } else {
                                // update the pitches
                                // self.pitches[self.median_index] = self.pitch_val[0];
                                // update the ringbuf pointer
                                // self.median_index = (self.median_index + 1)
                                // % (self.params.median_nr.value() as usize);
                                // nih_trace!(
                                // "i: {}, Frequency: {}, Clarity: {}",
                                    // i, self.pitch_val[0], self.pitch_val[1]
                                    // );
                                // };
                                // self.previous_pitch = self.pitches[self.median_index];
                            }
                            // get the median pitch:
                            // copy the pitches, we don't want to sort the ringbuffer
                            // let mut sorted: Vec<f32> = self.pitches.clone();
                            // sort the copy
                            // sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            // get the middle one
                            // self.final_pitch = sorted[sorted.len() / 2];
                            // self.final_pitch = self.pitches.iter().sum::<f32>()
                            // / self.params.median_nr.value() as f32;
                            // self.final_pitch = self.pitches[self.median_index];
                            // self.previous_pitch = self.final_pitch;
                            // nih_trace!("pitch: {}", self.final_pitch);
                        }
                        // }
                    }
                    // positive saw at 1/4 freq, see https://github.com/magnetophon/VoiceOfFaust/blob/V1.1.4/lib/master.lib#L8
                    1 => {
                        *sample = self.previous_saw + (self.final_pitch / (self.sample_rate * 4.0));
                        *sample -= (*sample).floor();
                        self.previous_saw = *sample
                    }
                    // clarity:
                    2 => *sample = self.pitch_val[1],
                    // never happens:
                    _ => panic!("Why are we here?"),
                }
                // next channel
                const NR_OUTPUT_CHANNELS : i32 = 3;
                channel_counter = (channel_counter + 1) % NR_OUTPUT_CHANNELS;
            }

            // To save resources, a plugin can (and probably should!) only perform expensive
            // calculations that are only displayed on the GUI while the GUI is open
            if self.params.editor_state.is_open() {
                amplitude = (amplitude / num_samples as f32).abs();
                let current_peak_meter = self.peak_meter.load(std::sync::atomic::Ordering::Relaxed);
                let new_peak_meter = if amplitude > current_peak_meter {
                    amplitude
                } else {
                    current_peak_meter * self.peak_meter_decay_weight
                        + amplitude * (1.0 - self.peak_meter_decay_weight)
                };

                self.peak_meter
                    .store(new_peak_meter, std::sync::atomic::Ordering::Relaxed)
            }
        }

        fn staggered_index(i: usize, index: usize, size: usize, overlap: usize) -> usize {
            (index + (i * (size / overlap))) % size
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for VoiceMaster {
    const CLAP_ID: &'static str = "com.magnetophon.nl.voice_master";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("A master plugin for VoiceOfFaust");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for VoiceMaster {
    const VST3_CLASS_ID: [u8; 16] = *b"VoiceMaster_____";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Analyzer, Vst3SubCategory::Tools];
}

// nih_export_clap!(VoiceMaster);
// nih_export_vst3!(VoiceMaster);
