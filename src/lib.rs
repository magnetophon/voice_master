use atomic_float::AtomicF32;

use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use simple_eq::design::Curve;
use simple_eq::Equalizer;
use std::sync::Arc;

use bit_mask_ring_buf::BMRingBuf;
use pitch::detect;
use pitch::BitStream;
use pitch_detection::detector::mcleod::McLeodDetector;

mod editor;
mod mc_pitch;

// TODO:
//
//
// https://github.com/antoineschmitt/dywapitchtrack/blob/ff251761c6cfb1e6b9ffd9ffefacc7709aff508a/src/dywapitchtrack.c#L343C1-L355C5
//
// ***********************************
// the dynamic postprocess
// ***********************************

// It states:
// - a pitch cannot change much all of a sudden (20%) (impossible humanly,
// so if such a situation happens, consider that it is a mistake and drop it.
// - a pitch cannot double or be divided by 2 all of a sudden : it is an
// algorithm side-effect : divide it or double it by 2.
// - a lonely voiced pitch cannot happen, nor can a sudden drop in the middle
// of a voiced segment. Smooth the plot.
//
//
// compare pitch trackers:
//
// https://github.com/sevagh/pitch-detection/tree/master/misc/mcleod
//
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
//   - use midi input as an offset
//   - use midi input as an offset, polyphonic -> Jacob Colier
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
/// 2^7 = 128 samples
/// 44100/128 = 345 Hz at a samplerate of 44.1k
/// for when you want to track your picolo flute with really low latency!
const MIN_DETECTOR_SIZE_POWER: usize = 7;
// const MIN_DETECTOR_SIZE_POWER: usize = 11;
/// 2^13 = 8192
/// 192000/8192 = 23.4Hz at a samplerate of 192k
/// for when you want to play 6 string bassguitar at 192k
const MAX_DETECTOR_SIZE_POWER: usize = 13;
// const MAX_DETECTOR_SIZE_POWER: usize = 11;
/// the number of detectors we need, one for each size
const NR_OF_DETECTORS: usize = MAX_DETECTOR_SIZE_POWER - MIN_DETECTOR_SIZE_POWER + 1;
/// the maximum size of the detectors
const MAX_SIZE: isize = 2_isize.pow(MAX_DETECTOR_SIZE_POWER as u32);
/// the maximum nr of times the detector is updated each 2048 samples
const MAX_OVERLAP: usize = 2048;
/// default samplerate
const DEFAULT_SAMPLERATE: f32 = 48000.0;

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
    delay_line: BMRingBuf<f32>,
    /// the signal to be used in the pitchtrackers
    signal: BMRingBuf<f32>,
    /// the sample index of the above signal
    signal_index: isize,
    /// ovelapping segments of the signal, to feed the pitchtrackers
    // overlap_signal: BMRingBuf<f32>,
    overlap_signal: Vec<f32>,
    // bitstream for pitch calculation
    bin: BitStream,
    /// the curent pitch and clarity
    pitch_val: [f32; 2],
    /// previous value of the output saw, to calculate the new one from
    previous_saw: f32,
    /// the final pitch that we are using
    final_pitch: f32,
    /// an array of pitch detectors, one for each size:
    detectors: [McLeodDetector<f32>; NR_OF_DETECTORS],
    eq: Equalizer<f32>,
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
    #[id = "max_diff"]
    pub max_diff: FloatParam,
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
            delay_line: BMRingBuf::<f32>::from_len(MAX_SIZE as usize),
            signal: BMRingBuf::<f32>::from_len(MAX_SIZE as usize),
            signal_index: 0,
            // overlap_signal: BMRingBuf::<f32>::from_len(MAX_SIZE),
            overlap_signal: vec![0.0; MAX_SIZE as usize],
            bin: BitStream::new(&vec![0.0; MAX_SIZE as usize], 0.0),
            pitch_val: [-1.0, 0.0],
            previous_saw: 0.0,
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
            ],
            eq: Equalizer::new(DEFAULT_SAMPLERATE),
        }
    }
}

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
                "Samples between pitch values",
                13,
                IntRange::Linear {
                    min: 1,
                    max: MAX_OVERLAP as i32,
                },
            )
            .with_unit(" samples"),

            power_threshold: FloatParam::new(
                "Power Threshold",
                0.13,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 10.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            ),
            clarity_threshold: FloatParam::new(
                "Clarity Threshold",
                0.7,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            pick_threshold: FloatParam::new(
                "Pick Threshold",
                0.98,
                FloatRange::Skewed {
                    min: 0.8,
                    max: 1.0,
                    factor: FloatRange::skew_factor(2.0),
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
            max_diff: FloatParam::new(
                "Maximum Difference",
                0.13,
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

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(1),
        main_output_channels: NonZeroU32::new(3),
        ..AudioIOLayout::const_default()
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type BackgroundTask = ();
    type SysExMessage = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(& mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
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

        for i in 0..NR_OF_DETECTORS {
            // let size = 2^i;
            let size = 2_usize.pow((i + MIN_DETECTOR_SIZE_POWER) as u32);
            let padding = size / 2;
            self.detectors[i] = McLeodDetector::new(size, padding);
        }

        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let mut channel_counter = 0;
        let size = 2_isize.pow((self.params.detector_size.value() as isize) as u32);
        let overlap = self.params.overlap.value() as usize;


        // set the latency, cannot do that from a callback
        if self.params.latency.value() {
            context.set_latency_samples(size as u32);
        } else {
            context.set_latency_samples(0);
        }
        // make sure the index doesn't get too big
        self.signal_index %= MAX_SIZE;

        // set the filter frequencies
        self.eq
            .set(0, Curve::Highpass, self.params.hp_freq.value(), 1.0, 0.0);
        self.eq
            .set(1, Curve::Lowpass, self.params.lp_freq.value(), 1.0, 0.0);

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
                        // let sample_filtered = *sample;
                        let sample_filtered = self.eq.process(*sample);

                        // let mut downsampling_index = self.signal_index % DOWNSAMPLE_RATIO;
                        // if downsampling_index == 0 {

                        // copy our filtered sample to signal
                        // self.signal[self.signal_index] = *sample;
                        self.signal[self.signal_index] = sample_filtered as f32;
                        // }
                        // if the user chooses to sync up the audio with the pitch
                        if self.params.latency.value() {
                            // if self.signal_index >= size {
                                //     // delay our sample
                                *sample =
                                    self.delay_line[self.signal_index - size];
                            // }
                        }

                        // update the index
                        // self.signal_index = (self.signal_index + 1) % MAX_SIZE as isize;
                        self.signal_index += 1;

                        if self.signal_index as usize % overlap == 0
                        // && (downsampling_index == 0)
                        {
                            // let index_plus_size =
                            // (self.signal_index + size) % MAX_SIZE as isize;
                            let slices =
                                self.signal.as_slices_len(self.signal_index, size as usize);
                            self.overlap_signal.clear();
                            self.overlap_signal.extend_from_slice(&slices.0);
                            // if wrap around:
                            // if (self.signal_index) >= index_plus_size {
                            self.overlap_signal.extend_from_slice(&slices.1);
                            // };

                            // call the pitchtracker
                            let detector = &mut self.detectors[self.params.detector_size.value()
                                                               as usize
                                                               - MIN_DETECTOR_SIZE_POWER];
                            self.pitch_val = mc_pitch::pitch(
                                self.sample_rate,
                                &self.overlap_signal,
                                // &resampled.clone(),
                                detector,
                                self.params.power_threshold.value(),
                                // clarity_threshold: use 0.0, so all pitch & clarity values are let trough
                                // we filter the pitches downstream but let all clarity values trough,
                                // for usage in the faust de-esser and re-esser
                                0.0,
                                self.params.pick_threshold.value(),
                            );

                            // call the other pitchtracker
                            let (hz_raw, _amplitude) = detect(
                                // &self.overlap_signal .as_slice().iter().map(|&x| x as f64).collect::<Vec<f64>>()
                                &self.overlap_signal,
                                &mut self.bin,
                            );
                            let hz = hz_raw * DEFAULT_SAMPLERATE / self.sample_rate;
                            // let hz = self.pitch_val[0];

                            // println!("hz: {}",hz);

                            // if clarity is high enough
                            if self.pitch_val[1] > self.params.clarity_threshold.value()
                            // and the pitch isn't too low or too high
                                && self.pitch_val[0] > self.params.min_pitch.value()
                                && self.pitch_val[0] < self.params.max_pitch.value()
                                && (hz as f32) > self.params.min_pitch.value()
                                && (hz as f32) < self.params.max_pitch.value()
                            {
                                let diff: f32 = if (hz as f32) < self.pitch_val[0] {
                                    (1.0 - (hz as f32 / self.pitch_val[0])).abs()
                                } else {
                                    (1.0 - (self.pitch_val[0] / hz as f32)).abs()
                                };
                                if diff < self.params.max_diff.value() {
                                    self.final_pitch = self.pitch_val[0];
                                }
                                // else {
                                // println!(
                                // "mc_pitch: {}, hz: {}, diff: {}, change {}",
                                // self.pitch_val[0],
                                // hz,
                                // diff,
                                // self.params.max_diff.value()
                                // );
                                // };
                            };
                        }
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
                const NR_OUTPUT_CHANNELS: i32 = 3;
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
