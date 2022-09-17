use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use pitch_detection::detector::mcleod::McLeodDetector;
use std::sync::Arc;
use simple_eq::*;
use simple_eq::design::Curve;



mod editor;
mod pitch;

/// The time it takes for the peak meter to decay by 12 dB after switching to complete silence.
const PEAK_METER_DECAY_MS: f64 = 150.0;

/// Blocksize of the detector, determines the lowest pitch that can be detected at a given samplerate.
// 2^6 = 64 samples
// 44100/64 = 689 Hz at a samplerate of 44.1k
// for when you want to track your picolo flute with really low latency!
const MIN_DETECTOR_SIZE_POWER: usize = 6;
// 2^13 = 8192
// 192000/8192 = 23.4Hz at a samplerate of 192k
// for when you want to play 6 string bassguitar at 192k
const MAX_DETECTOR_SIZE_POWER: usize = 13;
/// the number of detectors we need, one for each size
const NR_OF_DETECTORS: usize = MAX_DETECTOR_SIZE_POWER - MIN_DETECTOR_SIZE_POWER + 1;
const MAX_SIZE: usize = 2_usize.pow(MAX_DETECTOR_SIZE_POWER as u32);
/// the nr of times the detector is updated each DETECTOR_SIZE samples
// TODO: make variable?
const OVERLAP: usize = 32;
/// The median is taken from at max this nr of pitches
const MAX_MEDIAN_NR: usize = OVERLAP;
const MEDIAN_NR_DEFAULT: i32 = 1;

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
    /// an array of signals to be used in the pitchtrackers
    signals: [Vec<f32>; OVERLAP],
    /// the sample index of the above signals
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
    // detectors: [McLeodDetector<f32>;7],
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
            delay_line: vec![0.0;MAX_SIZE],
            signals: Default::default(),
            signal_index: 0,
            pitch_val: [-1.0, 0.0],
            pitches: Default::default(),
            median_index: 0,
            previous_saw: 0.0,
            previous_pitch: -1.0,
            final_pitch: 0.0,
            // they wil get the real size later
            // detectors: [McLeodDetector::new(2, 1);NR_OF_DETECTORS],
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
            eq: Equalizer::new(48000.0),
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
                    min: 0.98,
                    max: 1.0,
                    factor: FloatRange::skew_factor(3.7),
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
                // A2, min male vocal pitch
                82.407,
                FloatRange::Skewed {
                    // A0:
                    min: 27.5,
                    // D5, min of picolo flute
                    max: 587.33,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            // .with_callback({
            // self.updateHP = true;
            // })
                .with_unit(" Hz"),
            max_pitch: FloatParam::new(
                "Maximum Pitch",
                // A4, max male vocal pitch
                440.0,
                FloatRange::Skewed {
                    // A2
                    min: 82.407,
                    // C8, max of picolo flute
                    max: 4186.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            // .with_callback({
            // self.updateLP = true;
            // })
                .with_unit(" Hz"),

            hp_freq: FloatParam::new(
                "High Pass Frequency",

                // A2, min male vocal pitch
                82.407,
                FloatRange::Skewed {
                    // A0:
                    min: 0.1,
                    // D5, min of picolo flute
                    max: 587.33,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            // .with_callback({
            // self.updateHP = true;
            // })
                .with_smoother(SmoothingStyle::Logarithmic(50.0))
                .with_unit(" Hz"),
            lp_freq: FloatParam::new(
                "Low Pass Frequency",
                // A4, max male vocal pitch
                4400.0,
                FloatRange::Skewed {
                    // A2
                    min: 82.407,
                    // C8, max of picolo flute
                    max: 2.2e4,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            // .with_callback({
            // self.updateLP = true;
            // })
                .with_smoother(SmoothingStyle::Logarithmic(50.0))
                .with_unit(" Hz"),

            ok_change: FloatParam::new(
                "OK Change Rate",
                // 0.001,
                1.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            ),
            max_change: FloatParam::new(
                "max Change Rate",
                1.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            ),
            change_compression: FloatParam::new(
                "Change Compression",
                0.042,
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

    const DEFAULT_INPUT_CHANNELS: u32 = 1;
    const DEFAULT_OUTPUT_CHANNELS: u32 = 3;

    const DEFAULT_AUX_INPUTS: Option<AuxiliaryIOConfig> = None;
    const DEFAULT_AUX_OUTPUTS: Option<AuxiliaryIOConfig> = None;

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&self) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.peak_meter.clone(),
            self.params.editor_state.clone(),
        )
    }

    fn accepts_bus_config(&self, config: &BusConfig) -> bool {
        // This works with any symmetrical IO layout
        config.num_input_channels == Self::DEFAULT_INPUT_CHANNELS
            && config.num_output_channels == Self::DEFAULT_OUTPUT_CHANNELS
    }

    fn initialize(
        &mut self,
        _bus_config: &BusConfig,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext,
    ) -> bool {
        // After `PEAK_METER_DECAY_MS` milliseconds of pure silence, the peak meter's value should
        // have dropped by 12 dB
        self.peak_meter_decay_weight = 0.25f64
            .powf((buffer_config.sample_rate as f64 * PEAK_METER_DECAY_MS / 1000.0).recip())
            as f32;
        self.sample_rate = buffer_config.sample_rate;

        // create an EQ for a given sample rate
        self.eq = Equalizer::new(self.sample_rate as f32);


        // init all signals with the max size
        for i in 0..OVERLAP {
            self.signals[i].resize(MAX_SIZE, 0.0);
        }
        for i in 0..NR_OF_DETECTORS {
            // let size = 2^i;
            let size = 2_usize.pow((i + MIN_DETECTOR_SIZE_POWER) as u32);
            let padding = size / 2;
            self.detectors[i] = McLeodDetector::new(size, padding);
            // println!("i: {}, pow: {}, size: {}",i, i+MIN_DETECTOR_SIZE_POWER, size)
        }
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext,
    ) -> ProcessStatus {
        let mut channel_counter = 0;
        let size = 2_usize.pow((self.params.detector_size.value() as usize) as u32);
        // if new detector size
        if self.signals[0].len() != size {
            // resize the detector input buffers
            for i in 0..OVERLAP {
                self.signals[i].resize(size as usize, 0.0);
            }
        }

        // set the latency, cannot do that fro a callback
        if self.params.latency.value() {
            context.set_latency_samples(size as u32);
        } else {
            context.set_latency_samples(0);
        }

        // if there is a new median_nr value
        if self.pitches.len() != (self.params.median_nr.value() as usize) {
            // resize the pitches vector
            self.pitches
                .resize(self.params.median_nr.value() as usize, 440.0);
            // reset the median index
            self.median_index = self.median_index % self.params.median_nr.value() as usize;
        }

        let len = self.signals[0].len();



        for channel_samples in buffer.iter_samples() {
            let mut amplitude = 0.0;
            let num_samples = channel_samples.len();

            let gain = self.params.gain.smoothed.next();
            // set the filter frequencies
            self.eq.set(0, Curve::Highpass, self.params.hp_freq.smoothed.next(), 1.0, 0.0);
            self.eq.set(1, Curve::Lowpass, self.params.lp_freq.smoothed.next(), 1.0, 0.0);
            // self.eq.set(0, Curve::Highpass, 220.0, 1.0, 0.0);
            // self.eq.set(1, Curve::Lowpass, 440.0, 1.0, 0.0);

            for sample in channel_samples {
                // which of the 3 channels are we on
                match channel_counter {
                    // audio:
                    0 => {
                        let signal_index = self.signal_index % len;
                        *sample *= gain;
                        amplitude += *sample;

                        // fill the delay line
                        self.delay_line[signal_index] = *sample;
                        // apply the filters to a copy of the sample
                        // we don't want to filter the main audio output, just the pitch detector input
                        let mut sample_filtered = *sample;
                        sample_filtered = self.eq.process(sample_filtered);

                        // copy our filtered sample to signal
                        for i in 0..OVERLAP {
                            self.signals[i][staggered_index(i, signal_index, len)] =
                                sample_filtered as f32;
                        }
                        // if the user chooses to sync up the audio with the pitch
                        if self.params.latency.value() {
                            // delay our sample
                            *sample = self.delay_line[(signal_index + 1) % len];
                        }

                        // update the index
                        self.signal_index += 1;
                        if self.signal_index == len {
                            self.signal_index = 0;
                        };

                        // do OVERLAP nr of times:
                        for i in 0..OVERLAP {
                            // if index[i] == 0
                            // so IOW: when the buffer is full
                            if staggered_index(i, self.signal_index, len) == 0 {
                                // call the pitchtracker
                                self.pitch_val = pitch::pitch(
                                    self.sample_rate,
                                    &self.signals[i],
                                    &mut self.detectors[(self.params.detector_size.value()
                                                         as usize
                                                         - MIN_DETECTOR_SIZE_POWER)],
                                    self.params.power_threshold.value(),
                                    // clarity_threshold: use 0.0, so all pitch values are let trough
                                    0.0,
                                    self.params.pick_threshold.value(),
                                );
                                // if clarity is high enough
                                if self.pitch_val[1] > self.params.clarity_threshold.value()
                                // and the pitch isn't too low or too high
                                    && self.pitch_val[0] > self.params.min_pitch.value()
                                    && self.pitch_val[0] < self.params.max_pitch.value()
                                {
                                    let ratio = self.previous_pitch / self.pitch_val[0];
                                    let change = (ratio - 1.0).abs();
                                    // let prev_change =
                                    // ((self.pitches[self.median_index] / self.pitch_val[0]) - 1.0).abs();
                                    // let sign = if ratio > 1.0 { 1.0 } else { -1.0 };
                                    let sign = ratio > 1.0;
                                    let sp = ((change - self.params.ok_change.value())
                                              * self.params.change_compression.value() as f32)
                                        + self.params.ok_change.value();
                                    let ratioo = if sign {
                                        1.0 + sp
                                        // (1.0 + sp).min(self.params.max_change.value())
                                        } else {
                                        1.0 - sp
                                        // (1.0 - sp).max(0.0-self.params.max_change.value())
                                        };


                                    if change > self.params.ok_change.value() {
                                        // update the pitches

                                        self.pitches[self.median_index] =
                                        // (ratioo) * self.pitch_val[0];
                                            self.previous_pitch / ratioo;
                                        // update the ringbuf pointer
                                        self.median_index = (self.median_index + 1)
                                            % (self.params.median_nr.value() as usize);
                                        // self.previous_pitch = self.pitch_val[0];
                                        if (ratio - ratioo).abs() > 0.05 {
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

                                    } else {
                                        // update the pitches
                                        self.pitches[self.median_index] = self.pitch_val[0];
                                        // update the ringbuf pointer
                                        self.median_index = (self.median_index + 1)
                                            % (self.params.median_nr.value() as usize);
                                        // nih_trace!(
                                        // "i: {}, Frequency: {}, Clarity: {}",
                                        // i, self.pitch_val[0], self.pitch_val[1]
                                        // );
                                    };
                                    // self.previous_pitch = self.pitches[self.median_index];
                                }
                                // get the median pitch:
                                // copy the pitches, we don't want to sort the ringbuffer
                                let mut sorted: Vec<f32> = self.pitches.clone();
                                // sort the copy
                                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                // get the middle one
                                // self.final_pitch = sorted[sorted.len() / 2];
                                self.final_pitch = self.pitches.iter().sum::<f32>()/self.params.median_nr.value() as f32;
                                // self.final_pitch = self.pitches[self.median_index];
                                self.previous_pitch = self.final_pitch;
                                // nih_trace!("pitch: {}", self.final_pitch);
                            }
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
                    _ => nih_trace!("Why are we here?"),
                }
                // next channel
                channel_counter = (channel_counter + 1) % Self::DEFAULT_OUTPUT_CHANNELS;
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

        fn staggered_index(i: usize, index: usize, len: usize) -> usize {
            (index + (i * (len / OVERLAP))) % len
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
    const VST3_CATEGORIES: &'static str = "Fx|Dynamics";
}

// nih_export_clap!(VoiceMaster);
// nih_export_vst3!(VoiceMaster);
