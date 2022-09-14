use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use pitch_detection::detector::mcleod::McLeodDetector;
use std::sync::Arc;

mod editor;
mod pitch;

/// The time it takes for the peak meter to decay by 12 dB after switching to complete silence.
const PEAK_METER_DECAY_MS: f64 = 150.0;

/// Blocksize of the detector, determines the lowest pitch that can be detected at a given samplerate.
// TODO: make variable
// TODO: make switchable delay and latency compensation
// 2^6 = 64 samples
// 44100/64 = 689 Hz at a samplerate of 44.1k
// for when you want to track your picolo flute with really low latency!
const MIN_DETECTOR_SIZE_POWER: usize = 6;
// 2^13 = 8192
// 192000/8192 = 23.4Hz at a samplerate of 192k
// for when you want to play 6 string bassguitar at 192k
// const MAX_DETECTOR_SIZE_POWER: usize = 13;
const MAX_DETECTOR_SIZE_POWER: usize = 13;
const NR_OF_DETECTORS:usize = MAX_DETECTOR_SIZE_POWER-MIN_DETECTOR_SIZE_POWER+1;
/// the nr of times the detector is updated each DETECTOR_SIZE samples
// TODO: make variable?
const OVERLAP: usize = 32;
/// The median is taken from this nr of pitches
// TODO: make variable
const MEDIAN_NR: usize = 31;

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
    /// an array of signals to be used in the pitchtrackers
    signals: [Vec<f32>; OVERLAP],
    /// the sample index of the above signals
    signal_index: usize,
    /// the curent pitch and clarity
    pitch_val: [f32; 2],
    /// array of pitches to pick the median from
    pitches: [f32; MEDIAN_NR],
    /// index into "pitches", to use it as a ringbuffer
    median_index: usize,
    /// previous value of the output saw, to calculate the new one from
    previous_saw: f32,
    /// the final pitch that we are using
    final_pitch: f32,
    /// an array of pitch detectors, one for each size:
    detectors: [McLeodDetector<f32>;NR_OF_DETECTORS],
    // detectors: [McLeodDetector<f32>;7],
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
    // TODO: add learn function?
    // TODO: use note names?
    #[id = "min_pitch"]
    pub min_pitch: FloatParam,
    #[id = "max_pitch"]
    pub max_pitch: FloatParam,
}

impl Default for VoiceMaster {
    fn default() -> Self {
        Self {
            params: Arc::new(VoiceMasterParams::default()),

            peak_meter_decay_weight: 1.0,
            peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            sample_rate: 0.0,
            signals: Default::default(),
            signal_index: 0,
            pitch_val: [-1.0, 0.0],
            pitches: Default::default(),
            median_index: 0,
            previous_saw: 0.0,
            final_pitch: 0.0,
            // detectors: [McLeodDetector::new(2, 1);MAX_DETECTOR_SIZE_POWER-MIN_DETECTOR_SIZE_POWER],
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
                5,
                IntRange::Linear {
                    min: 0,
                    max: (NR_OF_DETECTORS - 1) as i32,
                },
            ),

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
            min_pitch: FloatParam::new(
                "Minimum Pitch",
                80.0,
                FloatRange::Skewed {
                    min: 40.0,
                    max: 4000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            ),
            max_pitch: FloatParam::new(
                "Maximum Pitch",
                440.0,
                FloatRange::Skewed {
                    min: 80.0,
                    max: 8000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            ),
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

        for i in 0..OVERLAP {
            // self.signals[i].resize(DETECTOR_SIZE * (self.sample_rate as usize) / 48000, 0.0);
            self.signals[i].resize(2_usize.pow(MAX_DETECTOR_SIZE_POWER as u32), 0.0);
            // self.signals[i].resize(2^11, 0.0);

            // let len = self.signals[i].len();
            // println!("len {}: {}",i,len);
            // println!("index: {}",self.signal_index+(i*(len/OVERLAP))%len);
        }
        for i in 0..NR_OF_DETECTORS {
            // let size = 2^i;
            let size = 2_usize.pow((i+MIN_DETECTOR_SIZE_POWER) as u32);
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
        let size = 2_usize.pow(
            (MIN_DETECTOR_SIZE_POWER + self.params.detector_size.value() as usize)
                as u32);

        if self.signals[0].len() != size {
            for i in 0..OVERLAP {
                self.signals[i].resize(size as usize, 0.0);
            }
            context.set_latency_samples(size as u32);
        }

        let len = self.signals[0].len();


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
                        // copy our sample to signal
                        for i in 0..OVERLAP {
                            self.signals[i][staggered_index(i,self.signal_index,len)] =
                                *sample as f32;
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
                            if staggered_index(i,self.signal_index,len) == 0 {
                                // call the pitchtracker
                                self.pitch_val = pitch::pitch(
                                    self.sample_rate,
                                    &self.signals[i],
                                    // &mut self.detectors[MAX_DETECTOR_SIZE_POWER-MIN_DETECTOR_SIZE_POWER-1],
                                    &mut self.detectors[self.params.detector_size.value() as usize],
                                    // detector,
                                    self.params.power_threshold.value(),
                                    // clarity_threshold: use 0.0, so all pitch values are let trough
                                    0.0,
                                    self.params.pick_threshold.value(),
                                );
                                // if clarity is high enough
                                if self.pitch_val[1] > self.params.clarity_threshold.value()
                                // and the pitch isn't too high
                                    && self.pitch_val[0] > self.params.min_pitch.value()
                                    && self.pitch_val[0] < self.params.max_pitch.value()
                                {
                                    // update the pitches
                                    self.pitches[self.median_index] = self.pitch_val[0];
                                    // update the ringbuf pointer
                                    self.median_index = (self.median_index + 1) % MEDIAN_NR;
                                }
                                // nih_trace!(
                                // "i: {}, Frequency: {}, Clarity: {}",
                                // i, self.pitch_val[0], self.pitch_val[1]
                                // );

                                // get the median pitch:
                                // copy the pitches, we don't want to sort the ringbuffer
                                let mut sorted : [f32; MEDIAN_NR] = self.pitches;
                                // sort the copy
                                if sorted[0] != 0.0 {
                                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                }

                                // get the middle one
                                self.final_pitch = sorted[sorted.len() / 2];
                                // nih_trace!("pitch: {}", self.final_pitch);
                                // println!("pitch: {}", self.final_pitch);
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

        fn staggered_index(i:usize,index:usize,len:usize) -> usize {
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

nih_export_clap!(VoiceMaster);
nih_export_vst3!(VoiceMaster);
