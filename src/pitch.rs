use pitch_detection::detector::mcleod::McLeodDetector;
use pitch_detection::detector::PitchDetector;

pub(crate) fn pitch(sample_rate: f32, signal: &Vec<f32>) -> [f32; 2] {
    // Include only notes that exceed a power threshold which relates to the
    // amplitude of frequencies in the signal. Use the suggested default
    // value of 5.0 from the library.
    const POWER_THRESHOLD: f32 = 0.0;
    // The clarity measure describes how coherent the sound of a note is. For
    // example, the background sound in a crowded room would typically be would
    // have low clarity and a ringing tuning fork would have high clarity.
    // This threshold is used to accept detect notes that are clear enough
    // (valid values are in the range 0-1).
    const CLARITY_THRESHOLD: f32 = 0.0;

    let size = signal.len() as usize;

    let padding = size / 2 as usize;

    let mut detector = McLeodDetector::new(size, padding);

    let optional_pitch = detector.get_pitch(
        &signal,
        sample_rate as usize,
        POWER_THRESHOLD,
        CLARITY_THRESHOLD,
    );

    let mut pitch_val: [f32; 2] = [-1.0, 0.0];

    match optional_pitch {
        Some(pit) => {
            pitch_val[0] = pit.frequency;
            pitch_val[1] = pit.clarity;
        }
        None => {
            pitch_val[0] = -1.0;
            pitch_val[1] = 0.0;
        }
    };

    return pitch_val;
}
