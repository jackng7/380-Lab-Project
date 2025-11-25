import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# We assume this script lives in the same folder as your CSV files
SUBJECTS = [1, 2]
MUSCLES = ["Triceps", "Deltoid"]

# Six conditions in your study
CONDITIONS = [
    "PP NB",
    "PP SW",
    "PP WB",
    "Regular NB",
    "Regular SW",
    "Regular WB",
]


def load_csv(subject: int, suffix: str) -> pd.DataFrame:
    """
    Load a csv file like:
    'Subject 1 PP NB.csv'      (suffix='PP NB')
    'Subject 1 MVC Deltoid.csv' (suffix='MVC Deltoid')
    """
    path = f"Subject {subject} {suffix}.csv"
    return pd.read_csv(path)


def get_sampling_rate(df: pd.DataFrame) -> float:
    """Estimate sampling rate from the time_sec column."""
    t = df["time_sec"].values
    dt = np.diff(t[:10]).mean()
    fs = 1.0 / dt
    return fs


def bandpass_filter(signal: np.ndarray, fs: float,
                    low: float = 20.0, high: float = 250.0,
                    order: int = 4) -> np.ndarray:
    """Butterworth band-pass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def rms_envelope(signal: np.ndarray, fs: float,
                 window_s: float = 0.20) -> np.ndarray:
    """
    Compute RMS envelope with a moving window (default 200 ms).
    """
    window_samples = int(window_s * fs)
    if window_samples < 1:
        return signal

    squared = signal ** 2
    kernel = np.ones(window_samples) / window_samples
    rms = np.sqrt(np.convolve(squared, kernel, mode="same"))
    return rms


def compute_mvc_peak(subject: int, muscle: str) -> float:
    """
    Compute the peak RMS of the MVC trial for a given subject & muscle.
    Uses:
      - Subject X MVC Deltoid.csv  (Deltoid column)
      - Subject X MVC Tricep.csv   (Triceps column)
    """
    if muscle == "Deltoid":
        df_mvc = load_csv(subject, "MVC Deltoid")
        col = "Deltoid"
    else:  # Triceps
        df_mvc = load_csv(subject, "MVC Tricep")
        col = "Triceps"

    fs = get_sampling_rate(df_mvc)
    sig = df_mvc[col].values
    filt = bandpass_filter(sig, fs)
    env = rms_envelope(filt, fs)
    peak = np.max(np.abs(env))
    return peak


def process_trial(subject: int, condition: str, muscle: str,
                  mvc_peak: float):
    """
    For a single subject/condition/muscle:
      - load csv
      - bandpass filter
      - RMS envelope
      - normalize to MVC peak

    Returns:
      mean_norm, t, raw, filt, norm_env, fs
    """
    df = load_csv(subject, condition)
    fs = get_sampling_rate(df)

    sig = df[muscle].values
    filt = bandpass_filter(sig, fs)
    env = rms_envelope(filt, fs)
    norm_env = env / mvc_peak
    mean_norm = norm_env.mean()

    return mean_norm, df["time_sec"].values, sig, filt, norm_env, fs


def main():
    summary_rows = []
    example_plotted = False

    for subject in SUBJECTS:
        for muscle in MUSCLES:
            mvc_peak = compute_mvc_peak(subject, muscle)
            print(f"Subject {subject} {muscle}: MVC peak RMS = {mvc_peak:.3f}")

            for condition in CONDITIONS:
                try:
                    (mean_norm, t, raw, filt,
                     norm_env, fs) = process_trial(subject, condition,
                                                   muscle, mvc_peak)
                except FileNotFoundError:
                    print(f"  Missing file for Subject {subject}, {condition}")
                    continue

                summary_rows.append(
                    {
                        "Subject": subject,
                        "Muscle": muscle,
                        "Condition": condition,
                        "MeanNormRMS": mean_norm,
                    }
                )

                # Make one nice example figure: Subject 1, Triceps, PP NB
                if (not example_plotted and subject == 1
                        and muscle == "Triceps"
                        and condition == "PP NB"):
                    plt.figure(figsize=(10, 7))

                    plt.subplot(3, 1, 1)
                    plt.plot(t, raw)
                    plt.title("Subject 1 – Triceps – PP NB: Raw EMG")
                    plt.ylabel("mV")

                    plt.subplot(3, 1, 2)
                    plt.plot(t, filt)
                    plt.title("Band-pass filtered (20–250 Hz)")
                    plt.ylabel("mV")

                    plt.subplot(3, 1, 3)
                    plt.plot(t, norm_env)
                    plt.title("Normalized RMS envelope (vs MVC)")
                    plt.ylabel("Norm. units")
                    plt.xlabel("Time (s)")

                    plt.tight_layout()
                    example_plotted = True

    # Put all results into a DataFrame
    summary_df = pd.DataFrame(summary_rows)
    print("\nSummary: mean normalized RMS by subject / condition / muscle")
    print(summary_df)

    # Quick condition-level plot (averaged over subjects 1–2)
    plt.figure(figsize=(8, 5))
    for muscle in MUSCLES:
        means = (
            summary_df[summary_df["Muscle"] == muscle]
            .groupby("Condition")["MeanNormRMS"]
            .mean()
            .reindex(CONDITIONS)
        )
        plt.plot(CONDITIONS, means, marker="o", label=muscle)

    plt.xticks(rotation=45)
    plt.ylabel("Mean normalized RMS (vs MVC)")
    plt.title("Subjects 1–2: Mean normalized EMG per condition")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()