import os
from pathlib import Path

import pytest

# Eyes-closed resting-state EEG from a healthy control in OpenNeuro ds004504
# ("A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal
# dementia and Healthy subjects"). sub-039 has a strong occipital alpha
# rhythm, so phase estimates on it can be checked against every target
DATASET = 'ds004504'
DATASET_TAG = '1.0.8'
SUBJECT = 'sub-039'
CHANNEL = 'O1'
DATASET_DIR = Path(__file__).parent / 'test_data' / DATASET
RECORDING = f'{SUBJECT}/eeg/{SUBJECT}_task-eyesclosed_eeg.set'


@pytest.fixture(scope='session')
def resting_state_eeg():
    '''
    O1 from one participant's eyes-closed resting-state recording, split in
    half for training and testing. The recording (~33 MB) is downloaded
    into tests/test_data on first use and reused afterwards

    Returns
    -------
    training_data : array (n_samples,)
        First half of the recording
    testing_data : array (n_samples,)
        Second half of the recording
    sampling_rate : int
        Sampling rate in Hz (500)
    '''
    if os.getenv('GITHUB_ACTIONS') == 'true':
        pytest.skip('real data tests are not run on GitHub Actions')

    mne = pytest.importorskip('mne')
    openneuro = pytest.importorskip('openneuro')

    path = DATASET_DIR / RECORDING
    if not path.exists():
        try:
            openneuro.download(dataset=DATASET, tag=DATASET_TAG,
                               target_dir=DATASET_DIR, include=[RECORDING])
        except Exception as error:
            pytest.skip(f'could not download {DATASET} {SUBJECT}: {error}')

    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    data = raw.pick([CHANNEL]).get_data()[0]
    split = len(data) // 2

    return data[:split], data[split:], int(raw.info['sfreq'])
