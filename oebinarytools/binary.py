from oebinarytools.paths import TEST_DATA_DIR
import json
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw


class BinaryExperiment:
    def __init__(
        self, experiment_paths,
    ):
        # retreive experiment info
        dat_files, info_files, sync_files = retrive_experiments(experiment_paths)

        # create a dictionary of recording info
        self.recordings = create_recordings_dict(dat_files, info_files, sync_files)


def retrive_experiments(experiment_paths):
    """ Grab all experiments from paths
    """
    all_dat_files = []
    all_info_files = []
    all_recording_message_files = []
    for experiment_loc in experiment_paths:
        dat_files = list(experiment_loc.glob("**/*.dat"))
        info_files = list(experiment_loc.glob("**/structure.oebin"))
        recording_message_files = list(experiment_loc.glob("**/sync_messages.txt"))
        all_dat_files += dat_files
        all_info_files += info_files
        all_recording_message_files += recording_message_files
    return all_dat_files, all_info_files, all_recording_message_files


RPIOPERANT_DIGITAL_CHANNELS = {
    1: "Left Peck",
    2: "Center Peck",
    3: "Right Peck",
    4: "Hopper",
    5: "TXD",
    6: "GPIO22",
    7: "GPIO27",
    8: "GPIO8EXT",
}

ADC_CHANNELS = {"sine": 0, "audio": 1, "audio_binary": 2}


def get_recording_ttl_events(dat_file, FPGA_events_folder):
    """ load and grab FPGA TTL events from corresponding dat file
    """
    FPGA_events = dat_file.parent.parent.parent / "events" / FPGA_events_folder
    channel_states = np.load(FPGA_events / "channel_states.npy")
    channels = np.load(FPGA_events / "channels.npy")
    full_words = np.load(FPGA_events / "full_words.npy")
    timestamps = np.load(FPGA_events / "timestamps.npy")

    TTL_events = pd.DataFrame(
        {
            "channels": channels,
            "channel_states": channel_states,
            "timestamps": timestamps,
            "full_words": full_words,
        }
    )

    # merge with channel information
    TTL_events["channel_ID"] = [
        RPIOPERANT_DIGITAL_CHANNELS[i] for i in TTL_events["channels"]
    ]

    return TTL_events


def parse_Network_Events(recording):
    """ goes through Network Events folder and grabs all relevant data
    """
    # search through recording for network events
    network_events_dfs = []
    for event_folder in recording["info"]["events"]:
        if "Network_Events" in event_folder["folder_name"]:
            if "TEXT_group" in event_folder["folder_name"]:

                # load text and timestamp events
                zmq_text_events_folter = (
                    recording["dat_file"].parents[2]
                    / "events"
                    / event_folder["folder_name"]
                )
                channels = np.load(zmq_text_events_folter / "channels.npy")
                metadata = np.load(zmq_text_events_folter / "metadata.npy")
                text = np.load(zmq_text_events_folter / "text.npy")
                timestamps = np.load(zmq_text_events_folter / "timestamps.npy")

                network_events_df = pd.DataFrame(
                    {
                        "channels": channels,
                        "metadata": metadata,
                        "text": text.astype("str"),
                        "timestamps": timestamps,
                    }
                )
                network_events_dfs.append(network_events_df)
    if len(network_events_dfs) > 0:
        return pd.concat(network_events_dfs)


def find_first_peak(data, thresh_factor=0.3):
    """ Find the first peak in a sine wave 
        Originally by Zeke Arneodo
    """
    data = data - np.mean(data)
    thresh = np.max(data) * thresh_factor
    a = data[1:-1] - data[2:]
    b = data[1:-1] - data[:-2]
    c = data[1:-1]
    max_pos = np.where((a >= 0) & (b > 0) & (c > thresh))[0] + 1
    return int(max_pos[0])


def detect_audio_events(
    recording, sine_channel_num=None, audio_binary_channel_num=None, signal_pad_s=0.1
):
    """ Find audio events given a sine channel and an audio onset/offset signal
    """

    # get recording info
    dat_file = recording["dat_file"]
    sample_rate = recording["info"]["continuous"][0]["sample_rate"]
    num_channels = recording["info"]["continuous"][0]["num_channels"]

    # load recording data into spikeinterface
    recording_si = se.BinDatRecordingExtractor(
        dat_file, sampling_frequency=sample_rate, numchan=num_channels, dtype="int16"
    )

    recording_timestamps = np.load(dat_file.parent / "timestamps.npy")

    # assume sine channel is first ADC channel
    if sine_channel_num is None:
        sine_channel_num = recording["ADC_data_channels"][ADC_CHANNELS["sine"]]
    # assume audio binary channel is third ADC channel
    if audio_binary_channel_num is None:
        audio_binary_channel_num = recording["ADC_data_channels"][
            ADC_CHANNELS["audio_binary"]
        ]

    # get sine channel
    sine_channel = recording_si.get_traces()[sine_channel_num]
    # get audio_on channel
    audio_on_channel = recording_si.get_traces()[audio_binary_channel_num]
    # make binary
    audio_on_channel_binary = audio_on_channel / np.max(audio_on_channel) > 0.1

    # get starts and ends of playback
    audio_starts = np.where(
        (audio_on_channel_binary[:-1] == 0) & (audio_on_channel_binary[1:] == 1)
    )[0]
    audio_stops = np.where(
        (audio_on_channel_binary[:-1] == 1) & (audio_on_channel_binary[1:] == 0)
    )[0]
    # set equal, in case we begin or end in the middle of a playback
    audio_stops = audio_stops[-len(audio_starts) :]
    audio_starts = audio_starts[: len(audio_stops)]

    # sync audio playback with sine channel
    signal_pad_frames = signal_pad_s * sample_rate

    # get where the sine wave begins in that region
    start_frames = np.zeros(len(audio_starts))
    end_frames = np.zeros(len(audio_stops))
    # get sine beginning and ending
    for playback_i, (start, stop) in enumerate(zip(audio_starts, audio_stops)):
        window_start = start + signal_pad_frames
        window_end = stop - signal_pad_frames
        window = sine_channel[int(window_start) : int(window_end)]
        fp = find_first_peak(window) + window_start
        lp = len(window) - find_first_peak(window[::-1]) + window_start
        start_frames[playback_i] = recording_timestamps[int(fp)]
        end_frames[playback_i] = recording_timestamps[int(lp)]

    playback_df = pd.DataFrame(
        {"timestamp_begin": start_frames, "timestamp_end": end_frames}
    )

    return playback_df


def merge_network_audio_events_with_FPGA_sine(recording, stim_search_padding_s=0.4):
    """ Given a dataset of network events over ZMQ, and a dataset of
     detected sine audio events, merge the events
    """

    # stim declaration via zmq can precede stim playback by up to this amount
    sample_rate = recording["info"]["continuous"][0]["sample_rate"]
    stim_search_padding_frames = stim_search_padding_s * sample_rate

    stim_mask = ["stim" in i for i in recording["network_events"].text]
    audio_network_events = recording["network_events"][stim_mask]

    # for each stim playback, if a zmq audio event exists (within
    #     stim_search_padding_s) , merge it
    recording["audio_events"]["stim"] = None
    recording["audio_events"]["metadata"] = None
    recording["audio_events"]["channels"] = None
    recording["audio_events"]["timestamp_stim"] = None
    for idx, row in recording["audio_events"].iterrows():

        starts_before = audio_network_events.timestamps.values > (
            row.timestamp_begin - stim_search_padding_frames
        )
        ends_after = audio_network_events.timestamps.values < (row.timestamp_end)

        if np.any(starts_before & ends_after):
            matching_row = audio_network_events.iloc[
                np.where(starts_before & ends_after)[0][0]
            ]
            recording["audio_events"].loc[idx, "channel"] = matching_row.channels
            recording["audio_events"].loc[idx, "stim"] = matching_row.text
            recording["audio_events"].loc[idx, "metadata"] = matching_row.metadata
            recording["audio_events"].loc[
                idx, "timestamp_stim"
            ] = matching_row.timestamps

    return recording["audio_events"]


def create_recordings_dict(dat_files, info_files, sync_files):
    """ create a dictionary of all recordings and parameters
    """

    recordings = {}
    for ri, (dat_file, info_file, sync_file) in enumerate(
        zip(dat_files, info_files, sync_files)
    ):
        #
        recordings[ri] = {
            "dat_file": dat_file,
            "info_file": info_file,
            "sync_file": sync_file,
        }

        # add info from .oebin
        with open(info_file.as_posix(), "r") as f:
            info = f.read()
        recordings[ri]["info"] = json.loads(info)

        # add sync messages
        with open(sync_file.as_posix(), "r") as f:
            sync_message = f.read()
        recordings[ri]["sync_message"] = sync_message

        # grab TTL events from FPGA events folder
        FPGA_events_folder = recordings[ri]["info"]["events"][0]["folder_name"]
        recordings[ri]["FPGA_events"] = get_recording_ttl_events(
            dat_file, FPGA_events_folder
        )

        # specify neural data channels
        channel_desc = recordings[ri]["info"]["continuous"][0]["channels"]
        recordings[ri]["neural_data_channels"] = [
            i["source_processor_index"]
            for i in channel_desc
            if i["description"] == "Headstage data channel"
        ]
        recordings[ri]["AUX_data_channels"] = [
            i["source_processor_index"]
            for i in channel_desc
            if i["description"] == "Auxiliar data channel"
        ]

        recordings[ri]["ADC_data_channels"] = [
            i["source_processor_index"]
            for i in channel_desc
            if i["description"] == "ADC data channel"
        ]

        # parse network events for each recording
        recordings[ri]["network_events"] = parse_Network_Events(recordings[ri])

        # get audio events
        recordings[ri]["audio_events"] = detect_audio_events(recordings[ri])

        # merge audio events with network events
        recordings[ri]["audio_events"] = merge_network_audio_events_with_FPGA_sine(
            recordings[ri]
        )

        # merge network events with TTL events
        # TODO

    return recordings
