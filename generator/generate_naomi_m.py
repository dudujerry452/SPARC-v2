#!/usr/bin/env python3
"""Render naomi_generate.m from a JSON configuration file."""

import argparse
import json
import sys
from pathlib import Path


def matlab_bool(value: bool) -> str:
    return "true" if value else "false"


def matlab_string(value: str) -> str:
    return f"'{value}'"


def matlab_array(values: list) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def validate_config(cfg: dict) -> None:
    required = {
        "image_size": list,
        "pixel_size": (int, float),
        "num_frames": int,
        "frame_rate": (int, float),
        "imaging_depth": (int, float),
        "sample_depth": (int, float),
        "avg_neuron_radius": (int, float),
        "num_neurons": int,
        "avg_firing_rate": (int, float),
        "vasculature_on": bool,
        "background_dendrites_on": bool,
        "calcium_indicator": str,
        "fluorophore_conc": (int, float),
        "excitation_NA": (int, float),
        "detection_NA": (int, float),
        "wavelength": (int, float),
        "rep_rate": (int, float),
        "pulse_width": (int, float),
        "excitation_power": (int, float),
        "obj_focal_length": (int, float),
        "psf_type": str,
        "brain_motion": bool,
        "uint16_scale": (int, float),
        "random_seed": (str, int),
        "output_prefix": str,
    }

    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    for key, expected_type in required.items():
        if not isinstance(cfg[key], expected_type):
            raise TypeError(
                f"Config key '{key}' must be {expected_type}, got {type(cfg[key]).__name__}"
            )

    seed = cfg["random_seed"]
    if isinstance(seed, int) and seed < 0:
        raise ValueError("random_seed integer must be non-negative")

    if "'" in cfg["output_prefix"]:
        raise ValueError("output_prefix must not contain single quotes")

    if len(cfg["image_size"]) != 2 or not all(isinstance(v, int) for v in cfg["image_size"]):
        raise ValueError("image_size must be a list of two integers")

    if cfg["num_frames"] <= 0 or cfg["num_neurons"] <= 0:
        raise ValueError("num_frames and num_neurons must be positive")


TEMPLATE = """function naomi_generate()
% generate_tpm_movie_custom
%
% Generate a clean (noise-free) two-photon microscopy movie using NAOMi-Sim
% without fast-mask, with configurable arbitrary pixel size.

disp("Begin to Generate NAOMi Data... ");

installNAOMi;
{rng_line}  % Initialize random seed

%% User-configurable parameters
image_size    = {image_size};     % Output image size [height, width] (pixels)
pixel_size    = {pixel_size};          % Pixel size (um)
num_frames    = {num_frames};           % Number of frames
frame_rate    = {frame_rate};             % Hz
imaging_depth = {imaging_depth};            % um
sample_depth  = {sample_depth};             % um

avg_neuron_radius = {avg_neuron_radius};        % um
num_neurons   = {num_neurons};
avg_firing_rate = {avg_firing_rate};         % Hz
vasculature_on = {vasculature_on};
background_dendrites_on = {background_dendrites_on};
calcium_indicator = {calcium_indicator};
fluorophore_conc = {fluorophore_conc};          % uM

excitation_NA = {excitation_NA};
detection_NA  = {detection_NA};
wavelength    = {wavelength};            % nm
rep_rate      = {rep_rate};             % MHz
pulse_width   = {pulse_width};            % fs
excitation_power = {excitation_power};          % mW
obj_focal_length = {obj_focal_length};         % mm
psf_type      = {psf_type};
brain_motion  = {brain_motion};

time_tag = datestr(now, 'yyyymmddHHMM');
save_filename = ['{output_prefix}_' time_tag '.mat'];

%% Parameter structs
fov = round(image_size * pixel_size);

vol_params.vol_sz    = [fov(1), fov(2), sample_depth];
vol_params.vol_depth = imaging_depth;
vol_params.vres      = 2;
vol_params.N_neur    = num_neurons;

neur_params.avg_rad  = avg_neuron_radius;

if vasculature_on, vasc_params.flag = 1; else, vasc_params.flag = 0; end
if background_dendrites_on, bg_params.flag = 1; else, bg_params.flag = 0; end

spike_opts.nt    = num_frames;
spike_opts.dt    = 1 / frame_rate;
spike_opts.rate  = avg_firing_rate;
spike_opts.prot  = calcium_indicator;

psf_params.NA        = excitation_NA;
psf_params.objNA     = detection_NA;
psf_params.lambda    = wavelength / 1000;
psf_params.obj_fl    = obj_focal_length;
psf_params.type      = psf_type;
psf_params.fastmask  = true;

tpm_params.pavg      = excitation_power;
tpm_params.f         = rep_rate;
tpm_params.tau       = pulse_width;
tpm_params.conc      = fluorophore_conc;
tpm_params.lambda    = wavelength / 1000;
tpm_params.nac       = detection_NA;

scan_params.motion    = brain_motion;
scan_params.scan_buff = 0;
scan_params.sfrac     = 1;
scan_params.movout    = 1;

%% Check parameters
vol_params   = check_vol_params(vol_params);
vasc_params  = check_vasc_params(vasc_params);
neur_params  = check_neur_params(neur_params);
dend_params  = check_dend_params([]);
axon_params  = check_axon_params([]);
bg_params    = check_bg_params(bg_params);
spike_opts   = check_spike_opts(spike_opts);
noise_params = check_noise_params([]);
psf_params   = check_psf_params(psf_params);
scan_params  = check_scan_params(scan_params);
tpm_params   = check_tpm_params(tpm_params);

%% Generate volume, PSF, activity, and scan
[vol_out, vol_params, neur_params, vasc_params, dend_params, bg_params, axon_params] = ...
    simulate_neural_volume(vol_params, neur_params, vasc_params, dend_params, ...
    bg_params, axon_params, psf_params);

PSF_struct = simulate_optical_propagation(vol_params, psf_params, vol_out);

spike_opts.K = size(vol_out.gp_vals, 1);
[neur_act, spikes] = generateTimeTraces(spike_opts, [], vol_out.locs);

[~, Fsim_clean] = scan_volume(vol_out, PSF_struct, neur_act, ...
    scan_params, noise_params, spike_opts, tpm_params);

%% Resize to exact target size if necessary
if size(Fsim_clean, 1) ~= image_size(1) || size(Fsim_clean, 2) ~= image_size(2)
    Fsim_tmp = zeros([image_size, num_frames], 'single');
    for t = 1:num_frames
        Fsim_tmp(:,:,t) = imresize(Fsim_clean(:,:,t), image_size);
    end
    Fsim_clean = Fsim_tmp;
end

%% Convert to uint16 and save
uint16_scale = {uint16_scale};                               % Scale so most values ~0-150, max ~0-1600
Fsim_clean(isnan(Fsim_clean)) = 0;                % Replace NaN with 0
Fsim_clean(Fsim_clean < 0) = 0;                   % Clamp negative values to 0
Fsim_uint16 = uint16(Fsim_clean * uint16_scale);  % Convert to uint16

[~, name, ~] = fileparts(save_filename);
write_TPM_movie(Fsim_uint16, [name '.tif'], 'uint16');

end
"""


def render(cfg: dict) -> str:
    validate_config(cfg)

    seed = cfg["random_seed"]
    if isinstance(seed, str):
        if seed != "shuffle":
            raise ValueError("random_seed string must be 'shuffle'")
        rng_line = "rng('shuffle')"
    else:
        rng_line = f"rng({seed})"

    return TEMPLATE.format(
        rng_line=rng_line,
        image_size=matlab_array(cfg["image_size"]),
        pixel_size=cfg["pixel_size"],
        num_frames=cfg["num_frames"],
        frame_rate=cfg["frame_rate"],
        imaging_depth=cfg["imaging_depth"],
        sample_depth=cfg["sample_depth"],
        avg_neuron_radius=cfg["avg_neuron_radius"],
        num_neurons=cfg["num_neurons"],
        avg_firing_rate=cfg["avg_firing_rate"],
        vasculature_on=matlab_bool(cfg["vasculature_on"]),
        background_dendrites_on=matlab_bool(cfg["background_dendrites_on"]),
        calcium_indicator=matlab_string(cfg["calcium_indicator"]),
        fluorophore_conc=cfg["fluorophore_conc"],
        excitation_NA=cfg["excitation_NA"],
        detection_NA=cfg["detection_NA"],
        wavelength=cfg["wavelength"],
        rep_rate=cfg["rep_rate"],
        pulse_width=cfg["pulse_width"],
        excitation_power=cfg["excitation_power"],
        obj_focal_length=cfg["obj_focal_length"],
        psf_type=matlab_string(cfg["psf_type"]),
        brain_motion=matlab_bool(cfg["brain_motion"]),
        uint16_scale=cfg["uint16_scale"],
        output_prefix=cfg["output_prefix"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render naomi_generate.m from a JSON configuration file."
    )
    parser.add_argument("config", help="Path to the JSON configuration file")
    parser.add_argument(
        "output",
        nargs="?",
        default="naomi_generate.m",
        help="Output MATLAB script path (default: naomi_generate.m)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {config_path}: {exc}", file=sys.stderr)
        return 1

    try:
        rendered = render(cfg)
    except (ValueError, TypeError) as exc:
        print(f"Error: config validation failed: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
