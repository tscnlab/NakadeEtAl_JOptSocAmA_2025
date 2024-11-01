# Repository for the code used in the paper "Predicting visual field boundaries from head features"
### Authors: Uday Nakade and Prof. Dr. Manuel Spitschan

This repository contains the code used in our paper. In order to run it, 
you will need at least one of the following variants of [Mitsuba 3](https://www.mitsuba-renderer.org/): 
`cuda_ad_spectral` (the one we used), `cuda_spectral`, `llvm_ad_spectral`, `llvm_spectral`, `scalar_spectral`, and [Blender](https://www.blender.org/).

In a terminal, run the following commands in the given order to reproduce our results:

```bash
blender -b ICT-FaceKit-just-face-tri.blend -P export_from_blender.py
source /path/to/venv/bin/activate
python3 get_eye_centers.py
source /path/to/mitsuba3/setpath.sh
python3 render_ply.py
python3 get_vf_boundaries.py
python3 optimize_vf_boundaries.py
python3 create_hemispherical_vf_images.py  # Time consuming, can be skipped
python3 plotting.py
```

The `ICT-FaceKit-just-face-tri.blend` file was generated using the [ICT FaceKit](https://github.com/ICT-VGL/ICT-FaceKit).
We first imported the face model into Blender, removed irrelevant parts (eyeballs, eyelashes, internal structure of the 
mouth, etc.) and triangulated all faces (required by Mitsuba 3).

We used `python` version `3.10.12` and `Blender` version `3.3.21`. The commit hash of the version of `Mitsuba 3` we used was 
`d310cfd4dc5662903e0ebcbaf4a3704e8d57c953` with the output of `git submodule status` as follows:
```
 cae01e3964a44d76cb32ba574d80828217636704 ext/asmjit (heads/master)
 50532d291b2dcf3fc910fcba751d452d7cfedb78 ext/drjit (heads/master)
 7d93c92c2ea6c4f1369d1b8688528b5fe26e91b4 ext/embree (heads/master)
 052975dd5f8166d0f9e4a215fa75a349d5985b91 ext/fastfloat (052975d)
 88a56e0ecd162667c7afd2ee9969221d62a32509 ext/ittnotify (88a56e0)
 d3841d172c83f709151a9654e5aaed006cc81ff7 ext/libjpeg (heads/master)
 d14bc51bf5f8f5ab71a42ece806a43485aedb5d4 ext/libpng (heads/master)
 a6afde86f48bf2e3a00689c7145746c89fa474a3 ext/nanogui (a6afde8)
 dbabb6f9500ee628c1faba21bb8add2649cc32a6 ext/openexr (heads/master)
 aa7280f2b3359575efe5401eea58e5d7851c923b ext/pugixml (heads/master)
 80dc998efced8ceb2be59756668a7e90e8bef917 ext/pybind11 (v2.10.1)
 635345c75bd95891ee041ac51ce74ebc891d5bab ext/tinyformat (635345c)
 080a732c47e86444034c1f99355368d35c1e458a ext/zlib (heads/master)
 8b05b3188d84ddf623b45af22e45a6d77ba079d2 resources/data (heads/master)
 0b77266a0eff13719ee5000a049d33320d3637bf tutorials (heads/master)
```
We only installed `numpy`, `scipy`, `matplotlib` and `tensorflow` in the Python environment. 
For reference, we have included a `requirements.txt` file in the repository (result of `pip freeze`).
