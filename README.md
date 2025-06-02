<div align="center">

# Kiseki

**Automatic Line Art 2D Animation Colorization Based on References**

[![Dynamic JSON Badge][discord-shield]][discord-url]

<!-- Workaround to display total user from https://github.com/badges/shields/issues/4500#issuecomment-2060079995 -->

[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fisekaicreation%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total
[discord-url]: https://discord.gg/u5eBYN8Qns

<p align="center">
  <img src="assets/baka_line.gif" width="30%" />
  <img src="assets/baka_ref.png" width="30%" />
  <img src="assets/baka.gif" width="30%" />
</p>

</div>

## Updates

- 29/05/2025: v0.0.1 WIP improving upon existing works

## Pipeline

1. Deduplicating Frames
2. Segmentation
3. Colorization
4. Visualization

## TO-DOs

### Critical

- [ ] Work on NEW Colorization Model to be used more freely and commercially.
- [ ] Reimplement Segmentation Algorithm to be blazingly fast
- [ ] NEW Colorization Model Architecture needed to be truly reference-based, parrallelized
- [ ] Build a simple API interface, EG: for Discord Bot/Apps Cloud Plugins
- [ ] Build Plugins/Interface for different Apps

### Nice-to-haves

- [ ] Reimplement parts in C++, for better performance, otherwise use Numba
- [ ] Additional Integration for CUDA, ROCM or XLA to have better performance
- [ ] Additional Processing for non-transparency images
- [ ] A Refiner/Cleanup Step can be added before Segmentation
- [ ] Setup as a CLI package

## Preparation

### Download Model

```
wget -O ckpt/basicpbc.pth https://huggingface.co/radna/Kiseki-ckpt/resolve/main/basicpbc.pth
```

### Data Placement

Now place your 'lines' and 'references' images in `datatest/test/{name}`, correspondingly within `line_raw` and `ref_raw` folders

## Usage

```
sh scripts/inference.sh datatest/test/{name}
```

## Contact

If you need anything, please feel free reach me on Discord 24/7 at `_radna` or through isekaicreationofficial@gmail.com.

## License/Important

~~This project is licensed under `Kiseki License 0.1`. Usage of this tool for commercial purposes should follow this license.~~

_Currently the following project can't be used commercially in anyway. New works are being done to make it possible to use it freely and commercially._
