# Changelog

All notable changes to this project will be documented in this file.

## [0.1.2](https://github.com/jejjohnson/gauss_flows/compare/v0.1.1...v0.1.2) (2026-03-01)


### Features

* add flowguide - gaussianization flow variational guide for NumPyro SVI ([9eb1a0f](https://github.com/jejjohnson/gauss_flows/commit/9eb1a0f1e85eeac6c80793ca70ec798948ab2ebc))


### Bug Fixes

* remove unused imports to fix ruff lint failures ([a9097c2](https://github.com/jejjohnson/gauss_flows/commit/a9097c2ea0fb9924c6f1fede478bc86886d3a984))

## [0.1.1](https://github.com/jejjohnson/gauss_flows/compare/v0.1.0...v0.1.1) (2026-03-01)


### Features

* added rbig and gaussianization flows library with flowjax ([64bed5d](https://github.com/jejjohnson/gauss_flows/commit/64bed5d30d8c617dddfbe0fe1aac41db681317c5))

## [Unreleased]

### Added
- Initial port of `rbig_jax` from `jej_vc_snippets` into `gauss_flows` package
- Package structure following `spectraldiffx` conventions (flat layout with `_src/`)
- `gauss_flows._src.flows`: `gaussianization_flow`, `coupling_gaussianization_flow`, `iterative_rbig`
- `gauss_flows._src.transforms.marginal`: `MixtureGaussianCDF`, `MixtureLogisticCDF`, `RQSplineMarginal`, `HistogramCDF`, `InverseGaussCDF`
- `gauss_flows._src.transforms.rotation`: `HouseholderRotation`, `OrthogonalRotation`, `FixedRotation`
- `gauss_flows._src.transforms.coupling`: `ActNorm1D`, `AffineCoupling`, `RQSplineCoupling`, `DeepSigmoidCoupling`
- `gauss_flows._src.transforms.conv`: `Invertible1x1Conv`, `ActNorm`, `HaarWavelet`, `Squeeze`
- `gauss_flows._src.info_theory`: `entropy`, `total_correlation`, `mutual_information`, `kl_divergence`, `negentropy`
- `gauss_flows._src.train`: `fit_gaussianization_flow`
- `gauss_flows._src.numpyro_compat`: `FlowDist`
- `pyproject.toml` with hatchling build backend
- CI workflow (GitHub Actions)
- Test suite
