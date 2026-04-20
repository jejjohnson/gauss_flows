# Changelog

All notable changes to this project will be documented in this file.

## [0.1.4](https://github.com/jejjohnson/gauss_flows/compare/v0.1.3...v0.1.4) (2026-04-20)


### Features

* add Orthogonal1x1Conv transform ([#63](https://github.com/jejjohnson/gauss_flows/issues/63)) ([a85d791](https://github.com/jejjohnson/gauss_flows/commit/a85d79129aad56b61cb7d00dd101ac25f2cc0954))
* add Orthogonal1x1Conv transform ([#63](https://github.com/jejjohnson/gauss_flows/issues/63)) ([a85d791](https://github.com/jejjohnson/gauss_flows/commit/a85d79129aad56b61cb7d00dd101ac25f2cc0954))
* batch norm bijection (single-event, fail-loud) — supersedes [#47](https://github.com/jejjohnson/gauss_flows/issues/47) ([#58](https://github.com/jejjohnson/gauss_flows/issues/58)) ([21b9f37](https://github.com/jejjohnson/gauss_flows/commit/21b9f378f5477c501380f4753b681931561e8909))
* orthogonal conv exponential via skew-symmetric kernel (supersedes [#56](https://github.com/jejjohnson/gauss_flows/issues/56)) ([#59](https://github.com/jejjohnson/gauss_flows/issues/59)) ([6a476f7](https://github.com/jejjohnson/gauss_flows/commit/6a476f7dbf96d474e48a798823d9cd9b346b5ca4))
* **pr-43:** simple surjections — Abs, Sort, MaxPool2d, StochasticPermutation ([#43](https://github.com/jejjohnson/gauss_flows/issues/43)) ([074014d](https://github.com/jejjohnson/gauss_flows/commit/074014d9d69ebbfbcba0d74748b9ae15e0a97cce))
* **pr-44:** slice + augment surjections — encoder/decoder-driven dim changers ([#44](https://github.com/jejjohnson/gauss_flows/issues/44)) ([213e8a0](https://github.com/jejjohnson/gauss_flows/commit/213e8a084ed94b22b2c36a708a3108f1c2d05f7d))
* **pr-45:** finish DeepSigmoidCoupling + HistogramCDF stubs ([#45](https://github.com/jejjohnson/gauss_flows/issues/45)) ([a0201d0](https://github.com/jejjohnson/gauss_flows/commit/a0201d0b2ef009297009143b3d7d27195e758191))
* true circular RQ spline + Fourier-feature coupling (supersedes [#46](https://github.com/jejjohnson/gauss_flows/issues/46)) ([#50](https://github.com/jejjohnson/gauss_flows/issues/50)) ([9e82155](https://github.com/jejjohnson/gauss_flows/commit/9e8215581428b206f4170b8f267ccc4f83f47c57))
* vae stochastic transform (supersedes [#57](https://github.com/jejjohnson/gauss_flows/issues/57)) ([#60](https://github.com/jejjohnson/gauss_flows/issues/60)) ([2d8ee21](https://github.com/jejjohnson/gauss_flows/commit/2d8ee2105ae512e872949b0a64f4a750f69312f0))

## [0.1.3](https://github.com/jejjohnson/gauss_flows/compare/v0.1.2...v0.1.3) (2026-04-14)


### Features

* **pr-10:** add GaussianPCA base distribution ([#40](https://github.com/jejjohnson/gauss_flows/issues/40)) ([bee257f](https://github.com/jejjohnson/gauss_flows/commit/bee257f32c3be54a8edeedc646e49b15ca63a8d8))
* **pr-12:** add LULinearPermute bijection ([#37](https://github.com/jejjohnson/gauss_flows/issues/37)) ([e173270](https://github.com/jejjohnson/gauss_flows/commit/e173270cc9cdfb0643a77a00a898ee7d1d2ca143))
* **pr-21:** add PlanarFlow and SylvesterFlow bijections ([#38](https://github.com/jejjohnson/gauss_flows/issues/38)) ([d315536](https://github.com/jejjohnson/gauss_flows/commit/d31553605d67773be5f5403323d6df53714d9528))
* **pr-9:** add GaussianMixture base distribution ([#39](https://github.com/jejjohnson/gauss_flows/issues/39)) ([dd44ce8](https://github.com/jejjohnson/gauss_flows/commit/dd44ce8404c96cdb804f3ce23ead2e477ed2e4f5))

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
