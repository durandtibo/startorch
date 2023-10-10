# startorch

<p align="center">
    <a href="https://github.com/durandtibo/startorch/actions">
        <img alt="CI" src="https://github.com/durandtibo/startorch/workflows/CI/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/startorch/">
        <img alt="Documentation" src="https://github.com/durandtibo/startorch/workflows/Documentation/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/startorch/actions">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/startorch/workflows/Nightly%20Tests/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/startorch/actions">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/startorch/workflows/Nightly%20Package%20Tests/badge.svg">
    </a>
    <br/>
    <a href="https://codecov.io/gh/durandtibo/startorch">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/startorch/branch/main/graph/badge.svg">
    </a>
    <a href="https://codeclimate.com/github/durandtibo/startorch/maintainability">
        <img src="https://api.codeclimate.com/v1/badges/05a12c503bf3be80a00b/maintainability" />
    </a>
    <a href="https://codeclimate.com/github/durandtibo/startorch/test_coverage">
        <img src="https://api.codeclimate.com/v1/badges/05a12c503bf3be80a00b/test_coverage" />
    </a>
    <br/>
    <a href="https://pypi.org/project/startorch/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/startorch">
    </a>
    <a href="https://pypi.org/project/startorch/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/startorch.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/startorch">
    </a>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <br/>
    <a href="https://pepy.tech/project/startorch">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/startorch">
    </a>
    <a href="https://pepy.tech/project/startorch">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/startorch/month">
    </a>
    <br/>
</p>

---

## Overview

Collecting datasets to train Machine Learning models can be time consuming.
Another alternative is to use synthetic datasets.
`startorch` is a Python library to generate synthetic time-series.
As the name suggest, `startorch` relies mostly on PyTorch to generate the time series and to control
the randomness.
`startorch` is built to be modular, flexible and extensible.
Below show some generated sequences by `startorch` where the values are sampled from different
distribution.

<table align="center">
  <tr>
    <td><img src="https://durandtibo.github.io/startorch/assets/figures/uniform.png" width="400" align="center"></td>
    <td><img src="https://durandtibo.github.io/startorch/assets/figures/log-uniform.png" width="400" align="center"></td>
  </tr>
  <tr>
    <td align="center">uniform</td>
    <td align="center">log-uniform</td>
  </tr>
  <tr>
    <td><img src="https://durandtibo.github.io/startorch/assets/figures/sinewave.png" width="400" align="center"></td>
    <td><img src="https://durandtibo.github.io/startorch/assets/figures/wiener.png" width="400" align="center"></td>
  </tr>
  <tr>
    <td align="center">sine wave</td>
    <td align="center">Wiener process</td>
  </tr>
</table>

- [Documentation](https://durandtibo.github.io/startorch/)
- [Installation](https://durandtibo.github.io/startorch/get_started/)
- [Contributing](#contributing)
- [API stability](#api-stability)
- [License](#license)

## Dependencies

| `startorch` | `coola`            | `objectory`      | `redcat`           | `torch`      | `matplotlib`<sup>*</sup> | `python`      |
|-------------|--------------------|------------------|--------------------|--------------|--------------------------|---------------|
| `main`      | `>=0.0.20,<0.0.25` | `>=0.0.7,<0.0.9` | `>=0.0.16,<0.0.18` | `>=2.0,<2.2` | `>=3.6,<3.9`             | `>=3.9,<3.12` |
| `0.0.6`     | `>=0.0.20,<0.0.25` | `>=0.0.7,<0.0.9` | `>=0.0.16,<0.0.18` | `>=2.0,<2.2` | `>=3.6,<3.9`             | `>=3.9,<3.12` |
| `0.0.5`     | `>=0.0.20,<0.0.24` | `>=0.0.7,<0.0.8` | `>=0.0.16,<0.0.17` | `>=2.0,<2.1` | `>=3.6,<3.9`             | `>=3.9,<3.12` |
| `0.0.4`     | `>=0.0.20,<0.0.24` | `>=0.0.7,<0.0.8` | `>=0.0.16,<0.0.17` | `>=2.0,<2.1` | `>=3.6,<3.9`             | `>=3.9,<3.12` |
| `0.0.3`     | `>=0.0.20,<0.0.24` | `>=0.0.7,<0.0.8` | `>=0.0.9,<0.0.10`  | `>=2.0,<2.1` | `>=3.6,<3.9`             | `>=3.9,<3.12` |

<sup>*</sup> indicates an optional dependency

## Contributing

Please check the instructions in [CONTRIBUTING.md](.github/CONTRIBUTING.md).

## API stability

:warning: While `startorch` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `startorch` to a new version will possibly break any code
that was using the old version of `startorch`.

## License

`startorch` is licensed under BSD 3-Clause "New" or "Revised" license available
in [LICENSE](LICENSE) file.
