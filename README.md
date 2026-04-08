# ⚠️ DEPRECATED

**This package has been deprecated in favor of [hbw](https://github.com/finite-sample/hbw).**

[![PyPI version](https://img.shields.io/pypi/v/hessband.svg)](https://pypi.org/project/hessband/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why hbw?

**hbw** provides everything hessband does, plus:

- **Multivariate support**: KDE and Nadaraya-Watson for multivariate data
- **More kernels**: 6 kernels (gaussian, epanechnikov, uniform, biweight, triweight, cosine) vs 2
- **Numba acceleration**: 16-49× faster than pure NumPy implementations
- **Large data handling**: Automatic subsampling via `max_n` parameter
- **Active development**: Regular updates and improvements

## Migration

```bash
pip uninstall hessband
pip install hbw
```

```python
# Old (hessband)
from hessband import select_kde_bandwidth, select_nw_bandwidth

# New (hbw)
from hbw import kde_bandwidth, nw_bandwidth
```

## Links

- **hbw repository**: https://github.com/finite-sample/hbw
- **hbw on PyPI**: https://pypi.org/project/hbw/
- **hbw documentation**: https://finite-sample.github.io/hbw/
