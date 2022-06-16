# Colorimeter-Error-Estimation
## Description:
Four color calibration is commonly implenmented in 2D imaging colorimeters. A 3*3 (or 3*4) color matrix is generated afterwards to compensate spectral offset between color filter and CIE color matching function. Spectral offset between DUT and calibration light source (CLS) can introduce errors in both luminance and chromaticity. This python library can be used to calculate these errors based on the colorimeter spetral response and CLS&amp;DUT spectra.
## Usage:
Check "Example.ipynb"
## Data Loading: 
csv file, must follow specific data format
1. Colorimeter spectal response: four columns (wavelength, x filter, y filter, z filter). An example can found in "Example\\RGB_response.csv".
2. Calibration light source spectra: four columns (wavelength, Red channel, Green channel, Blue channel). An example can found in "Example\\Calibration_LS.csv".
3. DUT spectra: Three options
	1) Shifting CLS spectra
	2) Loading DUT spectra
	3) Assume DUT spectra Gaussian distribution, input central wavelength&FWHM (nm). An example can found in "Example\\FWHM_and_central_wavelength.csv".
