# SPT-PikLiteHM-TTTEEE-likelihood
This likelihood is inspired by the one proposed and used in Kable et al., 2023 (https://iopscience.iop.org/article/10.3847/1538-4357/acfed0). The latter is constructed performing a false cut to the covariance matrix, namely "inflating" the errors relative to the TT-block of the covariance matrix from $\ell$=650, in order to make the $\chi^2$ too large at higher multipoles.
By "inflating" we mean multiplying these errors by a factor so large that the corresponding blocks of the covariance matrix are ignored by the likelihood, like if we made an actual cut.
The choice of $\ell$=650 was shown to provide good agreement with WMAP constraints while avoiding excessive overlap with the SPT multipole range.
