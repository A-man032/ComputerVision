File List:

Mingyuan_nips2009_final.pdf: the final version of the paper "Non-Parametric Bayesian Dictionary Learning for Sparse Image Representations," Neural Information Processing Systems (NIPS), 2009.
Model and Inference.pdf: The model and inference for the above paper.
house.png: the orignial house image. 
barbara.png: the original barbara image.
castle_damaged.png: the castle image with 80% pixels missing.
castle_original.png: the orignial castle image.
fig13_damaged.png, fig13_damaged.bmp: the New Orleans image corrupted by text.
fig13_original.png: the original New Orleans image.
Solve out of memory error.txt: Instructions for allowing Matlab to manage 3GB memory instead of the default 2GB memory in Windows XP.

Demo files:
Demo_Denoising.m: Running file for BPFA image denoising.
Demo_Inpainting_GRAY.m: Running file for BPFA gray scale image inpainting.
Demo_Inpainting_RGB.m: Running file for BPFA rgb image inpainting.

Main programs:
BPFA_Denoising.m: The BPFA grayscale image denoising program.
BPFA_Inpainting.m: The BPFA image inpainting program.

Subprograms for Gibbs sampling:
SampleDZS.m: Sampling the dictionary D, the binary indicating matrix Z, and the pseudo weight matrix S. Used for no missing data case.
SampleDSZ_MissingData: Sampling the dictionary D, the binary indicating matrix Z, and the pseudo weight matrix S. Used when there are missing data.
SamplePi.m: Sampling Pi.
Samplephi.m: Sampling the noise precision phi.
Samplealpha: Sampling alpha, the precision of si.

Subprograms for the updates in sequential learning
XUpdate.m: Update the training data set in sequential learning. Used for no missing data case.
XUpdate_MissingData.m: Update the training data set in sequential learning. Used when there are missing data.
SZUpdate.m: Update the pseudo weight matrix S and binary indicating matrix Z in sequential learning.
idexUpdate.m: Update the index of the training data in the input data matrix X.
YfalgUpdate.m: Update the binary mask matrix Yflag associated with the input data matrix X.

Other subprograms:
sparsity.m: Squeeze out zero components in the sparse matrix.
DispDictionary.m: Display the dictionary elements as a image
DenoiseOutput.m: Reconstruct the image
InitMatrix.m: Initialization