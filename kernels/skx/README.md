What's New
------------

 * **Support AVX512 level-3/1f/1 kernels on SkylakeX** 
 * **Unified GEMM/GEMMSUP for better performance on small and medium problem sizes** 

Key Features
------------

 * **Edges self-adaptive**: Leverages macro templates to make it much easier to code up the numerous edge case kernels that arise for the SkylakeX's larger microtile;

 * **From inline ASM to instrinsics**: In order to accommodate the complex process logic caused by edges self-adaptive, coded the kernels with the instruction instead of inline ASM. Additional performance losses are avoided by following some necessary compiler efficiency best practices;

 * **Unified GEMM/GEMMSUP for for better performance on small and medium problem sizes**  by introducing matrix-shape conditioned cntl tree to achieve the flexibility of unpacking/packing A or B individually controlled by BLIS-MT and BLIS-NT. Additional configure --enable-fup --disable-fup (default) were applied to enable/disable this feature. 
 
     Basing on the observation by force A or B individually in SUP mode ( instead of both) and there should be performance positive contributions at different range for supa and supb whose upperlimit could be defined as MT and NT (showed in the chart labeled “blis-gemm-supa” and “blis-gemm-supb”). Since normally NT>3MT as per the experience on SkylakeX, we could got a better performance named with “blis-gemmfup” to first run in supa+supb in [0, MT], then supb in [MT, NT], and in conventional mode in [NT, $`+\infty`$). This case is better than “blis-gemmsup” for order range [MT, NT].

Performance Benchmark
------------
<p align="center">
<img height="320px" class="center-block" src="https://cloud.realabai.com/f/17c4101e896b4544b3b7/?dl=1">
<img height="320px" class="center-block" src="https://cloud.realabai.com/f/b0b5eee69412413a83f2/?dl=1">
</p>
<p align="center">
<img height="320px" class="center-block" src="https://cloud.realabai.com/f/2172beaff82643238408/?dl=1">
<img height="320px" class="center-block" src="https://cloud.realabai.com/f/7305db86d6eb4ebe8ab4/?dl=1">
</p>
