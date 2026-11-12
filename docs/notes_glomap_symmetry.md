# Why GLOMAP scores low on clean Gerrard Hall

Clean Gerrard Hall (50 images, 1024 px, SIFT, exhaustive matching): COLMAP incremental AUC@10 = 0.97, while GLOMAP
(`pycolmap.global_mapping`) gets 0.57, with all 50 images registered and a mean reprojection error of 0.35 px.

Per-image median relative-rotation error against the reference model:
* 37 images: 0.4° to 1.2°
* 12 consecutive images (one side of the building): about 135°, plus a few at 29° to 174°

So one contiguous block of views is consistently rotated. That is the signature of a symmetric/repetitive-facade
ambiguity: wrong but geometrically verified two-view matches between similar-looking facades survive into global
rotation averaging. Incremental SfM avoids this because each new image must agree with the existing 3D points.

This is a property of the method, not a bug in our wrapper. We report it as a result of E6.
