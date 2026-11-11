# Related Work: Stress-Testing Structure-from-Motion

CS 5672, Fall 2026. Literature review covering the five experiment groups. Pipelines under test: COLMAP (incremental SfM) [Schonberger16], GLOMAP (global SfM) [Pan24], and ALIKED+LightGlue [Zhao23, Lindenberger23] with SIFT as the handcrafted baseline. Every paper cited below was found and checked during this review. Bracketed tags point to the reference list at the end.

## 0. Evaluation conventions (Image Matching Challenge)

- **Jin et al., "Image Matching Across Wide Baselines: From Paper to Practice," IJCV 2021** [Jin21]. This paper defines the **mean Average Accuracy (mAA)** used by the Image Matching Challenge (IMC). For each co-visible pair, it takes the angular errors of the estimated rotation and translation and computes accuracy at a range of thresholds. mAA is the area under that accuracy curve up to 10 degrees. Errors above 10 degrees count as failures, and in the multiview task any pair that involves an unregistered image gets infinite error. The paper also finds that a well-tuned RootSIFT stays within about 5% (relative) of the best multiview method at 8k features, and that tuning RANSAC and the ratio test can change method rankings.
- **Bonilla et al., "Mismatched," arXiv 2024** [Bonilla24]. The authors run COLMAP on IMC24 and warn that mAA can mislead when few images register, because the score then mostly reflects how many images registered rather than how accurate the poses are.

**Our convention.** We report pose AUC/mAA at 5, 10 and 20 degrees, following Jin et al., and the fraction of images registered as a separate number. We count unregistered images as infinite error.

## 1. Weather, illumination, season, and desert or low-texture imagery

- **Sattler et al., CVPR 2018** [Sattler18]. Introduced Aachen Day-Night, RobotCar Seasons and CMU Seasons. Night queries against a daytime map were the hardest case. On CMU Seasons, foliage roughly halved Active Search's high-precision recall (about 29% vs 53%). Daytime weather changes such as rain, snow and overcast skies mattered little in urban scenes. Dawn and direct sun were harder because of overexposure.
- **Toft et al., TPAMI 2020** [Toft20]. An extended version of the long-term benchmark with more methods. It confirms that changes in illumination and season are still the main failure mode.
- **Wenzel et al., 4Seasons, IJCV 2024/25** [Wenzel24]. Over 300 km of driving in Munich across seasons. Visual odometry and SLAM degrade sharply at night. Sunny scenes did better than cloudy ones because shadows add static texture. Accurate pixel-level matches across seasons remain an open problem.
- **Burnett et al., Boreas, IJRR 2023 (arXiv 2203.10168)** [Burnett23]. One year of repeated driving on the same route, including rain and snow, with camera, lidar and radar plus centimetre-accurate ground truth. It provides the multi-season imagery our weather-stratified splits draw on.
- **Sansoterra et al., ISPRS Archives 2026** [Sansoterra26]. Arctic UAV imagery over snow. On mixed terrain, SIFT with nearest-neighbour matching was the most complete method. On a homogeneous snow subset, SIFT registered only 2 of 56 images, against SuperPoint 8 and DISK 6. Uniform snow and dense canopy defeated every method.
- **Grohmann et al., Computers & Geosciences 2020** [Grohmann20]. SfM-MVS over aeolian dunes reached 0.08 m RMSE against terrestrial lidar. Matching only worked thanks to incidental texture (footprints, sandboard tracks, mineral layering), and to damp sand under diffuse light.
- **Liu et al., DeMatchNet, Electronics 2025** [Liu25]. Haze degrades LoFTR matching, and the paper improves it by fusing dehazed and hazy features. The authors say explicitly that sandstorms were not evaluated.
- **Li et al., Dehazing-NeRF, arXiv 2023** [Li23]. The method depends on COLMAP poses and lists as a limitation that COLMAP "may fail when the haze is very severe."

**Gap our experiment fills.** Existing benchmarks evaluate *localization* against a fixed map, and stratify by condition only at the query level. None of them runs full SfM (both COLMAP and GLOMAP) on collections that are each made of a single condition. None covers desert scenes, sandstorms or heat haze. The desert photogrammetry work is limited to geomorphology case studies, and the dehazing work stops at matching or NeRF.

**Hypotheses from the literature.**
- *Why it may fail:* Night images, uniform sand or snow, and haze or dust (which lowers contrast) all reduce the number of repeatable keypoints. Those are the failure modes reported in [Sattler18, Sansoterra26, Li23]. Heat haze also adds local warping that breaks the epipolar constraint.
- *Why it may work:* Daytime rain, snow and overcast conditions barely hurt urban localization [Sattler18]. Diffuse light and surface marks can keep sand matchable [Grohmann20]. Learned features may do better on low-texture subsets [Sansoterra26].

## 2. Robustness to image corruptions and generative edits

- **Hendrycks & Dietterich, ICLR 2019** [Hendrycks19]. Defines ImageNet-C: 15 corruptions (noise, blur, weather, digital), each at 5 severity levels. We reuse this protocol.
- **Wang et al., "Benchmarking Robustness in Neural Radiance Fields," CVPRW 2024** [Wang24]. LLFF-C and Blender-C. Fog and blur (glass, motion, defocus) damage NeRF the most, while JPEG and pixelation are mild. This ranking differs from the one for classifiers. COLMAP was barely affected with 10% of images corrupted, but registration and observations per image fell quickly at higher ratios.
- **Liu et al., RealX3D, arXiv 2025/26** [LiuS25]. A benchmark with physically degraded captures. COLMAP fails on the degraded views, so the authors took poses from clean views. Blur hurts pose accuracy the most. Feed-forward models such as VGGT lose only about 4% AUC@5.
- **Schmalfuss et al., RobustSpring, arXiv 2025** [Schmalfuss25]. A corruption benchmark in the ImageNet-C style for optical flow, scene flow and stereo. It shows that dense-correspondence models also lose accuracy under corruptions.
- **Haque et al., Instruct-NeRF2NeRF, ICCV 2023** [Haque23]. Editing each view independently with a 2D diffusion model gives results that are inconsistent across views. The method needs iterative dataset updates to converge to one 3D-consistent scene.
- **Asim et al., MEt3R, CVPR 2025** [Asim25]. Measures multi-view consistency of generated images with DUSt3R warping. Many generative novel-view-synthesis methods are measurably inconsistent across views.
- **Paul et al., "Can These Views Be One Scene?", arXiv 2026** [Paul26]. Feed-forward 3D models produce plausible geometry even from unrelated images or noise. Metrics based on COLMAP, which treat a failed registration as inconsistency, correlated better with human judgments.

**Gap our experiment fills.** Corruption benchmarks exist for classifiers, NeRF and optical flow. None reports *SfM* outcomes (registration rate, mAA, track length) across corruption type and severity, or compares SIFT with ALIKED+LightGlue and COLMAP with GLOMAP. Work on generative edits measures consistency of *generated* views. It does not test whether AI-edited photos mixed into a real collection still register, or whether they quietly distort the reconstruction.

**Hypotheses from the literature.**
- *Fail:* Blur and fog will hurt most and JPEG least [Wang24, LiuS25]. Diffusion edits change local texture inconsistently from view to view, so edited regions should stop matching [Haque23, Asim25].
- *Work:* RANSAC and verification should reject inconsistent matches, so a few edited images may simply fail to register instead of corrupting the model [Paul26].

## 3. Super-resolution as preprocessing

- **Pashaei et al., Remote Sensing 2020** [Pashaei20]. ESRGAN applied to 4x-downsampled UAS imagery gave about 6x more keypoint correspondences and about 17x more dense points than the low-resolution images, and SfM still recovered the camera geometry.
- **Tavakoli et al., ISPRS Annals 2026** [Tavakoli26]. Real-ESRGAN fine-tuned on the target domain improved orthomosaics and meshes from UAV imagery over bicubic upsampling and the pretrained model.
- **Zhang et al., JACIII 2024** [Zhang24]. When the SR model is trained without noise, it amplifies noise. Noise-aware SR training lowered SfM mean reprojection error by up to 27% and increased the number of dense points.
- **Wang et al., NeRF-SR, ACM MM 2022 (arXiv 2112.01759)** [WangC22]. Gets super-resolved NeRF from multi-view supersampling, which avoids per-image SR.
- **Asthana et al., SplatSuRe, CVPR 2026** [Asthana26]. Applying SR to each view independently introduces multi-view inconsistencies that blur 3DGS. SplatSuRe applies SR only in undersampled regions.

**Gap our experiment fills.** The positive SR results come from photogrammetry case studies that count points. They do not measure pose accuracy against ground truth, and they do not compare against simply giving the matcher the native-resolution images. The 3D-vision work points out that GAN or diffusion SR hallucinates details that are inconsistent across views, but measures this only in rendering. We measure what SR does to *matching and poses*, and contrast GAN SR with bicubic upsampling.

**Hypotheses.** *Work:* SR raises keypoint counts and the share of matches that survive RANSAC on genuinely low-resolution inputs [Pashaei20, Zhang24]. *Fail:* hallucinated texture produces confident matches that are geometrically wrong, and noise gets amplified [Asthana26, Zhang24]. More points therefore need not mean better mAA.

## 4. Transient objects, crowds, and dynamic people

- **Snavely et al., Photo Tourism, SIGGRAPH 2006** [Snavely06]. The original SfM on internet photo collections. It relies on robust matching and verification to tolerate people and clutter.
- **Schonberger & Frahm, COLMAP, CVPR 2016** [Schonberger16]. Incremental SfM with geometric verification and robust triangulation, designed for unordered internet photos. It also supports image masks for features.
- **Martin-Brualla et al., NeRF-W, CVPR 2021** [MartinBrualla21]. Models transient objects and appearance per image on Photo Tourism scenes, using COLMAP poses, which shows that SfM still works on crowded landmarks.
- **Sabour et al., RobustNeRF, CVPR 2023** [Sabour23]. Treats distractors as outliers and removes them with a robust (trimmed) loss.
- **Sabour et al., SpotLessSplats, 2024** [Sabour24] and **Kulhanek et al., WildGaussians, NeurIPS 2024** [Kulhanek24]. Extend distractor handling to 3D Gaussian Splatting.
- **Goli et al., RoMo, ICCV 2025** [Goli25]. Moving objects break the rigidity assumption of SfM, and COLMAP often fails on dynamic videos. Masking moving regions before COLMAP reduced ATE from 0.70 to 0.40 on their Casual Motion dataset.

**Gap our experiment fills.** The NeRF and 3DGS work handles transients *after* SfM, assuming its poses are correct. RoMo masks *motion* in videos. Nobody has measured, in a controlled way, how pose accuracy and registration fall as crowd density rises in still-image collections, or whether semantic person masks help ALIKED+LightGlue and GLOMAP as much as they help SIFT+COLMAP.

**Hypotheses.** *Work:* people standing in different places in each image do not form consistent tracks, so verification rejects them [Snavely06, Schonberger16]. *Fail:* dense crowds hide the static structure, and people who stay still, or the same crowd seen in a burst, create consistent but non-rigid tracks [Goli25]. Global rotation averaging in GLOMAP may be more sensitive to such wrong edges in the view graph than incremental registration is.

## 5. Extras

**(a) Partial contamination.** Wang et al. [Wang24] is the only direct evidence we found. COLMAP tolerates about 10% corrupted views, while registration drops steeply at higher ratios and severities. RealX3D [LiuS25] reports that COLMAP fails outright when all views are degraded. *Gap:* nobody has located the threshold where SfM breaks for each corruption, nor compared incremental and global SfM on it. *Hypothesis:* incremental SfM can leave out bad images (registration rate falls but accuracy holds). Global SfM solves for all cameras jointly, so it may spread the errors (accuracy falls).

**(b) Handcrafted vs learned features; incremental vs global SfM.**
- SIFT that is well tuned stays competitive on clean images [Jin21, Schonberger17].
- Out of domain, ALIKED+LightGlue registered 64% of IMC24 images against 22.5% for SIFT+LightGlue, with mAA 0.148 vs 0.026 [Bonilla24].
- Hierarchical localization with SuperPoint was more robust than SIFT for day-night localization [Sarlin19]. LightGlue [Lindenberger23] and ALIKED [Zhao23] report accuracy and speed above SuperGlue/SuperPoint. DISK [Tyszkiewicz20] is another learned baseline.
- Arctic snow shows the reverse: SIFT was best on mixed terrain [Sansoterra26].
- GLOMAP matches or beats COLMAP's accuracy and robustness on standard benchmarks while running orders of magnitude faster [Pan24]. No paper we found evaluates it under image degradations.

*Gap:* we found no factorial comparison of feature extractor x SfM backend under controlled corruptions.

**(c) Cross-season localization.** Benchmarks that localize against a map built in different conditions [Sattler18, Toft20, Wenzel24] show the largest losses for night and for vegetation changes. Hierarchical localization with learned features narrows the gap [Sarlin19]. *Gap:* these studies use one reference map built with SIFT, typically in daytime summer. We vary which condition the map is built in, as well as the feature extractor, and test with both COLMAP's image registrator and the GLOMAP pipeline.

## References

```
[Asim25]     Asim, Wewer, Wimmer, Schiele, Lenssen. MEt3R: Measuring Multi-View Consistency in Generated Images. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Asim_MET3R_Measuring_Multi-View_Consistency_in_Generated_Images_CVPR_2025_paper.html
[Asthana26]  Asthana, Hanson, Tu, Goldstein, Zwicker, Varshney. SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting. CVPR 2026. https://cvpr.thecvf.com/virtual/2026/poster/36956
[Bonilla24]  Bonilla, Di Vece, Daher, Ju, Stoyanov, Vasconcelos, Bano. Mismatched: Evaluating the Limits of Image Matching Approaches and Benchmarks. arXiv:2408.16445, 2024. https://arxiv.org/abs/2408.16445
[Burnett23]  Burnett, Yoon, Wu, et al., Barfoot. Boreas: A Multi-Season Autonomous Driving Dataset. IJRR 2023 / arXiv:2203.10168. https://arxiv.org/abs/2203.10168
[Goli25]     Goli, Sabour, Matthews, Brubaker, Lagun, Jacobson, Fleet, Saxena, Tagliasacchi. RoMo: Robust Motion Segmentation Improves Structure from Motion. ICCV 2025. https://arxiv.org/abs/2411.18650
[Grohmann20] Grohmann, Garcia, Affonso, Albuquerque. Aeolian dune modelling from airborne LiDAR, terrestrial LiDAR and SfM-MVS. Computers & Geosciences 143:104569, 2020. https://arxiv.org/abs/1910.06186
[Haque23]    Haque, Tancik, Efros, Holynski, Kanazawa. Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions. ICCV 2023. https://arxiv.org/abs/2303.12789
[Hendrycks19] Hendrycks, Dietterich. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR 2019. https://arxiv.org/abs/1903.12261
[Jin21]      Jin, Mishkin, Mishchuk, Matas, Fua, Yi, Trulls. Image Matching across Wide Baselines: From Paper to Practice. IJCV 129, 2021. https://arxiv.org/abs/2003.01587
[Kulhanek24] Kulhanek, Peng, Kukelova, Pollefeys, Sattler. WildGaussians: 3D Gaussian Splatting in the Wild. NeurIPS 2024. https://arxiv.org/abs/2407.08447
[Li23]       Li, Li, Wang, Feng. Dehazing-NeRF: Neural Radiance Fields from Hazy Images. arXiv:2304.11448, 2023. https://arxiv.org/abs/2304.11448
[Lindenberger23] Lindenberger, Sarlin, Pollefeys. LightGlue: Local Feature Matching at Light Speed. ICCV 2023. https://arxiv.org/abs/2306.13643
[Liu25]      Liu, Zhang, He, Liu, Hu, Liu. DeMatchNet: Joint Dehazing and Feature Matching in Adverse Weather. Electronics 14(5):940, 2025. https://www.mdpi.com/2079-9292/14/5/940
[LiuS25]     Liu, Bao, Cui, et al., Harada. RealX3D: A Physically-Degraded 3D Benchmark for Multi-view Visual Restoration and Reconstruction. arXiv:2512.23437, 2025. https://arxiv.org/abs/2512.23437
[MartinBrualla21] Martin-Brualla, Radwan, Sajjadi, Barron, Dosovitskiy, Duckworth. NeRF in the Wild. CVPR 2021. https://arxiv.org/abs/2008.02268
[Pan24]      Pan, Barath, Pollefeys, Schonberger. Global Structure-from-Motion Revisited (GLOMAP). ECCV 2024. https://arxiv.org/abs/2407.20219
[Pashaei20]  Pashaei, Starek, Kamangir, Berryhill. Deep Learning-Based Single Image Super-Resolution: An Investigation for Dense Scene Reconstruction with UAS Photogrammetry. Remote Sensing 12(11):1757, 2020. https://www.mdpi.com/2072-4292/12/11/1757
[Paul26]     Paul, Kaushik, Yuille. Can These Views Be One Scene? Evaluating Multiview 3D Consistency when 3D Foundation Models Hallucinate. arXiv:2605.18754, 2026. https://arxiv.org/abs/2605.18754
[Sabour23]   Sabour, Vora, Duckworth, Krasin, Fleet, Tagliasacchi. RobustNeRF: Ignoring Distractors with Robust Losses. CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Sabour_RobustNeRF_Ignoring_Distractors_With_Robust_Losses_CVPR_2023_paper.html
[Sabour24]   Sabour, Goli, Kopanas, Matthews, Lagun, Guibas, Jacobson, Fleet, Tagliasacchi. SpotLessSplats: Ignoring Distractors in 3D Gaussian Splatting. arXiv:2406.20055, 2024. https://arxiv.org/abs/2406.20055
[Sansoterra26] Sansoterra, Lenzano, Shuart, Anderson, Yilmaz, Toth. Evaluating Classical and Deep Keypoint Detectors for SfM Reconstruction in Arctic UAV Imagery. ISPRS Archives XLIX-B2-2026, 545. https://isprs-archives.copernicus.org/articles/XLIX-B2-2026/545/2026/
[Sarlin19]   Sarlin, Cadena, Siegwart, Dymczyk. From Coarse to Fine: Robust Hierarchical Localization at Large Scale. CVPR 2019. https://arxiv.org/abs/1812.03506
[Sattler18]  Sattler, Maddern, Toft, Torii, Hammarstrand, Stenborg, Safari, Okutomi, Pollefeys, Sivic, Kahl, Pajdla. Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions. CVPR 2018. https://arxiv.org/abs/1707.09092
[Schmalfuss25] Schmalfuss et al. RobustSpring: Benchmarking Robustness to Image Corruptions for Optical Flow, Scene Flow and Stereo. arXiv:2505.09368, 2025. https://arxiv.org/abs/2505.09368
[Schonberger16] Schonberger, Frahm. Structure-from-Motion Revisited. CVPR 2016. https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf
[Schonberger17] Schonberger, Hardmeier, Sattler, Pollefeys. Comparative Evaluation of Hand-Crafted and Learned Local Features. CVPR 2017. https://openaccess.thecvf.com/content_cvpr_2017/html/Schonberger_Comparative_Evaluation_of_CVPR_2017_paper.html
[Snavely06]  Snavely, Seitz, Szeliski. Photo Tourism: Exploring Photo Collections in 3D. SIGGRAPH (ACM TOG 25(3)) 2006. https://phototour.cs.washington.edu/
[Tavakoli26] Tavakoli, Eftekhari, SaadatSeresht, Jamshidpour. Enhancing Urban UAV Photogrammetric Products Through Domain-Specific Training of Real-ESRGAN. ISPRS Annals X-4/W8-2025, 785-792, 2026. https://isprs-annals.copernicus.org/articles/X-4-W8-2025/785/2026/
[Toft20]     Toft, Maddern, Torii, et al., Kahl, Sattler. Long-Term Visual Localization Revisited. IEEE TPAMI 2020. https://research.chalmers.se/publication/529149
[Tyszkiewicz20] Tyszkiewicz, Fua, Trulls. DISK: Learning Local Features with Policy Gradient. NeurIPS 2020. https://arxiv.org/abs/2006.13566
[WangC22]    Wang, Wu, Guo, Zhang, Tai, Hu. NeRF-SR: High Quality Neural Radiance Fields using Supersampling. ACM MM 2022. https://arxiv.org/abs/2112.01759
[Wang24]     Wang, Wang, Li, Yuille, Xie. Benchmarking Robustness in Neural Radiance Fields. CVPR Workshops (AdvML) 2024. https://arxiv.org/abs/2301.04075
[Wenzel24]   Wenzel, Yang, Wang, Zeller, Cremers, et al. 4Seasons: Benchmarking Visual SLAM and Long-Term Localization for Autonomous Driving in Challenging Conditions. IJCV 133(4), 2025 (online 2024). https://link.springer.com/article/10.1007/s11263-024-02230-4
[Zhang24]    Zhang, Nobuhara, Haris. Adding Noise to Super-Resolution Training Set: Method to Denoise Super Resolution for Structure from Motion Preprocessing. JACIII 28(6):1284-1298, 2024. https://www.jstage.jst.go.jp/article/jaciii/28/6/28_1284/_article/-char/ja
[Zhao23]     Zhao, Wu, Liu, Chen, Wen. ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation. IEEE TIM 2023. https://arxiv.org/abs/2304.03608
```
