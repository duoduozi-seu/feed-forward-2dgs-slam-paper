# `root.tex` 实际引用 BibTeX 核对表

核对日期：2026-05-23  
核对范围：仅核对 `root.tex` 实际引用并进入 `root.bbl` 的 35 个条目。当前 `references.bib` 中还有 52 个未被 `root.tex` 引用的条目，本次未清理。

| key | 权威来源 | 结论 | 已处理/建议 |
|---|---|---|---|
| `tosi2024nerfs` | [IEEE/Crossref DOI](https://doi.org/10.1109/TRO.2026.3666139) | 高优先级不一致：已从 arXiv 预印本正式发表到 T-RO 2026。 | 已更新为 `IEEE Transactions on Robotics`, vol. 42, pp. 1405--1427, 2026，并加 DOI。 |
| `sucar2021imap` | [IEEE/Crossref DOI](https://doi.org/10.1109/ICCV48922.2021.00617) | 高优先级不一致：页码原为 6229--6238，正式页码为 6209--6218。 | 已修正页码并保护 `iMAP` 大小写。 |
| `zhu2022nice` | [IEEE/Crossref DOI](https://doi.org/10.1109/CVPR52688.2022.01245) | 高优先级不一致：页码原为 12786--12796，正式页码为 12776--12786。 | 已修正页码并保护 `NICE-SLAM`/`SLAM`。 |
| `wang2023co` | [IEEE/Crossref DOI](https://doi.org/10.1109/CVPR52729.2023.01277) | 主要字段一致。 | 已补充标题大小写保护。 |
| `yugay2023gaussian` | [arXiv:2312.10070](https://arxiv.org/abs/2312.10070) | arXiv 元数据一致。 | 已补充 `Gaussian-SLAM`/`SLAM`/`Gaussian Splatting` 大小写保护。 |
| `wu2024mm` | [IEEE/Crossref DOI](https://doi.org/10.1109/IROS58592.2024.10801605) | 主要字段一致。 | 已补充标题大小写保护。 |
| `zhao2024tclc` | [arXiv:2404.02410](https://arxiv.org/abs/2404.02410) | arXiv 元数据一致。 | 已补充 `TCLC-GS`、`LiDAR-Camera` 等大小写保护。 |
| `lin2024r` | [IEEE/Crossref DOI](https://doi.org/10.1109/TPAMI.2024.3456473) | 中高优先级不一致：缺少卷号、期号、页码和 DOI。 | 已补全 TPAMI vol. 46, no. 12, pp. 11168--11185。 |
| `zheng2024fast` | [IEEE/Crossref DOI](https://doi.org/10.1109/TRO.2024.3502198) | 高优先级不一致：正式版为 T-RO 2025，作者列表和页码缺失。 | 已更新完整作者、vol. 41, pp. 326--346, 2025，并加 DOI。 |
| `lang2023coco` | [IEEE/Crossref DOI](https://doi.org/10.1109/LRA.2023.3315542) | 主要字段一致，缺 DOI。 | 已补 DOI，并保护标题大小写。 |
| `kerbl20233d` | [ACM DOI](https://doi.org/10.1145/3592433) | 中优先级不一致：缺页码和 DOI。 | 已补 pp. 1--14 和 DOI，并保护 `3D Gaussian Splatting`。 |
| `huang20242d` | [ACM DOI](https://doi.org/10.1145/3641519.3657428) | 主要字段一致，缺 DOI，标题大小写未保护。 | 已补 DOI 并保护 `2D Gaussian Splatting`。 |
| `mildenhall2021nerf` | [Springer DOI](https://doi.org/10.1007/978-3-030-58452-8_24) | 主要字段一致。 | 已保护 `NeRF` 大小写。 |
| `dai2024high` | [ACM DOI](https://doi.org/10.1145/3641519.3657441) | 主要字段一致，缺 DOI，标题大小写未保护。 | 已补 DOI 并保护标题大小写。 |
| `chen2024pgsr` | [IEEE/Crossref DOI](https://doi.org/10.1109/TVCG.2024.3494046) | 高优先级不一致：正式版为 TVCG 2025，缺卷期页码。 | 已更新 vol. 31, no. 9, pp. 6100--6111, 2025，并加 DOI。 |
| `jiang2024li` | [IEEE/Crossref DOI](https://doi.org/10.1109/LRA.2024.3522846) | 高优先级不一致：正式版为 RA-L 2025，缺卷期页码。 | 已更新 vol. 10, no. 2, pp. 1864--1871, 2025，并加 DOI。 |
| `charatan2024pixelsplat` | [IEEE/Crossref DOI](https://doi.org/10.1109/CVPR52733.2024.01840) | 主要字段一致。 | 已保护 `pixelSplat`、`3D` 等标题大小写。 |
| `chen2024mvsplat` | [项目页 BibTeX](https://donydchen.github.io/mvsplat/) / [Springer DOI](https://doi.org/10.1007/978-3-031-72664-4_21) | 项目页 BibTeX 仍给 arXiv；本地条目使用正式 ECCV/Springer 信息。 | 未降级为 arXiv；仅保护标题大小写。 |
| `liu2025mvsgaussian` | [项目页 BibTeX](https://mvsgaussian.github.io/) | 与项目页 BibTeX 的作者、venue、页码、年份一致。 | 仅补充标题大小写保护。 |
| `matsuki2024gaussian` | [CVPR OpenAccess](https://openaccess.thecvf.com/content/CVPR2024/html/Matsuki_Gaussian_Splatting_SLAM_CVPR_2024_paper.html) | 主要字段一致，作者缩写格式略不规范。 | 已改为 `Paul H. J. Kelly` 并保护标题大小写。 |
| `yan2024gs` | [IEEE/Crossref DOI](https://doi.org/10.1109/CVPR52733.2024.01853) | 主要字段一致。 | 已保护 `GS-SLAM`、`SLAM`、`3D Gaussian Splatting`。 |
| `keetha2024splatam` | [IEEE/Crossref DOI](https://doi.org/10.1109/CVPR52733.2024.02018) | 低优先级不一致：标题缺逗号。 | 已按正式标题改为 `Splat, Track \& Map` 并保护缩写。 |
| `huang2024photo` | [IEEE/Crossref DOI](https://doi.org/10.1109/CVPR52733.2024.02039) | 低优先级不一致：正式标题含 Monocular, Stereo, and RGB-D 的逗号结构。 | 已修正标题并保护缩写。 |
| `hong2024liv` | [IEEE/Crossref DOI](https://doi.org/10.1109/LRA.2024.3400149) | 高优先级不一致：本地多列了 `Shaojie Shen`，且缺卷期页码。 | 已按正式版改为 4 位作者，补 vol. 9, no. 11, pp. 9765--9772 和 DOI。 |
| `lang2024gaussian` | [IEEE/Crossref DOI](https://doi.org/10.1109/ICRA55743.2025.11128712) | 高优先级不一致：已由 arXiv 预印本进入 ICRA 2025。 | 已改为 `@inproceedings`，补 ICRA 2025、pp. 8500--8507 和 DOI。 |
| `xie2024gs` | [IEEE/Crossref DOI](https://doi.org/10.1109/ICCV51701.2025.02494) | 高优先级不一致：已由 arXiv 预印本进入 ICCV 2025。 | 已改为 `@inproceedings`，补 ICCV 2025 和 pp. 26869--26878。 |
| `xiao2024liv` | [IEEE/Crossref DOI](https://doi.org/10.1109/LRA.2024.3505777) | 高优先级不一致：正式版为 RA-L 2025，缺卷期页码。 | 已更新 vol. 10, no. 1, pp. 421--428, 2025，并加 DOI。 |
| `zhao2025lvi` | [IEEE/Crossref DOI](https://doi.org/10.1109/TIM.2025.3551585) | 中高优先级不一致：缺卷号和页码。 | 已补 IEEE TIM vol. 74, pp. 1--10 和 DOI。 |
| `hong2025gs` | [IEEE/Crossref DOI](https://doi.org/10.1109/TRO.2025.3582809) | 高优先级不一致：已由 arXiv 预印本进入 T-RO 2025。 | 已更新 T-RO vol. 41, pp. 4253--4268 和 DOI。 |
| `lang2025gaussianlic2` | [arXiv:2507.04004](https://arxiv.org/abs/2507.04004) | arXiv 元数据一致。 | 已补充标题大小写保护。 |
| `lin2022r` | [IEEE/Crossref DOI](https://doi.org/10.1109/ICRA46639.2022.9811935) | 主要字段一致，但标题中的 `R3LIVE` 被写成 `R 3 LIVE`。 | 已修正为 `R\textsuperscript{3}LIVE` 并保护大小写。 |
| `li2023spnet` | [PLOS DOI](https://doi.org/10.1371/journal.pone.0280886) | 主要字段一致。 | 已修正 `PLOS ONE` 和标题大小写。 |
| `lin2025da3` | [arXiv:2511.10647](https://arxiv.org/abs/2511.10647) | arXiv 元数据一致。 | 已保护 `Depth Anything 3` 标题大小写。 |
| `sun2024mm3dgs` | [IEEE/Crossref DOI](https://doi.org/10.1109/IROS58592.2024.10802389) | 主要字段一致。 | 已保护 `MM3DGS SLAM`、`3D`、`SLAM` 等大小写。 |
| `nguyen2024mcd` | [MCD 项目页 BibTeX](https://mcdviral.github.io/) / [CVPR OpenAccess](https://openaccess.thecvf.com/content/CVPR2024/html/Nguyen_MCD_Diverse_Large-Scale_Multi-Campus_Dataset_for_Robot_Perception_CVPR_2024_paper.html) | 中优先级不一致：本地用 `and others` 省略了最后作者。 | 已改为完整 11 位作者，并保护 `MCD`。 |

## 备注

- 对已正式出版的条目，优先采用 IEEE/ACM/Springer/CVF 的正式出版信息。
- 对仍未发现正式出版信息的条目，保留 arXiv 预印本形式。
- `chen2024mvsplat` 的项目页 BibTeX 仍是 arXiv，但 Crossref/Springer 已有 ECCV 章节信息；为避免倒退，本次保留正式 ECCV/Springer 条目。
