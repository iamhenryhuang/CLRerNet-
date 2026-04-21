import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.registry import MODELS


@MODELS.register_module()
class CLRerNet(SingleStageDetector):
    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        sgm_loss_weight=0.5,  # 接收 config 的 0.5
        sgm_dark_threshold=0.45,
        contrastive_loss_weight=0.1,
        contrastive_temperature=0.1,
        zero_dce=None,
        zero_dce_gamma=0.7,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        """CLRerNet detector."""
        super(CLRerNet, self).__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, data_preprocessor, init_cfg
        )
        self.sgm_loss_weight = float(sgm_loss_weight)  # 儲存權重
        self.sgm_dark_threshold = float(sgm_dark_threshold)
        self.contrastive_loss_weight = float(contrastive_loss_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.zero_dce = MODELS.build(zero_dce) if zero_dce is not None else None
        self.zero_dce_gamma = float(zero_dce_gamma)

    def _to_unit_range(self, img):
        # data_preprocessor uses std=[255,255,255], so inputs are already in [0,1].
        # We keep a `scaled` flag in case the legacy forward_train path ever passes
        # raw [0,255] tensors, but the threshold is intentionally conservative.
        x = img.float()
        scaled = bool(x.max() > 2.0)   # only True for genuine [0,255] inputs
        if scaled:
            x = x / 255.0
        return x.clamp(0.0, 1.0), scaled

    def _from_unit_range(self, x, scaled):
        x = x.clamp(0.0, 1.0)
        if scaled:
            x = x * 255.0
        return x

    def _apply_zero_dce(self, img, night_indices):
        """Apply ZeroDCE brightening only to night-scene images.

        Args:
            img: Image tensor (N, C, H, W).
            night_indices: List of batch indices that are night scenes.
        """
        if not night_indices:
            return img

        night_idx = list(night_indices)
        x, scaled = self._to_unit_range(img)

        # Only run ZeroDCE on night frames (avoid wasted compute on day frames)
        x_night = x[night_idx]
        # Dataset/preprocessor keep BGR order; ZeroDCE checkpoints expect RGB.
        x_night_rgb = x_night[:, [2, 1, 0], :, :]

        if self.zero_dce is None:
            # Fallback: gamma brightening
            enhanced_rgb = torch.pow(x_night_rgb.clamp(min=1e-6), self.zero_dce_gamma)
        else:
            module = self.zero_dce
            if not getattr(module, 'requires_grad', False):
                module.eval()
                with torch.no_grad():
                    enhanced_rgb = module(x_night_rgb)
            else:
                enhanced_rgb = module(x_night_rgb)
            # Apply gamma on top of ZeroDCE to strengthen brightening effect
            enhanced_rgb = torch.pow(enhanced_rgb.clamp(min=1e-6), self.zero_dce_gamma)

        # Convert back to BGR
        enhanced_night = self._from_unit_range(enhanced_rgb[:, [2, 1, 0], :, :], scaled)

        # All night → return directly; mixed → clone and fill night slots
        if len(night_idx) == img.size(0):
            return enhanced_night

        result = img.clone()
        result[night_idx] = enhanced_night
        return result

    def _apply_lane_enhance_positive(self, batch_inputs, batch_data_samples, night_indices):
        """Build positive samples by locally enhancing GT lane regions.

        The lane mask is used only as a soft blending weight. Image processing is
        applied to the original image, then mixed back only around GT lanes.
        """
        if not night_indices:
            return batch_inputs

        x_unit, scaled = self._to_unit_range(batch_inputs)

        masks = self._create_lane_mask(batch_inputs, batch_data_samples, night_indices)
        # Expand lane support, then smooth it into a soft alpha matte to avoid
        # hard mask artifacts becoming the contrastive shortcut.
        lane_alpha = F.max_pool2d(masks, kernel_size=15, stride=1, padding=7)
        lane_alpha = F.avg_pool2d(lane_alpha, kernel_size=11, stride=1, padding=5)
        lane_alpha = lane_alpha.clamp(0.0, 1.0).to(dtype=x_unit.dtype)

        if lane_alpha.sum() <= 0:
            return batch_inputs

        # Unsharp mask plus a stronger brightness boost so GT lane pixels are
        # visibly clearer while still avoiding synthetic neon lanes.
        blur = F.avg_pool2d(x_unit, kernel_size=5, stride=1, padding=2)
        sharp = x_unit + 1.2 * (x_unit - blur)
        enhanced = (sharp + 0.18).clamp(0.0, 1.0)

        positive = x_unit * (1.0 - lane_alpha) + enhanced * lane_alpha
        return self._from_unit_range(positive, scaled).to(dtype=batch_inputs.dtype)

    def _create_lane_mask(self, batch_inputs, batch_data_samples, night_indices=None):
        """Create lane masks from GT annotations, only for night-scene images.

        Args:
            batch_inputs: Image tensor (N, C, H, W)
            batch_data_samples: List of data samples with GT information
            night_indices: List of batch indices that are night scenes. If None,
                           masks are built for all images.

        Returns:
            torch.Tensor: Lane masks (N, 1, H, W), value=1 for lane, 0 for background
        """
        B, _, H, W = batch_inputs.shape
        masks = torch.zeros(B, 1, H, W, dtype=torch.float32, device=batch_inputs.device)

        indices = night_indices if night_indices is not None else range(B)

        for b_idx in indices:
            data_sample = batch_data_samples[b_idx]
            gt_points = None

            # Method 1: metainfo
            if hasattr(data_sample, 'metainfo') and isinstance(data_sample.metainfo, dict):
                gt_points = data_sample.metainfo.get('gt_points', None)

            # Method 2: direct attribute
            if gt_points is None and hasattr(data_sample, 'gt_points'):
                gt_points = data_sample.gt_points

            # Method 3: gt_instances
            if gt_points is None and hasattr(data_sample, 'gt_instances'):
                gt_inst = data_sample.gt_instances
                if hasattr(gt_inst, 'points'):
                    gt_points = gt_inst.points

            if gt_points is not None and len(gt_points) > 0:
                mask_np = np.zeros((H, W), dtype=np.uint8)

                for lane_points in gt_points:
                    if isinstance(lane_points, (list, tuple)):
                        points = []
                        for i in range(0, len(lane_points), 2):
                            if i + 1 < len(lane_points):
                                x = int(np.clip(lane_points[i], 0, W - 1))
                                y = int(np.clip(lane_points[i + 1], 0, H - 1))
                                points.append([x, y])
                        if len(points) >= 2:
                            points_arr = np.array(points, dtype=np.int32)
                            cv2.polylines(mask_np, [points_arr], False, 1, thickness=8)
                    elif isinstance(lane_points, torch.Tensor):
                        points_np = lane_points.cpu().numpy()
                        if points_np.ndim == 1:
                            points_np = points_np.reshape(-1, 2)
                        if points_np.ndim == 2 and points_np.shape[1] == 2:
                            points_arr = np.clip(points_np, [0, 0], [W - 1, H - 1]).astype(np.int32)
                            cv2.polylines(mask_np, [points_arr], False, 1, thickness=8)

                masks[b_idx, 0, :, :] = torch.from_numpy(mask_np).float()

        return masks

    def _apply_inpaint_negative(self, batch_inputs, batch_data_samples, night_indices):
        """Generate negative samples by erasing lanes and inpainting.

        Only night-scene images (given by night_indices) have their GT lanes
        removed; other images are returned unchanged.

        Args:
            batch_inputs: Image tensor (N, C, H, W)
            batch_data_samples: List of data samples with GT information
            night_indices: List of batch indices that are night scenes

        Returns:
            torch.Tensor: Inpainted image tensor (N, C, H, W)
        """
        x_unit, scaled = self._to_unit_range(batch_inputs)

        # Build lane masks only for night frames
        masks = self._create_lane_mask(batch_inputs, batch_data_samples, night_indices)

        img_np = x_unit.detach().cpu().numpy()   # (N, C, H, W), float in [0, 1]
        masks_np = masks.detach().cpu().numpy()  # (N, 1, H, W)

        inpainted = img_np.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        for b in night_indices:
            img_hwc = np.transpose(img_np[b], (1, 2, 0))
            img_hwc_u8 = np.clip(np.round(img_hwc * 255.0), 0, 255).astype(np.uint8)
            mask_hw = (masks_np[b, 0] > 0.5).astype(np.uint8)

            mask_dilated = cv2.dilate(mask_hw, kernel, iterations=1)
            inpainted_hwc = cv2.inpaint(img_hwc_u8, mask_dilated, 7, cv2.INPAINT_TELEA)
            inpainted[b] = np.transpose(inpainted_hwc.astype(np.float32) / 255.0, (2, 0, 1))

        out = torch.from_numpy(inpainted).to(dtype=batch_inputs.dtype, device=batch_inputs.device)
        return self._from_unit_range(out, scaled)

    def _compute_contrastive_loss(self, anchor_proj, pos_proj, neg_proj):
        """Compute contrastive loss.

        Caller must guarantee that all three projection tuples already contain
        only night samples (filtering is done upstream in ``loss()``).

        Args:
            anchor_proj: Tuple of (M, D) tensors — night samples only.
            pos_proj:    Tuple of (M, D) tensors — night samples only.
            neg_proj:    Tuple of (M, D) tensors — night samples only.
        """
        if self.contrastive_loss_weight <= 0:
            return None
        if not anchor_proj or not pos_proj or not neg_proj:
            return None

        per_level_losses = []
        n_levels = min(len(anchor_proj), len(pos_proj), len(neg_proj))
        for i in range(n_levels):
            a = anchor_proj[i]
            p = pos_proj[i]
            n = neg_proj[i]
            if a is None or p is None or n is None:
                continue
            if a.shape != p.shape or a.shape != n.shape:
                continue
            if a.size(0) == 0:
                continue

            temp = max(self.contrastive_temperature, 1e-6)
            sim_ap = F.cosine_similarity(a, p, dim=1, eps=1e-8)
            sim_an = F.cosine_similarity(a, n, dim=1, eps=1e-8)
            logits = torch.stack([sim_ap / temp, sim_an / temp], dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            loss_i = F.cross_entropy(logits, labels)
            per_level_losses.append(loss_i)

        if not per_level_losses:
            return None

        loss = torch.stack(per_level_losses).mean()
        return self.contrastive_loss_weight * loss

    def _build_scene_targets(self, img, img_metas, device, dtype):
        """Build binary GT scene labels for supervising the SGM gate.

        Priority:
        1) Use ``scene_label`` from ``img_metas`` when available (supervised setting).
        2) Fallback to brightness-based pseudo labels from the input image.

        These targets are used ONLY to train the SGM gate via BCE loss.
        Do NOT use this function to decide whether to run contrastive learning;
        use ``sgm_p`` from the neck instead (see ``loss()``).
        """
        gt_scene = []
        has_meta_labels = True
        for meta in img_metas:
            if isinstance(meta, dict) and "scene_label" in meta:
                v = meta["scene_label"]
                if isinstance(v, torch.Tensor):
                    v = v.detach().float().mean().item()
                elif isinstance(v, (list, tuple)):
                    v = float(v[0]) if len(v) > 0 else 0.0
                else:
                    v = float(v)
                gt_scene.append(v)
            else:
                has_meta_labels = False
                break

        if has_meta_labels and len(gt_scene) == img.size(0):
            return torch.as_tensor(gt_scene, device=device, dtype=dtype)

        # Brightness pseudo label: dark=1 (night), bright=0 (day)
        x = img.detach().float()
        if x.max() > 2.0:   # consistent with _to_unit_range threshold
            x = x / 255.0
        mean_luma = x.mean(dim=[1, 2, 3])
        pseudo_night = (mean_luma < self.sgm_dark_threshold).to(dtype=dtype)
        return pseudo_night.to(device=device)

    def _compute_sgm_loss(self, img, img_metas, p_global):
        """Compute Scene Aware Gate BCE loss.

        Args:
            img: Raw input images (used for pseudo label generation).
            img_metas: List of image metadata dicts.
            p_global: Per-image SGM probability tensor of shape (N,), already
                      extracted from ``neck.sgm_p``. Passed in explicitly so
                      ``loss()`` can reuse the same tensor for night_mask.
        """
        if self.sgm_loss_weight <= 0:
            return None

        if p_global is None:
            # SGM module not available; skip loss
            return None

        gt_scene = self._build_scene_targets(
            img, img_metas, device=p_global.device, dtype=p_global.dtype
        )

        if gt_scene.numel() != p_global.numel():
            return None

        loss_sgm = F.binary_cross_entropy(p_global, gt_scene)
        return self.sgm_loss_weight * loss_sgm

    def loss(self, batch_inputs, batch_data_samples):
        """MMDet3 training API."""
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)

        img_metas = [data_sample.metainfo for data_sample in batch_data_samples]

        # --- Unified night_mask from SGM gate ---
        # sgm_p: [N, 1, 1, 1] sigmoid output; high value = night.
        # This is the single source of truth for both SGM loss supervision
        # and deciding whether to run contrastive learning.
        sgm_p_raw = getattr(self.neck, 'sgm_p', None)
        if sgm_p_raw is not None:
            # p_global_grad: keeps gradient for SGM BCE loss training
            p_global_grad = sgm_p_raw.squeeze([1, 2, 3])          # (N,) in [0, 1]
            # p_global_det: detached copy used only for night_mask decision
            p_global_det = p_global_grad.detach()
            night_mask = (p_global_det >= 0.5)                     # (N,) bool
        else:
            p_global_grad = None
            p_global_det = None
            night_mask = torch.zeros(batch_inputs.size(0), dtype=torch.bool,
                                     device=batch_inputs.device)

        loss_sgm = self._compute_sgm_loss(batch_inputs, img_metas, p_global_grad)
        if loss_sgm is not None:
            losses['loss_sgm'] = loss_sgm

        # --- Contrastive learning: only for night samples ---
        # Need >= 2 night samples because the projection head uses BatchNorm1d,
        # which cannot compute running stats with batch size 1 in train mode.
        if self.contrastive_loss_weight > 0 and int(night_mask.sum().item()) >= 2:
            night_indices = night_mask.nonzero(as_tuple=False).squeeze(1).tolist()

            # Anchor: slice the already-computed FPN features for night samples.
            # Keep gradients on this path so contrastive loss can regularize
            # the anchor FPN representation. Positive/negative paths below stay
            # no_grad and act as references.
            x_night_anchor = tuple(feat[night_indices] for feat in x)
            self.neck.compute_proj_feats(x_night_anchor)
            anchor_proj = tuple(self.neck.proj_feats or ())

            # Slice night sub-batch once; all subsequent ops work on M images only.
            night_inputs = batch_inputs[night_indices]
            night_data_samples = [batch_data_samples[i] for i in night_indices]
            night_idx_local = list(range(len(night_indices)))  # [0, 1, ..., M-1]

            # Positive: locally brighten/sharpen GT lane regions only.
            # Use no_grad for backbone/FPN; gradients only flow through proj heads.
            pos_inputs = self._apply_lane_enhance_positive(
                night_inputs, night_data_samples, night_idx_local
            )
            with torch.no_grad():
                x_pos = self.extract_feat(pos_inputs)
            self.neck.compute_proj_feats(x_pos)
            pos_proj = tuple(self.neck.proj_feats or ())

            # Negative: GT lanes erased + inpainted, night sub-batch only.
            neg_inputs = self._apply_inpaint_negative(
                night_inputs, night_data_samples, night_idx_local
            )
            with torch.no_grad():
                x_neg = self.extract_feat(neg_inputs)
            self.neck.compute_proj_feats(x_neg)
            neg_proj = tuple(self.neck.proj_feats or ())

            loss_contrast = self._compute_contrastive_loss(
                anchor_proj, pos_proj, neg_proj
            )
            if loss_contrast is not None:
                losses['loss_contrast'] = loss_contrast

        return losses

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas)

        # keep compatibility for legacy train path
        sgm_p_raw = getattr(self.neck, 'sgm_p', None)
        # Do NOT detach here — SGM needs gradients to train via BCE loss.
        p_global_grad = sgm_p_raw.squeeze([1, 2, 3]) if sgm_p_raw is not None else None
        loss_sgm = self._compute_sgm_loss(img, img_metas, p_global_grad)
        if loss_sgm is not None:
            losses['loss_sgm'] = loss_sgm

        return losses

    def predict(self, img, data_samples, **kwargs):
        """
        Single-image test without augmentation.
        Args:
            img (torch.Tensor): Input image tensor of shape (1, 3, height, width).
            data_samples (List[:obj:`DetDataSample`]): The data samples
                that include meta information.
        Returns:
            result_dict (List[dict]): Single-image result containing prediction outputs and
             img_metas as 'result' and 'metas' respectively.
        """
        for i in range(len(data_samples)):
            data_samples[i].metainfo["batch_input_shape"] = tuple(img.size()[-2:])

        x = self.extract_feat(img)
        outputs = self.bbox_head.predict(x, data_samples)
        return outputs
