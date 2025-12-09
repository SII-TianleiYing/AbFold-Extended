import torch
import torch.nn as nn

from abfold.angle_resnet import AngleResnet
from abfold.attention import BiasAttentionModule
from abfold.backbone_update import BackboneUpdate
from abfold.data import data_transforms
from abfold.global_predict import GlobalPredict
from abfold.ipa import InvariantPointAttention
from abfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from abfold.transition import LayerNorm, StructureModuleTransition, MergeFeature
from abfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
    atom14_to_atom37,
)
from abfold.utils.rigid_utils import Rigid
from abfold.utils.tensor_utils import (
    dict_multimap,
)
from abfold.utils.ESMFold.model import EsmFoldTriangularSelfAttentionBlock, EsmFoldRelativePosition
from abfold.diffusion.sample import sample


class AbFold(nn.Module):
    """
    abfold
    implemented the abfold mentioned in 2022.9.19
    """

    def __init__(self, config, dropout_rate=0.1, no_transition_layers=1):
        super(AbFold, self).__init__()

        self.dropout_rate = dropout_rate
        self.default_frames = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None

        # merge af and igfold feature
        merge_config = config.merge_feat
        self.merge_feature = MergeFeature(merge_config)

        # EsmFoldTriangularSelfAttentionBlock
        trunk_config = config.EsmFoldTriangularSelfAttentionBlock
        self.pairwise_positional_embedding = EsmFoldRelativePosition(trunk_config)
        self.blocks = nn.ModuleList([EsmFoldTriangularSelfAttentionBlock(trunk_config) for _ in range(trunk_config.num_blocks)])

        self.ipa_layers = nn.ModuleList([])
        for _ in range(config['ipa']['num_layer']):
            ipa = InvariantPointAttention(**config['ipa']['module_params'])

            ipa_dropout = nn.Dropout(self.dropout_rate)
            layer_norm_ipa = LayerNorm(config['ipa']['module_params']['c_s'])

            transition = StructureModuleTransition(
                config['ipa']['module_params']['c_s'],
                no_transition_layers,
                self.dropout_rate,
            )

            # l_transition = StructureModuleTransition(
            #     config['ipa']['module_params']['c_s'],
            #     no_transition_layers,
            #     self.dropout_rate,
            # )

            bb_update = BackboneUpdate(config['ipa']['module_params']['c_s'])
            # self.ipa_layers.append(nn.ModuleList([h_self_ipa, l_self_ipa, h_cross_ipa, l_cross_ipa,
            #                                       ipa_dropout, layer_norm_ipa, h_transition, l_transition, bb_update]))
            self.ipa_layers.append(nn.ModuleList([ipa, ipa_dropout, layer_norm_ipa, transition, bb_update]))

        self.angle_resnet = AngleResnet(**config['angle_resnet'])
        # self.global_predict = GlobalPredict(config['ipa']['module_params']['c_s'])

        self.val_linear = nn.Linear(7, 7)

    def forward(
            self,
            batch,
            # mask=None,
            extract_embedding=False,
            diffuse_embedding=False,
    ):
        """
        :param s:
            [*, N_res, C_s] single representation
        :param z:
            [*, N_res, N_res, C_z] pair representation
        :param aatype:
            [*, N_res]
        :param r:
            [*, N_res]
        :param mask_h:
        :param mask_l:
        :return:
        """
        s = batch['s']
        z = batch['z']
        aatype = batch['aatype']
        r = batch.get('r', None)
        mask = batch.get('mask', None)
        point_feat = batch['point_feat']
        res_idx =  batch['res_idx']

        if r is None:
            r = Rigid.identity(
                s.shape[:-1],
                s.dtype,
                s.device,
                True,
                fmt="quat",
            )
        else:
            r = Rigid.from_tensor_4x4(r)
        if mask is None:
            mask = s.new_ones(s.shape[:-1])

        s = self.merge_feature(s, point_feat, batch.get('plddt', None))
        z = z + self.pairwise_positional_embedding(res_idx, mask=mask)
        for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=res_idx)

        if extract_embedding:
            return {'s': s.squeeze(), 'z': z.squeeze(), 'aatype': aatype.squeeze(), 'res_idx': res_idx.squeeze()}
        elif diffuse_embedding:
            z = sample(s, z)
        s_initial = s      

        # self_ipa and cross_ipa
        for ipa, ipa_dropout, layer_norm_ipa, transition, bb_update in self.ipa_layers:
            s = s + ipa(s, s, z, r, r, mask, mask)

            s = layer_norm_ipa(ipa_dropout(s))
            s = transition(s)

            r = r.compose_q_update_vec(bb_update(s))

        # r_global = Rigid.from_tensor_7(self.global_predict(s)[..., None, :], True)
        r = r.scale_translation(20)

        # [*, N, 7, 2]
        unnormalized_angles, angles = self.angle_resnet(s, s_initial)

        all_frames_to_global = self.torsion_angles_to_frames(
            r,
            angles,
            aatype,
        )

        pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            aatype,
        )

        preds = {
            "frames": r.to_tensor_7(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": pred_xyz,
            # "t_global_tensor_7": r_global.to_tensor_7(),
        }
        outputs_sm = [preds]
        outputs_sm = dict_multimap(torch.stack, outputs_sm)
        outputs_sm["single"] = s

        outputs = {'sm': outputs_sm}
        aatype = aatype
        feats = self._generate_feats_from_aatype(aatype)
        outputs["aatype"] = aatype
        outputs["seq_length"] = torch.full([aatype.shape[0]], fill_value=aatype.shape[1], device=aatype.device)
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        r.stop_rot_gradient()
        # r_global.stop_rot_gradient()

        return outputs
    
    def sample(self, s, z, aatype):
        mask = s.new_ones(s.shape[:-1])
        r = Rigid.identity(
                s.shape[:-1],
                s.dtype,
                s.device,
                True,
                fmt="quat",
            )
        # z = sample(s, z)

        s_initial = s      

        # self_ipa and cross_ipa
        for ipa, ipa_dropout, layer_norm_ipa, transition, bb_update in self.ipa_layers:
            s = s + ipa(s, s, z, r, r, mask, mask)

            s = layer_norm_ipa(ipa_dropout(s))
            s = transition(s)

            r = r.compose_q_update_vec(bb_update(s))

        # r_global = Rigid.from_tensor_7(self.global_predict(s)[..., None, :], True)
        r = r.scale_translation(20)

        # [*, N, 7, 2]
        unnormalized_angles, angles = self.angle_resnet(s, s_initial)

        all_frames_to_global = self.torsion_angles_to_frames(
            r,
            angles,
            aatype,
        )

        pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            aatype,
        )

        preds = {
            "frames": r.to_tensor_7(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": pred_xyz,
            # "t_global_tensor_7": r_global.to_tensor_7(),
        }
        outputs_sm = [preds]
        outputs_sm = dict_multimap(torch.stack, outputs_sm)
        outputs_sm["single"] = s

        outputs = {'sm': outputs_sm}
        aatype = aatype
        feats = self._generate_feats_from_aatype(aatype)
        outputs["aatype"] = aatype
        outputs["seq_length"] = torch.full([aatype.shape[0]], fill_value=aatype.shape[1], device=aatype.device)
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        r.stop_rot_gradient()
        # r_global.stop_rot_gradient()

        return outputs

    def _generate_feats_from_aatype(self, aatype):
        return data_transforms.make_atom14_masks({"aatype": aatype})

    def _init_residue_constants(self, float_dtype, device):
        if self.default_frames is None:
            self.default_frames = torch.tensor(
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.group_idx is None:
            self.group_idx = torch.tensor(
                restype_atom14_to_rigid_group,
                device=device,
                requires_grad=False,
            )
        if self.atom_mask is None:
            self.atom_mask = torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.lit_positions is None:
            self.lit_positions = torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
            self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

    def get_positions_from_r(self, r_h, r_l, s_h, s_l, s_h_initial, s_l_initial, aatype_h, aatype_l):
        unnormalized_angles_h, angles_h = self.angle_resnet(s_h, s_h_initial)
        unnormalized_angles_l, angles_l = self.angle_resnet(s_l, s_l_initial)

        all_frames_to_global_h = self.torsion_angles_to_frames(
            r_h,
            angles_h,
            aatype_h,
        )

        all_frames_to_global_l = self.torsion_angles_to_frames(
            r_l,
            angles_l,
            aatype_l,
        )

        pred_xyz_h = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global_h,
            aatype_h,
        )

        pred_xyz_l = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global_l,
            aatype_l,
        )

        positions = torch.cat((pred_xyz_h, pred_xyz_l), -3)
        aatype = torch.cat([aatype_h, aatype_l], -1)
        feats = self._generate_feats_from_aatype(aatype)
        positions = atom14_to_atom37(positions, feats)

        return positions
