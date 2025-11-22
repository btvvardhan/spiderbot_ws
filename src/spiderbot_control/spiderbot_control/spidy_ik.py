#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class LegConfig:
    name: str
    coxa_origin: np.ndarray
    coxa_axis: np.ndarray
    femur_offset: np.ndarray
    femur_axis: np.ndarray
    tibia_offset: np.ndarray
    tibia_axis: np.ndarray
    coxa_length: float = 0.0553
    femur_length: float = 0.08
    tibia_length: float = 0.12

def _axis_basis(u: np.ndarray):
    """Build an orthonormal basis aligned with the joint axis u.
       e2 = u (axis), e1 ⟂ u (close to world X), e3 = e2 × e1.
       The bending plane is span{e1, e3} (perpendicular to u).
    """
    e2 = np.asarray(u, dtype=float)
    e2 /= (np.linalg.norm(e2) + 1e-12)
    # choose helper not parallel to e2
    helper = np.array([1.0, 0.0, 0.0]) if abs(e2[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, e2) * e2
    e1 /= (np.linalg.norm(e1) + 1e-12)
    e3 = np.cross(e2, e1)
    return e1, e2, e3

class SpiderIK:
    def __init__(self):
        self.legs = self._init_leg_configs()

    def _init_leg_configs(self) -> dict:
        # Values pulled from your URDF
        fl = LegConfig(
            name="fl",
            coxa_origin=np.array([0.0802, -0.1049, -0.0717]),
            coxa_axis=np.array([0.0, 0.0, 1.0]),
            femur_offset=np.array([0.055296, -0.015981, -0.0171]),
            femur_axis=np.array([0.707107, 0.707107, 0.0]),
            tibia_offset=np.array([0.056568, -0.056568, 0.0]),
            tibia_axis=np.array([0.707107, 0.707107, 0.0]),
        )
        fr = LegConfig(
            name="fr",
            coxa_origin=np.array([-0.1202, -0.10365, -0.07235]),
            coxa_axis=np.array([0.0, 0.0, 1.0]),
            femur_offset=np.array([-0.015981, -0.055296, -0.0171]),
            femur_axis=np.array([0.707107, -0.707107, 0.0]),
            tibia_offset=np.array([-0.055507, -0.057629, 0.0]),
            tibia_axis=np.array([0.707107, -0.707107, 0.0]),
        )
        rl = LegConfig(
            name="rl",
            coxa_origin=np.array([0.0802, 0.0649, -0.0717]),
            coxa_axis=np.array([0.0, 0.0, 1.0]),
            femur_offset=np.array([0.015981, 0.055296, -0.0171]),
            femur_axis=np.array([-0.707107, 0.707107, 0.0]),
            tibia_offset=np.array([0.056568, 0.056568, 0.0]),
            tibia_axis=np.array([-0.707107, 0.707107, 0.0]),
        )
        rr = LegConfig(
            name="rr",
            coxa_origin=np.array([-0.1202, 0.0649, -0.0717]),
            coxa_axis=np.array([0.0, 0.0, 1.0]),
            femur_offset=np.array([-0.055296, 0.015981, -0.0156]),
            femur_axis=np.array([-0.707107, -0.707107, 0.0]),
            tibia_offset=np.array([-0.056568, 0.056568, 0.0]),
            tibia_axis=np.array([-0.707107, -0.707107, 0.0]),
        )
        return {"fl": fl, "fr": fr, "rl": rl, "rr": rr}

    def solve_leg_ik(self, leg_name: str, target_pos: np.ndarray) -> Optional[Tuple[float, float, float]]:
        leg = self.legs[leg_name]

        # 1) Coxa yaw: rotate target into coxa frame
        rel = target_pos - leg.coxa_origin
        coxa_angle = np.arctan2(rel[1], rel[0])
        c, s = np.cos(-coxa_angle), np.sin(-coxa_angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        tgt_coxa = Rz @ rel

        # 2) From femur joint
        tgt_femur = tgt_coxa - leg.femur_offset

        # 3) Build axis-aligned bending plane basis in coxa frame
        e1, e2, e3 = _axis_basis(leg.femur_axis)  # e2 is femur/tibia axis (parallel for this leg)

        # 4) Project target into bending plane (e1-e3)
        xp = np.dot(tgt_femur, e1)
        zp = np.dot(tgt_femur, e3)
        r = np.hypot(xp, zp)

        # 5) Reach clamp (handles donut hole + numerical edges)
        L1, L2 = leg.femur_length, leg.tibia_length
        eps = 1e-3
        min_r = abs(L1 - L2) + eps
        max_r = (L1 + L2) - eps
        if r < min_r or r > max_r:
            r_clamp = np.clip(r, min_r, max_r)
            if r < 1e-9:
                # push along +e1
                xp, zp = r_clamp, 0.0
            else:
                scale = r_clamp / r
                xp, zp = xp * scale, zp * scale
            print(f"Info: Clamped {leg_name} target to reachable band (min={min_r:.3f}, max={max_r:.3f}); used r={r_clamp:.3f}")
            r = r_clamp

        # 6) 2-link planar IK in (e1,e3) plane
        # tibia (exterior)
        cos_t = (L1**2 + L2**2 - r**2) / (2 * L1 * L2)
        cos_t = np.clip(cos_t, -1.0, 1.0)
        tibia_angle = np.pi - np.arccos(cos_t)

        # femur
        cos_f = (L1**2 + r**2 - L2**2) / (2 * L1 * r)
        cos_f = np.clip(cos_f, -1.0, 1.0)
        angle_to_tgt = np.arctan2(zp, xp)   # angle in bending plane from +e1 toward +e3
        femur_angle = angle_to_tgt - np.arccos(cos_f)

        # Done: angles are about the correct axes in the coxa frame
        return (coxa_angle, femur_angle, tibia_angle)

    def get_standing_pose(self, body_height: float = 0.10) -> dict:
        default_feet = {
            'fl': np.array([0.18, -0.18, -body_height]),
            'fr': np.array([-0.18, -0.18, -body_height]),
            'rl': np.array([0.18,  0.18, -body_height]),
            'rr': np.array([-0.18,  0.18, -body_height]),
        }
        out = {}
        for leg, p in default_feet.items():
            ang = self.solve_leg_ik(leg, p)
            if ang is None:
                continue
            c, f, t = ang
            out[f"{leg}_coxa_joint"]  = c
            out[f"{leg}_femur_joint"] = f
            out[f"{leg}_tibia_joint"] = t
        return out
