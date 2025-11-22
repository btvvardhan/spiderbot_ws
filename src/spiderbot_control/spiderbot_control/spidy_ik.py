
#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
spidy_ik.py
-----------
Minimal leg IK + URDF helpers for a quadruped "spiderbot" with 3-DoF legs:
(hip_yaw, hip_pitch, knee).

- Parses URDF to extract leg joint order, hip offsets, approximate link lengths.
- Provides LegIK.solve(x,y,z) -> (q_yaw, q_hip, q_knee).

Assumptions:
- Hip yaw axis is vertical (z) in the hip frame.
- Hip pitch and knee pitch rotate in the sagittal plane.
- Link lengths are approximated from joint origins; override via config if needed.

Usage:
    from spidy_ik import RobotKinematics, LegIK
    kin = RobotKinematics.from_urdf("/path/to/spidy.urdf")
    q = kin.legs['LF'].ik.solve( x, y, z )

"""

from __future__ import annotations
import math, json, xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

def _fltlist(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    return [float(x) for x in s.strip().split()]

@dataclass
class JointInfo:
    name: str
    type: str
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    axis: Optional[Tuple[float, float, float]]
    lower: float
    upper: float

@dataclass
class LegInfo:
    joints: List[JointInfo]           # [hip_yaw, hip_pitch, knee]
    hip_offset_in_body: Tuple[float, float, float]
    lengths: Tuple[float, float, float]   # (coxa, femur, tibia)
    names: Tuple[str, str, str]       # joint names in IK order

class RobotKinematics:
    def __init__(self, legs: Dict[str, LegInfo], neutral_pose: Dict[str, float]):
        self.legs = legs
        self.neutral_pose = neutral_pose

    @staticmethod
    def from_urdf(urdf_path: str, neutral_path: Optional[str]=None) -> "RobotKinematics":
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # parse joints
        joints = {}
        children_of = {}
        for j in root.findall('joint'):
            jname = j.attrib.get('name')
            jtype = j.attrib.get('type')
            parent = j.find('parent').attrib['link']
            child = j.find('child').attrib['link']
            origin = j.find('origin')
            xyz = _fltlist(origin.attrib.get('xyz')) if origin is not None else [0.0,0.0,0.0]
            rpy = _fltlist(origin.attrib.get('rpy')) if origin is not None else [0.0,0.0,0.0]
            axis_tag = j.find('axis')
            axis = _fltlist(axis_tag.attrib.get('xyz')) if axis_tag is not None else None
            limit = j.find('limit')
            lower = float(limit.attrib.get('lower', 'nan')) if limit is not None and 'lower' in limit.attrib else float('nan')
            upper = float(limit.attrib.get('upper', 'nan')) if limit is not None and 'upper' in limit.attrib else float('nan')
            ji = JointInfo(jname, jtype, parent, child, tuple(xyz), tuple(rpy), tuple(axis) if axis else None, lower, upper)
            joints[jname] = ji
            children_of[parent] = children_of.get(parent, []) + [jname]

        # heuristic: identify 4 leg chains with 3 revolute/continuous joints
        chains: List[List[str]] = []
        # find base links
        all_children_links = set(j.child for j in joints.values())
        all_links = set(root.iterfind('link'))
        base_candidates = []
        for link_tag in root.findall('link'):
            lname = link_tag.attrib['name']
            if lname not in all_children_links:
                base_candidates.append(lname)
        base_link = base_candidates[0] if base_candidates else 'base_link'

        # BFS build chains
        from collections import deque
        dq = deque([(base_link, [])])
        link_children = {}
        for j in joints.values():
            link_children.setdefault(j.parent, []).append(j)
        link_to_children_links = {}
        for j in joints.values():
            link_to_children_links.setdefault(j.parent, []).append(j.child)

        leaves = set(j.child for j in joints.values()) - set(j.parent for j in joints.values())
        # gather path of joint names to leaves
        def extend_paths(link, path):
            if link not in link_children:
                if path:
                    chains.append(path)
                return
            for j in link_children[link]:
                extend_paths(j.child, path+[j.name])

        extend_paths(base_link, [])

        def is_leg(chain: List[str]) -> bool:
            types = [joints[j].type for j in chain]
            ok = all(t in ('revolute','continuous','prismatic') for t in types) and (2 <= len(chain) <= 4)
            end_link = joints[chain[-1]].child
            hint = any(s in end_link.lower() for s in ['foot','toe','tip','tibia','ankle'])
            return ok or hint

        leg_chains = [c for c in chains if is_leg(c)]
        # pick 4 longest chains if more appear
        leg_chains = sorted(leg_chains, key=len, reverse=True)[:4]

        # Attempt to label legs LF, RF, LH, RH by hip offsets (x forward, y left)
        def joint_origin(jn): return joints[jn].origin_xyz
        def hip_of(chain): return joints[chain[0]].origin_xyz
        hips = [hip_of(c) for c in leg_chains]
        # sort by y then -x to group left/right and front/back
        idxs = list(range(len(leg_chains)))
        idxs.sort(key=lambda i: (-hips[i][1], -hips[i][0]))  # left first (y>0), within each, front first (x>0)
        labels = ['LF','LH','RF','RH'] if len(idxs)==4 else [f"L{i}" for i in range(len(idxs))]
        labeled = {}
        for label, i in zip(labels, idxs):
            chain = leg_chains[i]
            # compute lengths: norm of origins for first 3 joints as (coxa,femur,tibia) approx
            def norm(v): return (v[0]**2+v[1]**2+v[2]**2)**0.5
            lens = []
            for jn in chain[:3]:
                lens.append(norm(joints[jn].origin_xyz))
            if len(lens)<3:
                lens += [0.05]*(3-len(lens))
            ji = [joints[jn] for jn in chain[:3]]
            hip_off = joints[chain[0]].origin_xyz
            labeled[label] = LegInfo(
                joints=ji,
                hip_offset_in_body=tuple(hip_off),
                lengths=tuple(lens[:3]),
                names=tuple(j.name for j in ji)
            )

        neutral = {}
        if neutral_path:
            try:
                with open(neutral_path,'r') as f:
                    neutral = json.load(f).get('neutral_pose', {})
            except Exception:
                pass

        return RobotKinematics(labeled, neutral)

class LegIK:
    """Planar 2-link + yaw decomposition IK with joint limits and simple clamping."""
    def __init__(self, coxa: float, femur: float, tibia: float,
                 q_limits: Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]]):
        self.coxa = max(1e-6, float(coxa))
        self.femur = max(1e-6, float(femur))
        self.tibia = max(1e-6, float(tibia))
        self.lims = q_limits

    def clamp(self, i, q):
        lo, hi = self.lims[i]
        if math.isnan(lo) or math.isnan(hi):
            return q
        return max(lo, min(hi, q))

    def solve(self, x: float, y: float, z: float) -> Tuple[float,float,float]:
        # yaw to align sagittal plane
        q_yaw = math.atan2(y, x)
        r = math.hypot(x, y)
        # distance from hip pitch axis after coxa
        xp = max(1e-6, r - self.coxa)
        zp = z
        # 2-link IK
        L1, L2 = self.femur, self.tibia
        D = (xp*xp + zp*zp - L1*L1 - L2*L2) / (2*L1*L2)
        D = max(-1.0, min(1.0, D))
        q_knee = math.acos(D) - math.pi  # knee "bend" negative convention; adjust as needed
        phi = math.atan2(zp, xp)
        psi = math.atan2(L2*math.sin(q_knee+math.pi), L1 + L2*math.cos(q_knee+math.pi))
        q_hip = phi - psi

        # clamp
        q_yaw = self.clamp(0, q_yaw)
        q_hip = self.clamp(1, q_hip)
        q_knee = self.clamp(2, q_knee)
        return (q_yaw, q_hip, q_knee)

def build_robot_kinematics(urdf_path: str, neutral_path: Optional[str]=None) -> RobotKinematics:
    kin = RobotKinematics.from_urdf(urdf_path, neutral_path)
    # attach IK solvers with limits
    for leg in kin.legs.values():
        # extract limits
        lims = []
        for j in leg.joints:
            lo = j.lower if not math.isnan(j.lower) else -1e9
            hi = j.upper if not math.isnan(j.upper) else +1e9
            lims.append((lo, hi))
        leg.ik = LegIK(leg.lengths[0], leg.lengths[1], leg.lengths[2], tuple(lims))
    return kin
