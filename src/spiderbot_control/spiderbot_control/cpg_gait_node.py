#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
cpg_gait_node.py (order-robust + IK lengths + x-bias + gait modes + adaptive knee sign)
- Output joint order is taken from neutral_json["leg_chains"] if present (e.g., [fl, fr, rl, rr])
- Canonical legs (LF, RF, RL, RR) are mapped by joint-name prefix (fl_/fr_/rl_/rr_), not hip offsets
- signs.json can be either a 12-element list (legacy) or a dict {joint_name: sign}
- IK segment lengths are taken from ik_dims.json if available (strongly recommended)
- A per-leg neutral horizontal reach (x-bias) keeps r - coxa > 0 so the femur/tibia articulate
- Gait modes: crawl (4-beat), trot (diagonal coupling), pace (lateral), bound (front vs rear)
- Uses one master oscillator to perfectly couple paired legs
- Knee angle sign is adapted to URDF limits (if knee cannot go negative, use positive-flexion convention)
- Publishes Float64MultiArray(12) to /position_controller/commands
"""

from __future__ import annotations
import math, json, os, xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

# ---------- helpers ----------
def _fltlist(s: Optional[str]):
    if s is None: return None
    return [float(x) for x in s.strip().split()]

@dataclass
class JointInfo:
    name: str; type: str; parent: str; child: str
    origin_xyz: Tuple[float,float,float]
    origin_rpy: Tuple[float,float,float]
    axis: Optional[Tuple[float,float,float]]
    lower: float; upper: float

@dataclass
class LegInfo:
    joints: List[JointInfo]                 # [hip_yaw, hip_pitch, knee]
    hip_offset_in_body: Tuple[float,float,float]
    lengths: Tuple[float,float,float]       # (coxa,femur,tibia) *approx*; overridden by ik_dims if provided
    names: Tuple[str,str,str]               # joint names in IK order

class LegIK:
    def __init__(self, coxa, femur, tibia, qlims):
        self.coxa=max(1e-6,float(coxa)); self.femur=max(1e-6,float(femur)); self.tibia=max(1e-6,float(tibia))
        self.lims=qlims  # [(lo,hi),(lo,hi),(lo,hi)]
        # Determine knee flexion sign from limits:
        # If knee cannot go negative (lower >= 0), assume positive-flexion convention (0=straight, + = bend).
        k_lo, k_hi = qlims[2]
        self.knee_positive = (not math.isnan(k_lo)) and (k_lo >= -1e-6)

    def clamp(self,i,q):
        lo,hi=self.lims[i]
        if not math.isnan(lo): q = max(lo,q)
        if not math.isnan(hi): q = min(hi,q)
        return q

    def solve(self,x,y,z):
        # yaw to align sagittal plane
        q_yaw=math.atan2(y,x); r=math.hypot(x,y)
        # distance from hip pitch axis after coxa
        xp=max(1e-6,r-self.coxa); zp=z
        # 2-link IK in sagittal plane
        L1,L2=self.femur,self.tibia
        # internal knee angle theta in [0, pi], where 0=straight (R=L1+L2), pi=folded
        # cos(theta) = (L1^2 + L2^2 - R^2)/(2 L1 L2)
        R2 = xp*xp + zp*zp
        cos_theta = (L1*L1 + L2*L2 - R2)/(2.0*L1*L2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)
        # map to joint variable: either positive-flexion (0..pi) or negative-flexion (0..-pi)
        q_knee = theta if self.knee_positive else -theta

        # hip pitch using standard geometry (elbow-down)
        # use the "elbow" angle at the knee for vector from hip to foot along femur
        phi = math.atan2(zp, xp)
        # angle between L1 vector and the line to foot
        # For elbow-down, the angle at the elbow is (pi - theta)
        elbow = math.pi - theta
        psi = math.atan2(L2*math.sin(elbow), L1 + L2*math.cos(elbow))
        q_hip = phi - psi

        # clamp to URDF limits
        q_yaw = self.clamp(0,q_yaw)
        q_hip = self.clamp(1,q_hip)
        q_knee = self.clamp(2,q_knee)
        return (q_yaw, q_hip, q_knee)

class RobotKinematics:
    def __init__(self, legs: Dict[str,LegInfo], neutral_pose: Dict[str,float]):
        self.legs=legs; self.neutral_pose=neutral_pose

    @staticmethod
    def from_paths(urdf_path: str, neutral_json_path: str) -> "RobotKinematics":
        root=ET.parse(urdf_path).getroot()
        with open(neutral_json_path,"r") as f:
            neutral=json.load(f).get("neutral_pose",{})
        joints: Dict[str,JointInfo]={}; children_of={}
        for j in root.findall('joint'):
            jname=j.attrib.get('name'); jtype=j.attrib.get('type')
            parent=j.find('parent').attrib['link']; child=j.find('child').attrib['link']
            origin=j.find('origin')
            xyz=_fltlist(origin.attrib.get('xyz')) if origin is not None else [0,0,0]
            rpy=_fltlist(origin.attrib.get('rpy')) if origin is not None else [0,0,0]
            axis_tag=j.find('axis'); axis=_fltlist(axis_tag.attrib.get('xyz')) if axis_tag is not None else None
            limit=j.find('limit')
            lower=float(limit.attrib.get('lower','nan')) if limit is not None and 'lower' in limit.attrib else float('nan')
            upper=float(limit.attrib.get('upper','nan')) if limit is not None and 'upper' in limit.attrib else float('nan')
            joints[jname]=JointInfo(jname,jtype,parent,child,tuple(xyz),tuple(rpy),tuple(axis) if axis else None,lower,upper)
            children_of.setdefault(parent,[]).append(jname)
        # base link
        children_links=set(j.child for j in joints.values())
        base_link=next((l.attrib['name'] for l in root.findall('link') if l.attrib['name'] not in children_links),'base_link')
        # DFS to build chains
        link_children={}
        for j in joints.values(): link_children.setdefault(j.parent,[]).append(j)
        chains=[]
        def dfs(link,path):
            if link not in link_children:
                if path: chains.append(path); return
            for jj in link_children[link]: dfs(jj.child, path+[jj.name])
        dfs(base_link,[])
        def is_leg(chain):
            types=[joints[k].type for k in chain]
            ok=all(t in('revolute','continuous','prismatic') for t in types) and (2<=len(chain)<=4)
            end=joints[chain[-1]].child; hint=any(s in end.lower() for s in ['foot','toe','tip','tibia','ankle'])
            return ok or hint
        leg_chains=[c for c in chains if is_leg(c)]
        leg_chains=sorted(leg_chains,key=len,reverse=True)[:4]
        # label by hip offsets (old heuristic; we override later by name prefixes)
        def hip_xyz(chain): return joints[chain[0]].origin_xyz
        hips=[hip_xyz(c) for c in leg_chains]
        idxs=list(range(len(leg_chains))); idxs.sort(key=lambda i:(-hips[i][1],-hips[i][0]))
        labels=['LF','LH','FR','RH'] if len(idxs)==4 else [f"L{i}" for i in idxs]
        labeled={}
        for label,i in zip(labels,idxs):
            chain=leg_chains[i]
            # NOTE: lengths here are only rough placeholders; overridden with ik_dims if provided
            def norm(v): return (v[0]**2+v[1]**2+v[2]**2)**0.5
            lens=[norm(joints[jn].origin_xyz) for jn in chain[:3]]
            while len(lens)<3: lens.append(0.05)
            ji=[joints[jn] for jn in chain[:3]]
            hip_off=joints[chain[0]].origin_xyz
            leg=LegInfo(ji,tuple(hip_off),(lens[0],lens[1],lens[2]),(ji[0].name,ji[1].name,ji[2].name))
            # attach limits to IK
            lims=[]
            for j in leg.joints:
                lo=j.lower if not math.isnan(j.lower) else -1e9
                hi=j.upper if not math.isnan(j.upper) else +1e9
                lims.append((lo,hi))
            leg.ik=LegIK(leg.lengths[0],leg.lengths[1],leg.lengths[2],tuple(lims))
            labeled[label]=leg
        return RobotKinematics(labeled, neutral)

# ---------- alias helpers ----------
def _resolve_leg_aliases(available: List[str]) -> Dict[str,str]:
    """
    Canonical -> actual keys in URDF map.
    Canonical: LF, RF, RL, RR ; Accepts aliases FL↔LF, FR↔RF, LH↔RL, RH↔RR.
    """
    s=set(available)
    def pick(*opts):
        for o in opts:
            if o in s: return o
        return None
    m={
        'LF': pick('LF','FL'),
        'RF': pick('RF','FR'),
        'RL': pick('RL','LH'),
        'RR': pick('RR','RH'),
    }
    return {k:v for k,v in m.items() if v is not None}

def _alias_by_joint_prefix(legs: Dict[str,LegInfo]) -> Dict[str,str]:
    """Prefer mapping by first joint name (fl_/fr_/rl_/rr_) for robustness."""
    m={}
    for lbl, leg in legs.items():
        if not leg.names: continue
        j0=leg.names[0].lower()
        if j0.startswith(('fl_','lf_')): m['LF']=lbl
        elif j0.startswith(('fr_','rf_')): m['RF']=lbl
        elif j0.startswith(('rl_','lh_')): m['RL']=lbl
        elif j0.startswith(('rr_','rh_')): m['RR']=lbl
    return m

# ---------- Node ----------
@dataclass
class GaitParams:
    duty: float=0.85; base_freq: float=0.8; max_stride: float=0.08
    ground_z: float=-0.10; clearance: float=0.03

class CPGNode(Node):
    def __init__(self):
        super().__init__("cpg_gait_node")

        # params (hardcoded defaults set to your workspace)
        self.declare_parameter("urdf_path", "/home/teja/spiderbot_ws/src/spiderbot_description/urdf/spidy.urdf")
        self.declare_parameter("neutral_json", "/home/teja/spiderbot_ws/src/spiderbot_control/spiderbot_control/spidy_neutral_pose.json")
        self.declare_parameter("signs_json", "/home/teja/spiderbot_ws/src/spiderbot_control/spiderbot_control/signs.json")
        self.declare_parameter("ik_dims_json", "/home/teja/spiderbot_ws/src/spiderbot_control/spiderbot_control/ik_dims.json")
        self.declare_parameter("out_topic","/position_controller/commands")
        self.declare_parameter("rate_hz",100.0)
        self.declare_parameter("duty",0.85)
        self.declare_parameter("base_freq_hz",0.8)
        self.declare_parameter("max_stride_m",0.08)
        self.declare_parameter("ground_z",-0.10)
        self.declare_parameter("swing_clearance",0.03)
        self.declare_parameter("wait_for_first_cmd", True)
        self.declare_parameter("hold_on_idle", True)
        self.declare_parameter("idle_deadband", 1e-3)
        self.declare_parameter("idle_timeout", 0.4)
        # neutral horizontal reach (x-bias) to keep xp = r - coxa > 0 (so femur/tibia articulate)
        self.declare_parameter("stance_x_bias_m", 0.12)
        # gait mode
        self.declare_parameter("gait_mode", "crawl")  # crawl | trot | pace | bound

        urdf_path=self.get_parameter("urdf_path").get_parameter_value().string_value
        neutral_path=self.get_parameter("neutral_json").get_parameter_value().string_value
        signs_path=self.get_parameter("signs_json").get_parameter_value().string_value
        ik_dims_path=self.get_parameter("ik_dims_json").get_parameter_value().string_value
        self.out_topic=self.get_parameter("out_topic").get_parameter_value().string_value

        # Expand environment variables in paths
        urdf_path = os.path.expandvars(urdf_path)
        neutral_path = os.path.expandvars(neutral_path)
        signs_path = os.path.expandvars(signs_path)
        ik_dims_path = os.path.expandvars(ik_dims_path)

        self.kin=RobotKinematics.from_paths(urdf_path, neutral_path)

        rate_hz=self.get_parameter("rate_hz").get_parameter_value().double_value
        self.dt=1.0/max(1e-3,rate_hz)
        self.g=GaitParams(
            duty=self.get_parameter("duty").get_parameter_value().double_value,
            base_freq=self.get_parameter("base_freq_hz").get_parameter_value().double_value,
            max_stride=self.get_parameter("max_stride_m").get_parameter_value().double_value,
            ground_z=self.get_parameter("ground_z").get_parameter_value().double_value,
            clearance=self.get_parameter("swing_clearance").get_parameter_value().double_value,
        )
        self.wait_for_first_cmd=bool(self.get_parameter("wait_for_first_cmd").get_parameter_value().bool_value)
        self.hold_on_idle=bool(self.get_parameter("hold_on_idle").get_parameter_value().bool_value)
        self.idle_deadband=float(self.get_parameter("idle_deadband").get_parameter_value().double_value)
        self.idle_timeout=float(self.get_parameter("idle_timeout").get_parameter_value().double_value)

        # Build alias map: geometry -> override with name-prefix mapping
        available=list(self.kin.legs.keys())
        alias=_resolve_leg_aliases(available)
        alias.update(_alias_by_joint_prefix(self.kin.legs))  # name-based mapping wins
        self.alias = alias  # store for later

        # ---- Choose output order ----
        # Prefer neutral_json["leg_chains"] (e.g., [fl, fr, rl, rr]) to match controller.
        pref_leg_chains=None
        try:
            with open(neutral_path,"r") as f:
                cfg=json.load(f)
                pref_leg_chains=cfg.get("leg_chains")
        except Exception:
            pref_leg_chains=None

        self.leg_labels=[]
        self.leg_joint_order={}
        self.joint_names=[]

        def canonical_from_joint_name(jn: str) -> Optional[str]:
            j=jn.lower()
            if j.startswith(('fl_','lf_')): return 'LF'
            if j.startswith(('fr_','rf_')): return 'RF'
            if j.startswith(('rl_','lh_')): return 'RL'
            if j.startswith(('rr_','rh_')): return 'RR'
            return None

        if isinstance(pref_leg_chains, list) and all(isinstance(c, list) and len(c)>=1 for c in pref_leg_chains):
            # Use config order; map each chain's first joint to the actual leg label via alias
            for chain in pref_leg_chains:
                can = canonical_from_joint_name(chain[0])
                if can and can in alias:
                    lbl = alias[can]
                    self.leg_labels.append(lbl)
                    self.leg_joint_order[lbl]=list(chain[:3])
                    self.joint_names += list(chain[:3])
            self.get_logger().info("Using leg_chains order from neutral config for output joint order.")
        else:
            # Fallback to canonical [LF, RF, RL, RR]
            out_order=['LF','RF','RL','RR']
            for can in out_order:
                if can in alias:
                    lbl=alias[can]
                    self.leg_labels.append(lbl)
                    self.leg_joint_order[lbl]=list(self.kin.legs[lbl].names)
                    self.joint_names += list(self.kin.legs[lbl].names)

        # ---- Override IK link lengths from ik_dims.json (if present) ----
        self.x_bias_by_leg: Dict[str, float] = {}

        def _solve_bias_for_leg(lbl, target_qh=-0.8, z0=None):
            leg = self.kin.legs[lbl]
            if z0 is None:
                z0 = self.g.ground_z
            # binary search on x so that qh ~ target_qh at (x,0,z0)
            Lc, Lf, Lt = leg.ik.coxa, leg.ik.femur, leg.ik.tibia
            # feasible x range: from just beyond coxa to near full reach
            lo = Lc + 0.02
            hi = min(Lc + Lf + Lt - 0.02, 0.40)
            def hip_at(x):
                qy,qh,qk = leg.ik.solve(x, 0.0, z0)
                return qh
            # If even at hi the hip is still too negative, return mid; likewise for lo
            qh_lo = hip_at(lo); qh_hi = hip_at(hi)
            if math.isnan(qh_lo) or math.isnan(qh_hi):
                return max(0.05, 0.5*Lf + 0.6*Lt)
            # If monotonic increasing w.r.t x, bsearch; otherwise fallback
            increasing = qh_hi > qh_lo
            if not increasing:
                # try expand range a bit
                hi = min(hi + 0.1, Lc + Lf + Lt - 0.01)
                qh_hi = hip_at(hi)
                increasing = qh_hi > qh_lo
            if not increasing:
                return max(0.05, 0.5*Lf + 0.6*Lt)
            # bsearch for qh ~ target
            for _ in range(40):
                mid = 0.5*(lo+hi)
                qh_mid = hip_at(mid)
                if qh_mid < target_qh:
                    lo = mid
                else:
                    hi = mid
            return 0.5*(lo+hi)
        
        try:
            if os.path.exists(ik_dims_path):
                with open(ik_dims_path, "r") as f:
                    dims = json.load(f)
                for can_key, dim_key in (('LF','fl'),('RF','fr'),('RL','rl'),('RR','rr')):
                    if can_key in alias and dim_key in dims:
                        cfg_dim = dims[dim_key]
                        Lc = float(cfg_dim.get('Lcoxa', 0.06))
                        Lf = float(cfg_dim.get('Lfemur', 0.08))
                        Lt = float(cfg_dim.get('Ltibia', 0.15))
                        lbl = alias[can_key]
                        # rebuild IK with proper lengths (preserve joint limits)
                        leg = self.kin.legs[lbl]
                        lims=[]
                        for j in leg.joints:
                            lo=j.lower if not math.isnan(j.lower) else -1e9
                            hi=j.upper if not math.isnan(j.upper) else +1e9
                            lims.append((lo,hi))
                        leg.ik = LegIK(Lc, Lf, Lt, tuple(lims))
                        # default x-bias from link lengths: keep some flex at ground_z
                        self.x_bias_by_leg[lbl] = max(0.05, 0.5*Lf + 0.6*Lt)
                self.get_logger().info(f"Loaded IK link lengths from {ik_dims_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load ik_dims_json: {e}")

        # If user provided stance_x_bias_m param, use it (same for all legs)
        x_bias_param = float(self.get_parameter("stance_x_bias_m").get_parameter_value().double_value)
        if x_bias_param > 0.0:
            for lbl in self.leg_labels:
                self.x_bias_by_leg[lbl] = x_bias_param
            self.get_logger().info(f"Using stance_x_bias_m={x_bias_param:.3f} for all legs")
        else:
            # Ensure every leg has some bias; if not set yet, estimate from URDF approx femur/tibia
            for lbl in self.leg_labels:
                if lbl not in self.x_bias_by_leg:
                    # fallback guess
                    Lf = float(self.kin.legs[lbl].lengths[1])
                    Lt = float(self.kin.legs[lbl].lengths[2])
                    self.x_bias_by_leg[lbl] = max(0.05, 0.5*Lf + 0.6*Lt)
            # Auto-tune bias so femur isn't clamped: aim hip pitch near -0.8 rad at neutral height
            for lbl in self.leg_labels:
                try:
                    xb = _solve_bias_for_leg(lbl, target_qh=-0.8, z0=self.g.ground_z)
                    # clamp to a sane range
                    xb = max(0.05, min(xb, 0.35))
                    self.x_bias_by_leg[lbl] = xb
                except Exception as _e:
                    pass
            self.get_logger().info("Per-leg x-bias (auto-tuned, m): " +
                                   ", ".join(f"{lbl}:{self.x_bias_by_leg[lbl]:.3f}" for lbl in self.leg_labels))
            for lbl in self.leg_labels:
                if lbl not in self.x_bias_by_leg:
                    # fallback guess
                    Lf = float(self.kin.legs[lbl].lengths[1])
                    Lt = float(self.kin.legs[lbl].lengths[2])
                    self.x_bias_by_leg[lbl] = max(0.05, 0.5*Lf + 0.6*Lt)
            self.get_logger().info("Per-leg x-bias (m): " +
                                   ", ".join(f"{lbl}:{self.x_bias_by_leg[lbl]:.3f}" for lbl in self.leg_labels))

        # ---- signs vector (dict or list) ----
        self.signs=self._load_signs(signs_path)
        # neutral offsets in same order as joint_names
        self.default_q=[self.kin.neutral_pose.get(jn,0.0) for jn in self.joint_names]

        # ---- Gait phases: use a master oscillator + offsets per leg ----
        gait_mode = self.get_parameter("gait_mode").get_parameter_value().string_value
        if gait_mode == "trot":
            canonical_phase = {'LF':0.00, 'RR':0.00, 'RF':0.50, 'RL':0.50}
            self.g.duty = min(self.g.duty, 0.60)
        elif gait_mode == "pace":
            canonical_phase = {'LF':0.00, 'RF':0.00, 'RL':0.50, 'RR':0.50}
            self.g.duty = min(self.g.duty, 0.60)
        elif gait_mode == "bound":
            canonical_phase = {'LF':0.00, 'RF':0.00, 'RL':0.50, 'RR':0.50}
            self.g.duty = min(self.g.duty, 0.55)
        else:  # crawl (4-beat)
            canonical_phase = {'LF':0.00, 'RR':0.25, 'RF':0.50, 'RL':0.75}
        self.phase_offset = { self.alias[k]: v for k, v in canonical_phase.items() if k in self.alias }
        self.phase_t = 0.0  # master oscillator

        # I/O & state
        self.pub=self.create_publisher(Float64MultiArray, self.out_topic, 10)
        self.sub=self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.timer=self.create_timer(self.dt, self.on_tick)
        self.cmd=Twist()
        self._got_cmd=False
        self._last_active_stamp=self.get_clock().now()
        self._last_q=None

        self.get_logger().info("Joint order (index:name): "+", ".join(f"{i}:{n}" for i,n in enumerate(self.joint_names)))
        self.get_logger().info(f"Publishing Float64MultiArray({len(self.joint_names)}) to {self.out_topic} @ {1.0/self.dt:.1f} Hz")
        self.get_logger().info(f"Gait mode: {gait_mode}, duty={self.g.duty:.2f}")

    # ----- robust signs loader -----
    def _load_signs(self, path: str) -> List[float]:
        if os.path.exists(path):
            try:
                with open(path,"r") as f:
                    data=json.load(f)
                # Preferred: dict of {joint_name: sign}
                if isinstance(data, dict):
                    vec=[]
                    for i, jn in enumerate(self.joint_names):
                        if jn in data:
                            vec.append(float(data[jn]))
                        else:
                            # default: femurs negative; others positive
                            vec.append(-1.0 if ("femur" in jn.lower() or (i%3==1)) else 1.0)
                    self.get_logger().info(f"Loaded signs (by joint name) from {path}")
                    return vec
                # Backward‑compat: 12‑element list (must match current joint order)
                if isinstance(data, list) and len(data)==len(self.joint_names):
                    self.get_logger().info(f"Loaded signs from {path}")
                    return [float(x) for x in data]
            except Exception as e:
                self.get_logger().warn(f"Failed to read {path}: {e}")
        # Fallback: auto signs (femurs negative)
        s=[]
        for idx, jn in enumerate(self.joint_names):
            s.append(-1.0 if ("femur" in jn.lower() or (idx%3==1)) else 1.0)
        self.get_logger().info("Using auto signs (femurs negative). To override, create "+path)
        return s

    # ----- cmd & gait -----
    def on_cmd(self,msg:Twist):
        self.cmd=msg
        if (abs(msg.linear.x)>self.idle_deadband or
            abs(msg.linear.y)>self.idle_deadband or
            abs(msg.angular.z)>self.idle_deadband):
            self._got_cmd=True
            self._last_active_stamp=self.get_clock().now()

    def stance_point(self,s,Sx,Sy,z0): return ((1.0-2.0*s)*(Sx/2.0), Sy, z0)
    def swing_point(self,s,Sx,Sy,z0,h):
        x0,y0,z0a=-(Sx/2.0),Sy,z0; x3,y3,z3=(Sx/2.0),Sy,z0
        def bez(s,a,b,c,d): return (1-s)**3*a+3*(1-s)**2*s*b+3*(1-s)*s**2*c+s**3*d
        x1,y1,z1=x0+(Sx/3.0),Sy,z0+h; x2,y2,z2=x0+(2*Sx/3.0),Sy,z0+h
        return (bez(s,x0,x1,x2,x3), bez(s,y0,y1,y2,y3), bez(s,z0a,z1,z2,z3))

    def on_tick(self):
        if self.wait_for_first_cmd and not self._got_cmd:
            return

        holding = self.hold_on_idle and (
            (self.get_clock().now() - self._last_active_stamp).nanoseconds * 1e-9 > self.idle_timeout
        )

        vx=float(self.cmd.linear.x); vy=float(self.cmd.linear.y); wz=float(self.cmd.angular.z)
        Sx=max(0.0,min(self.g.max_stride, abs(vx)*0.6)); Sy_base=vy*0.2
        freq=self.g.base_freq*max(0.2, min(1.5, abs(vx)/0.2 + 0.3))

        if holding and self._last_q is not None:
            msg=Float64MultiArray(); msg.data=list(self._last_q); self.pub.publish(msg)
            return

        # advance master oscillator
        self.phase_t = (self.phase_t + freq*self.dt) % 1.0

        q_all: List[float] = []
        for lbl in self.leg_labels:
            # per-leg phase from master + offset
            phi_leg = (self.phase_t + self.phase_offset.get(lbl, 0.0)) % 1.0

            # Decide left/right from the first joint name
            names = list(self.leg_joint_order[lbl])
            first = names[0].lower()
            is_left = first.startswith(('fl_','lf_','rl_','lh_'))
            side_scale = (1.0 - 0.3*wz) if is_left else (1.0 + 0.3*wz)
            Sx_leg=Sx*side_scale; Sy_leg=Sy_base if is_left else -Sy_base

            # stance vs swing path around a *neutral horizontal reach bias*
            if phi_leg < self.g.duty:
                s=phi_leg/self.g.duty
                px,py,pz=self.stance_point(s,Sx_leg,Sy_leg,self.g.ground_z)
            else:
                s=(phi_leg-self.g.duty)/max(1e-3,(1.0-self.g.duty))
                px,py,pz=self.swing_point(s,Sx_leg,Sy_leg,self.g.ground_z,self.g.clearance)

            # ADD bias so r = hypot(x,y) exceeds coxa; otherwise xp ~ 0 and femur points down
            x_cmd = self.x_bias_by_leg.get(lbl, 0.10) + px
            y_cmd = py

            leg=self.kin.legs.get(lbl) or next(iter(self.kin.legs.values()))
            qy,qh,qk=leg.ik.solve(x_cmd, y_cmd, pz)

            # apply signs to deltas only, add neutral in configured output order
            deltas = [qy,qh,qk]
            for jn, dq in zip(names, deltas):
                idx = self.joint_names.index(jn)
                q0 = self.kin.neutral_pose.get(jn, 0.0)
                q_all.append(q0 + self.signs[idx]*dq)

        msg=Float64MultiArray(); msg.data=q_all; self.pub.publish(msg)
        self._last_q=list(q_all)

def main():
    rclpy.init()
    node=CPGNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()