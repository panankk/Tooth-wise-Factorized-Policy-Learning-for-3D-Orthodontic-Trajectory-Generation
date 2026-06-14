import numpy as np
import os
import fcl
import open3d as o3d
from tqdm import tqdm
import json
import sys
import torch
from scipy.spatial.transform import Rotation as R

# ==========================================
# ⚙️ 全局配置
# ==========================================
CONFIG = {
    "max_eval_cases": 1000,
    "compare_expert": True, # 必须开启以计算步数差
    "inference_root": "/home/pa/version-final/inference results-news/inferece results-ab07-new",    
    "expert_root": "/home/pa/version4/data/test_data", 
    "hull_root": "/home/pa/version4/data/hull_512",
    "remove_json": "/home/pa/version4/remove_idx_summary.json",
    "limits": {
        "trans_max": 0.5,
        "rot_max_deg": 3.0,
        "collision_depth": 0.3,
    },
    "up_ids": [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27],
    "down_ids": [47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37],
    "tf_fcl": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),
    "is_expert_ddim": False 
}

FULL_IDS = CONFIG["up_ids"] + CONFIG["down_ids"]

# 黑名单 (已缩减，请保持您原始脚本的完整列表)
ERROR_CASES = ['C01002722632.json', 'C01002722812.json', 'C01002724937.json', 'C01002726883.json'] 
BAD_CASES = ["C01002727839", "C01002759801", "C01002797016"]

# ==========================================
# 1. 工具函数
# ==========================================
def quat_to_matrix_numpy(quat):
    q = np.array(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 1e-6: q /= norm
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
    ])

def get_angle_error_single(q1, q2):
    m1 = quat_to_matrix_numpy(q1)
    m2 = quat_to_matrix_numpy(q2)
    rm = np.matmul(m1, m2.T)
    tr = np.trace(rm)
    return np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))

def get_collision_pairs():
    pairs = []
    for i in range(1, len(FULL_IDS)):
        pairs.append((i - 1, i, FULL_IDS[i-1], FULL_IDS[i]))
    return pairs

# ==========================================
# 2. 引擎与计算类 (保持逻辑一致)
# ==========================================
class CollisionEngine:
    def __init__(self, sample_name, hull_root):
        self.objs = {}
        for fdi in FULL_IDS:
            path = os.path.join(hull_root, f"{sample_name}_{fdi}.ply")
            self.objs[fdi] = None
            if os.path.exists(path):
                try:
                    mesh = o3d.io.read_triangle_mesh(path)
                    if not mesh.is_empty():
                        bvh = fcl.BVHModel()
                        bvh.beginModel()
                        bvh.addSubModel(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
                        bvh.endModel()
                        self.objs[fdi] = bvh
                except: pass

    def check_pose_dict(self, pose_dict, remove_indices, pairs):
        step_coll = 0
        TF_FCL = CONFIG["tf_fcl"]
        for idx_a, idx_b, id_a, id_b in pairs:
            if (idx_a in remove_indices) or (idx_b in remove_indices): continue
            if self.objs.get(id_a) is None or self.objs.get(id_b) is None: continue
            str_a, str_b = str(id_a), str(id_b)
            if str_a not in pose_dict or str_b not in pose_dict: continue
            d_a, d_b = pose_dict[str_a], pose_dict[str_b]
            t_a, t_b = TF_FCL @ np.array(d_a[:3]), TF_FCL @ np.array(d_b[:3])
            r_a, r_b = TF_FCL @ quat_to_matrix_numpy(d_a[3:]), TF_FCL @ quat_to_matrix_numpy(d_b[3:])
            obj_a = fcl.CollisionObject(self.objs[id_a], fcl.Transform(r_a, t_a))
            obj_b = fcl.CollisionObject(self.objs[id_b], fcl.Transform(r_b, t_b))
            req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True)
            res = fcl.CollisionResult()
            fcl.collide(obj_a, obj_b, req, res)
            if res.is_collision and res.contacts:
                depths = [c.penetration_depth for c in res.contacts]
                if (sum(depths)/len(depths)) > CONFIG["limits"]["collision_depth"]:
                    step_coll += 1
        return step_coll

class MetricCalculator:
    def __init__(self, sample_name, remove_indices):
        self.engine = CollisionEngine(sample_name, CONFIG["hull_root"])
        self.remove_indices = remove_indices
        self.pairs = get_collision_pairs()
        
    def eval_sequence(self, sequence_list):
        T = len(sequence_list)
        if T < 2: return 0, 0, 0, 0, 0
        sT, sR, vio, coll = 0.0, 0.0, 0, 0
        for t in range(1, T):
            prev, curr = sequence_list[t-1], sequence_list[t]
            coll += self.engine.check_pose_dict(curr, self.remove_indices, self.pairs)
            for idx, fdi in enumerate(FULL_IDS):
                if idx in self.remove_indices: continue
                fid = str(fdi)
                if fid not in prev or fid not in curr: continue
                dt = np.linalg.norm(np.array(curr[fid][:3]) - np.array(prev[fid][:3]))
                dr = get_angle_error_single(prev[fid][3:], curr[fid][3:])
                sT += dt
                sR += dr
                if dt > CONFIG["limits"]["trans_max"] or (dr * 180 / np.pi) > CONFIG["limits"]["rot_max_deg"]:
                    if dt > 0.01: vio += 1
        return sT, sR, vio, coll, (T - 1)

# ==========================================
# 3. 数据加载逻辑
# ==========================================
def load_json_case(sample_path, is_ddim=False):
    files = sorted([f for f in os.listdir(sample_path) if f.startswith("step") and f.endswith(".json")],
                   key=lambda x: int(x.replace("step", "").replace(".json", "")))
    seq = []
    for f_name in files:
        with open(os.path.join(sample_path, f_name), 'r') as f:
            data = json.load(f)
            if is_ddim:
                for k, v in data.items(): data[k] = v[:3] + [-1*x for x in v[3:]]
            seq.append(data)
    return seq

def load_txt_case(sample_path):
    files = sorted([f for f in os.listdir(sample_path) if f.endswith(".txt") and "step" in f],
                   key=lambda x: int(x.split(".")[0].split("_")[-1]))
    seq = []
    for f_name in files:
        data = {}
        with open(os.path.join(sample_path, f_name), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts: data[parts[0]] = [float(x) for x in parts[1:]]
        seq.append(data)
    return seq

# ==========================================
# 4. 主评价流程
# ==========================================
def main():
    inf_root = CONFIG["inference_root"]
    exp_root = CONFIG["expert_root"]
    
    # 1. 准备样本列表
    all_samples = sorted([d for d in os.listdir(inf_root) if os.path.isdir(os.path.join(inf_root, d))])
    all_samples = [s for s in all_samples if (f"{s}.json" not in ERROR_CASES) and (s not in BAD_CASES) and (s != "C01002841724")]
    if CONFIG["max_eval_cases"]: all_samples = all_samples[:CONFIG["max_eval_cases"]]

    # 2. 加载缺牙映射
    remove_map = {}
    if os.path.exists(CONFIG["remove_json"]):
        with open(CONFIG["remove_json"], 'r') as f: remove_map = json.load(f)

    # 3. 初始化统计
    results = {
        "inf": {"sumT":[], "sumR":[], "violate":[], "coll":[], "steps": []},
        "exp": {"sumT":[], "sumR":[], "violate":[], "coll":[], "steps": []},
        "abs_step_diff": []
    }

    print(f"🔄 Processing {len(all_samples)} samples...")

    for sample in tqdm(all_samples):
        inf_path = os.path.join(inf_root, sample)
        exp_path = os.path.join(exp_root, sample)
        remove_idx = set(remove_map.get(sample, []))
        
        # --- 推理数据评估 ---
        seq_inf = load_txt_case(inf_path)
        if seq_inf:
            calc = MetricCalculator(sample, remove_idx)
            sT, sR, vio, coll, steps = calc.eval_sequence(seq_inf)
            results["inf"]["sumT"].append(sT)
            results["inf"]["sumR"].append(sR)
            results["inf"]["violate"].append(vio)
            results["inf"]["coll"].append(coll)
            results["inf"]["steps"].append(steps)
            
            # --- 专家数据评估 (如果存在) ---
            if CONFIG["compare_expert"] and os.path.exists(exp_path):
                seq_exp = load_json_case(exp_path, is_ddim=CONFIG["is_expert_ddim"])
                if seq_exp:
                    # 专家指标
                    sTe, sRe, vioe, colle, stepse = calc.eval_sequence(seq_exp)
                    results["exp"]["sumT"].append(sTe)
                    results["exp"]["sumR"].append(sRe)
                    results["exp"]["violate"].append(vioe)
                    results["exp"]["coll"].append(colle)
                    results["exp"]["steps"].append(stepse)
                    
                    # 🚀 计算步数差: |专家步数 - 推理步数|
                    diff = abs(stepse - steps)
                    results["abs_step_diff"].append(diff)

    # --- 打印报告 ---
    print("\n" + "="*85)
    print(f"{'DENTAL ALIGNMENT EVALUATION REPORT':^85}")
    print("="*85)
    print(f"| {'Method':<15} | {'SumT':>10} | {'SumR':>10} | {'Violate':>10} | {'f_coll':>10} | {'Avg Steps':>10} |")
    print("-" * 85)

    def report(name, key):
        data = results[key]
        if not data["sumT"]: return
        n = len(data["sumT"])
        f_coll = np.sum(data["coll"]) / (n * 28.0)
        print(f"| {name:<15} | {np.mean(data['sumT']):>10.2f} | {np.mean(data['sumR']):>10.2f} | {np.mean(data['violate']):>10.2f} | {f_coll:>10.4f} | {np.mean(data['steps']):>10.2f} |")

    report("My Inference", "inf")
    if CONFIG["compare_expert"]:
        report("Expert (GT)", "exp")
        if results["abs_step_diff"]:
            avg_diff = np.mean(results["abs_step_diff"])
            print("-" * 85)
            print(f"✨ Mean Absolute Step Difference (Expert vs Inf): {avg_diff:.4f} steps")
    
    print("="*85)

if __name__ == "__main__":
    main()