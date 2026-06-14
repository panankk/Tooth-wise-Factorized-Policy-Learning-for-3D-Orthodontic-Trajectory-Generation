import torch
import os
import numpy as np
from torch.utils.data import Dataset

# ==========================================
# 🛑 黑名单配置 (保持与 V1/V2/V3 一致)
# ==========================================
RAW_ERROR_CASES = [
    'C01002722632.json', 'C01002722812.json', 'C01002724937.json', 'C01002726883.json', 'C01002728672.json', 
    'C01002737908.json', 'C01002739797.json', 'C01002739809.json', 'C01002740294.json', 'C01002742285.json', 
    'C01002742814.json', 'C01002743376.json', 'C01002748270.json', 'C01002752736.json', 'C01002753894.json', 
    'C01002757078.json', 'C01002760218.json', 'C01002760285.json', 'C01002762513.json', 'C01002764234.json', 
    'C01002770466.json', 'C01002772985.json', 'C01002774123.json', 'C01002774594.json', 'C01002775269.json', 
    'C01002784742.json', 'C01002791706.json', 'C01002792886.json', 'C01002796891.json', 'C01002800505.json', 
    'C01002807805.json', 'C01002809896.json', 'C01002810292.json', 'C01002811406.json', 'C01002811855.json', 
    'C01002812430.json', 'C01002817413.json', 'C01002818931.json', 'C01002828437.json', 'C01002828482.json', 
    'C01002837246.json', 'C01002838124.json', 'C01002838337.json', 'C01002840587.json', 'C01002844772.json', 
    'C01002849621.json', 'C01002722823.json', 'C01002725118.json', 'C01002725736.json', 'C01002727154.json', 
    'C01002727817.json', 'C01002728762.json', 'C01002735973.json', 'C01002736749.json', 'C01002736806.json', 
    'C01002737627.json', 'C01002738954.json', 'C01002742982.json', 'C01002744298.json', 'C01002744513.json', 
    'C01002744715.json', 'C01002745255.json', 'C01002746492.json', 'C01002746762.json', 'C01002746784.json', 
    'C01002747392.json', 'C01002748258.json', 'C01002750688.json', 'C01002751746.json', 'C01002752343.json', 
    'C01002752398.json', 'C01002756167.json', 'C01002761703.json', 'C01002763288.json', 'C01002763514.json', 
    'C01002764458.json', 'C01002764650.json', 'C01002767170.json', 'C01002767967.json', 'C01002770411.json', 
    'C01002772389.json', 'C01002772402.json', 'C01002772660.json', 'C01002775270.json', 'C01002777092.json', 
    'C01002778059.json', 'C01002778116.json', 'C010027781299.json', 'C010027782256.json', 'C010027782469.json', 
    'C010027785002.json', 'C010027787969.json', 'C010027788634.json', 'C010027791605.json', 'C010027791650.json', 
    'C010027792909.json', 'C010027793517.json', 'C010027795801.json', 'C010027796969.json', 'C010027799164.json', 
    'C01002800279.json', 'C01002801236.json', 'C01002805533.json', 'C01002808367.json', 'C01002811237.json', 
    'C01002811934.json', 'C01002821159.json', 'C01002823780.json', 'C01002824747.json', 'C01002830711.json', 
    'C01002831149.json', 'C01002834276.json', 'C01002835435.json', 'C01002836043.json', 'C01002836706.json', 
    'C01002840767.json', 'C01002844996.json', 'C01002846987.json', 'C01002847045.json', 'C01002722788.json', 
    'C01002747516.json', 'C01002774628.json', 'C010027785507.json', 'C01002796914.json', 'C01002803328.json', 
    'C01002815310.json', 'C01002735210.json', 'C01002737403.json', 'C01002756831.json', 'C01002763198.json', 
    'C01002763390.json', 'C01002775708.json', 'C010027789185.json', 'C01002801630.json', 'C01002814870.json', 
    'C01002826165.json', 'C01002725466.json', 'C01002726265.json', 'C01002745749.json', 'C01002757180.json', 
    'C01002766258.json', 'C01002767675.json', 'C01002771849.json', 'C010027780423.json', 'C01002801146.json', 
    'C01002847483.json', 'C01002744748.json', 'C01002776833.json', 'C01002790266.json', 'C01002796879.json', 
    'C01002826705.json', 'C01002827975.json', 'C01002838720.json', 'C01002845920.json'
]

EXISTING_BAD_CASES = [
    "C01002727839", "C01002759801", "C01002797016", "C01002722766", "C01002740980", 
    "C01002780412", "C01002828921", "C01002774033", "C01002774640", "C01002827896", 
    "C01002819910", "C01002812047", "C01002834490", "C01002845571", "C01002848710", 
    "C01002804419"
]

NEW_BAD_CASES = [
    'C01002721136', 'C01002721596', 'C01002722597', 'C01002722621', 'C01002722632', 'C01002722788', 'C01002722812', 
    'C01002722823', 'C01002722878', 'C01002723879', 'C01002723903', 'C01002724353', 'C01002724410', 'C01002724667', 
    'C01002724791', 'C01002724847', 'C01002724937', 'C01002725006', 'C01002725039', 'C01002725118', 'C01002725466', 
    'C01002725488', 'C01002725736', 'C01002725826', 'C01002726265', 'C01002726726', 'C01002726760', 'C01002726883', 
    'C01002727121', 'C01002727154', 'C01002727277', 'C01002727592', 'C01002727772', 'C01002727817', 'C01002727918', 
    'C01002728100', 'C01002728111', 'C01002728313', 'C01002728368', 'C01002728425', 'C01002728672', 'C01002728762', 
    'C01002728874', 'C01002729381', 'C01002729550', 'C01002729695', 'C01002729707', 'C01002729785', 'C01002729987', 
    'C01002730068', 'C01002730372', 'C01002730563', 'C01002730967', 'C01002730989', 'C01002731003', 'C01002731407', 
    'C01002731418', 'C01002731609', 'C01002731698', 'C01002731913', 'C01002732060', 'C01002732161', 'C01002732329', 
    'C01002732374', 'C01002732778', 'C01002732790', 'C01002733184', 'C01002733599', 'C01002733713', 'C01002733746', 
    'C01002734264', 'C01002734905', 'C01002735007', 'C01002735142', 'C01002735186', 'C01002735210', 'C01002735254', 
    'C01002735412', 'C01002735670', 'C01002735681', 'C01002735872', 'C01002735973', 'C01002736198', 'C01002736211', 
    'C01002736288', 'C01002736749', 'C01002736806', 'C01002736840', 'C01002737021', 'C01002737076', 'C01002737403', 
    'C01002737627', 'C01002737908', 'C01002737920', 'C01002738617', 'C01002738909', 'C01002738954', 'C01002738976', 
    'C01002739551', 'C01002739775', 'C01002739797', 'C01002739809', 'C01002740205', 'C01002740294', 'C01002740340', 
    'C01002740407', 'C01002740430', 'C01002740463', 'C01002740474', 'C01002740788', 'C01002740980', 'C01002741082', 
    'C01002741598', 'C01002741701', 'C01002742038', 'C01002742308', 'C01002742454', 'C01002742544', 'C01002742634', 
    'C01002742814', 'C01002742982', 'C01002743174', 'C01002743376', 'C01002743422', 'C01002743444', 'C01002743512', 
    'C01002743523', 'C01002743983', 'C01002744197', 'C01002744298', 'C01002744401', 'C01002744445', 'C01002744513', 
    'C01002744715', 'C01002744748', 'C01002745154', 'C01002745255', 'C01002745705', 'C01002745738', 'C01002745749', 
    'C01002745918', 'C01002745930', 'C01002746087', 'C01002746166', 'C01002746379', 'C01002746492', 'C01002746762', 
    'C01002746784', 'C01002746874', 'C01002747392', 'C01002747415', 'C01002747516', 'C01002747651', 'C01002747662', 
    'C01002747831', 'C01002747943', 'C01002748203', 'C01002748225', 'C01002748258', 'C01002748270', 'C01002748304', 
    'C01002748315', 'C01002748494', 'C01002748506', 'C01002748551', 'C01002749103', 'C01002749136', 'C01002749204', 
    'C01002749259', 'C01002749293', 'C01002749686', 'C01002749822', 'C01002749833', 'C01002749945', 'C01002749989', 
    'C01002750194', 'C01002750374', 'C01002750408', 'C01002750688', 'C01002751061', 'C01002751173', 'C01002751241', 
    'C01002751386', 'C01002751746', 'C01002751960', 'C01002752208', 'C01002752343', 'C01002752398', 'C01002752523', 
    'C01002752556', 'C01002752736', 'C01002752837', 'C01002753120', 'C01002753153', 'C01002753175', 'C01002753401', 
    'C01002753445', 'C01002753748', 'C01002753894', 'C01002753962', 'C01002754019', 'C01002754109', 'C01002754671', 
    'C01002755010', 'C01002755548', 'C01002755717', 'C01002755920', 'C01002755975', 'C01002756167', 'C01002756527', 
    'C01002756831', 'C01002757056', 'C01002757078', 'C01002757180', 'C01002757663', 'C01002758091', 'C01002758248', 
    'C01002758439', 'C01002758574', 'C01002758743', 'C01002758934', 'C01002759014', 'C01002759104', 'C01002759227', 
    'C01002759340', 'C01002759946', 'C01002760218', 'C01002760285', 'C01002760612', 'C01002760937', 'C01002761398', 
    'C01002761567', 'C01002761578', 'C01002761691', 'C01002761703', 'C01002761769', 'C01002762366', 'C01002762513', 
    'C01002762872', 'C01002763198', 'C01002763211', 'C01002763288', 'C01002763390', 'C01002763514', 'C01002763604', 
    'C01002763648', 'C01002763907', 'C01002764144', 'C01002764234', 'C01002764458', 'C01002764649', 'C01002764650', 
    'C01002765303', 'C01002765921', 'C01002766113', 'C01002766135', 'C01002766203', 'C01002766258', 'C01002766663', 
    'C01002766876', 'C01002766933', 'C01002766988', 'C01002767013', 'C01002767170', 'C01002767530', 'C01002767541', 
    'C01002767642', 'C01002767675', 'C01002767787', 'C01002767967', 'C01002768812', 'C01002769172', 'C01002769396', 
    'C01002769701', 'C01002769891', 'C01002769936', 'C01002770286', 'C01002770411', 'C01002770466', 'C01002770556', 
    'C01002770635', 'C010027770770', 'C01002770905', 'C01002770949', 'C01002771096', 'C01002771311', 'C01002771322', 
    'C01002771849', 'C01002772019', 'C01002772086', 'C01002772121', 'C01002772200', 'C01002772299', 'C01002772389', 
    'C01002772402', 'C01002772547', 'C01002772558', 'C01002772660', 'C01002772806', 'C01002772985', 'C010027773010', 
    'C01002773717', 'C01002773751', 'C01002773964', 'C01002774123', 'C01002774235', 'C01002774291', 'C01002774358', 
    'C01002774505', 'C01002774538', 'C01002774550', 'C01002774594', 'C01002774628', 'C01002774673', 'C01002774897', 
    'C01002774921', 'C010027774943', 'C01002775269', 'C01002775270', 'C01002775506', 'C01002775674', 'C01002775708', 
    'C01002775786', 'C01002776079', 'C01002776372', 'C01002776440', 'C01002776541', 'C01002776574', 'C01002776608', 
    'C01002776709', 'C01002776833', 'C01002777092', 'C01002777340', 'C01002777520', 'C01002777812', 'C01002778059', 
    'C01002778093', 'C01002778116', 'C01002778138', 'C01002778240', 'C01002778598', 'C01002778992', 'C01002779375', 
    'C01002779779', 'C010027780153', 'C010027780423', 'C010027780502', 'C010027780737', 'C010027780782', 'C010027780995', 
    'C010027781200', 'C010027781244', 'C010027781299', 'C010027781402', 'C010027781491', 'C010027781985', 'C010027782043', 
    'C010027782256', 'C010027782447', 'C010027782469', 'C010027782515', 'C010027782571', 'C010027782593', 'C010027782717', 
    'C010027783101', 'C010027783145', 'C010027783246', 'C010027783358', 'C010027783448', 'C010027784292', 'C010027784416', 
    'C010027784742', 'C0100277785002', 'C010027785024', 'C010027785215', 'C010027785406', 'C010027785507', 'C010027785596', 
    'C010027785675', 'C010027785822', 'C010027785855', 'C010027785923', 'C010027786126', 'C010027786328', 'C010027786340', 
    'C010027786429', 'C010027786564', 'C010027786609', 'C010027787217', 'C010027787240', 'C010027787509', 'C010027787813', 
    'C010027787857', 'C010027787969', 'C010027788050', 'C010027788320', 'C010027788634', 'C010027789084', 'C010027789107', 
    'C010027789174', 'C010027789185', 'C010027789376', 'C010027789488', 'C010027789994', 'C010027790121', 'C010027790198', 
    'C010027790266', 'C010027790312', 'C010027790446', 'C010027791054', 'C010027791087', 'C010027791605', 'C010027791650', 
    'C010027791706', 'C010027792189', 'C010027792246', 'C010027792257', 'C010027792482', 'C010027792673', 'C010027792853', 
    'C010027792886', 'C010027792909', 'C010027793168', 'C010027793517', 'C010027793663', 'C010027794068', 'C010027794080', 
    'C010027794158', 'C010027794417', 'C010027795250', 'C010027795496', 'C010027795542', 'C010027795801', 'C010027795834', 
    'C01002795878', 'C01002795980', 'C01002796284', 'C01002796385', 'C01002796857', 'C01002796879', 'C01002796891', 
    'C01002796914', 'C01002797038', 'C01002797397', 'C01002797577', 'C01002797779', 'C01002797858', 'C01002798006', 
    'C01002798062', 'C01002798354', 'C01002799164', 'C01002799681', 'C01002799816', 'C01002799917', 'C01002799962', 
    'C01002800033', 'C01002800279', 'C01002800369', 'C01002800448', 'C01002800505', 'C01002800549', 'C01002800695', 
    'C01002801012', 'C01002801146', 'C01002801168', 'C01002801236', 'C01002801258', 'C01002801461', 'C01002801584', 
    'C01002801618', 'C01002801630', 'C01002801753', 'C01002801977', 'C01002802024', 'C01002802271', 'C01002802697', 
    'C01002802798', 'C01002803328', 'C01002803362', 'C01002803429', 'C01002803968', 'C01002804150', 'C01002804396', 
    'C01002804464', 'C01002804475', 'C01002804521', 'C01002804600', 'C01002805184', 'C01002805533', 'C01002805937', 
    'C01002806365', 'C01002807041', 'C01002807625', 'C01002807805', 'C01002807861', 'C01002807894', 'C01002807973', 
    'C01002808008', 'C01002808154', 'C01002808222', 'C01002808367', 'C01002808424', 'C01002808457', 'C01002808727', 
    'C01002808783', 'C01002808840', 'C01002809201', 'C01002809379', 'C01002809380', 'C01002809571', 'C01002809829', 
    'C01002809863', 'C01002809896', 'C01002810135', 'C01002810292', 'C01002810551', 'C01002810573', 'C01002810685', 
    'C01002810887', 'C01002811035', 'C01002811170', 'C01002811237', 'C01002811305', 'C01002811406', 'C01002811428', 
    'C01002811631', 'C01002811800', 'C01002811811', 'C01002811855', 'C01002811934', 'C01002812137', 'C01002812430', 
    'C01002824747', 'C01002824950', 'C01002825265', 'C01002825827', 'C01002826154', 'C01002826165', 'C01002826288', 
    'C01002826334', 'C01002826648', 'C01002826660', 'C01002826705', 'C01002826750', 'C01002827896', 'C01002827975', 
    'C01002828202', 'C01002828325', 'C01002828437', 'C01002828482', 'C01002828572', 'C01002828662', 'C01002829269', 
    'C01002829304', 'C01002829438', 'C01002829832', 'C01002829876', 'C01002830126', 'C01002830182', 'C01002830407', 
    'C01002830441', 'C01002830711', 'C01002831149', 'C01002831273', 'C01002831408', 'C01002831789', 'C01002831813', 
    'C01002832140', 'C01002832375', 'C01002833017', 'C01002833129', 'C01002833400', 'C01002833949', 'C01002833950', 
    'C01002833961', 'C01002834030', 'C01002834153', 'C01002834221', 'C01002834276', 'C01002834300', 'C01002834636', 
    'C01002834669', 'C01002834670', 'C01002834928', 'C01002834973', 'C01002834995', 'C01002835288', 'C01002835378', 
    'C01002835435', 'C01002835884'
]

BLACKLIST_IDS = {case_name.replace('.json', '') for case_name in RAW_ERROR_CASES}
FINAL_BAD_CASES = set(EXISTING_BAD_CASES) | set(NEW_BAD_CASES)

# 0: 切牙，1: 尖牙，2: 前磨牙，3: 磨牙
TOOTH_TYPE_MAP = {
    11:0, 12:0, 21:0, 22:0, 31:0, 32:0, 41:0, 42:0, 
    13:1, 23:1, 33:1, 43:1,                         
    14:2, 15:2, 24:2, 25:2, 34:2, 35:2, 44:2, 45:2, 
    16:3, 17:3, 18:3, 26:3, 27:3, 28:3,             
    36:3, 37:3, 38:3, 46:3, 47:3, 48:3
}

FDI_LIST = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28, 
            48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


class OrthoDataset(Dataset):
    def __init__(self, processed_root, window_size=1):
        """
        V39 最终版 Dataset (支持时空解耦注意力)
        Args:
            processed_root: 数据处理后的根目录
            window_size: 🌟 历史滑动窗口大小。
                         如果为 1，则完全等价于老版本的无时序模型。
                         如果为 5，则输出过去 5 帧的数据以供时序 Transformer 提取惯性。
        """
        self.processed_root = processed_root
        self.window_size = window_size
        
        if not os.path.exists(processed_root):
             raise ValueError(f"Data root {processed_root} does not exist!")
        
        all_dirs = sorted(os.listdir(processed_root))
        
        self.cases = [
            d for d in all_dirs 
            if d not in BLACKLIST_IDS and d not in FINAL_BAD_CASES 
            and os.path.isdir(os.path.join(processed_root, d))
        ]
        print(f"🧹 V39 Strategic Dataset (Temporal Window: {self.window_size}): {len(self.cases)} Cases Loaded.")

    def __len__(self):
        return len(self.cases)

    @staticmethod
    def make_combined_state(curr_p, goal_p, teeth_mask):
        """
        构建单帧的物理共享特征 State [32, 18]
        """
        is_batch = (curr_p.dim() == 3)
        cp = curr_p if is_batch else curr_p.unsqueeze(0)
        gp = goal_p if is_batch else goal_p.unsqueeze(0)
        tm = teeth_mask if is_batch else teeth_mask.unsqueeze(0)

        curr_pos = cp[..., :3]
        mask_expand = tm.unsqueeze(-1)
        sum_pos = (curr_pos * mask_expand).sum(dim=1, keepdim=True)
        count = mask_expand.sum(dim=1, keepdim=True) + 1e-6
        centroid = sum_pos / count
        
        local_pos = curr_pos - centroid 
        diff_pos = gp[..., :3] - cp[..., :3]
        state_pos = torch.cat([local_pos, diff_pos], dim=-1)

        curr_rot = cp[..., 3:9]
        goal_rot = gp[..., 3:9]
        diff_rot = goal_rot - curr_rot 
        state_rot = torch.cat([curr_rot, diff_rot], dim=-1)
        
        state = torch.cat([state_pos, state_rot], dim=-1)
        return state.squeeze(0) if not is_batch else state


    @staticmethod
    def compute_gate_masks_at_step(poses_9d, t):
        """
        与 __getitem__ 中 GT Masks 的 Schmitt Trigger 逻辑保持一致。
        t 表示从 poses_9d[t] -> poses_9d[t+1] 的专家运动 step。
        返回当前 step 的平移/旋转专家激活 mask。
        """
        curr_p = poses_9d[t]
        next_p = poses_9d[t + 1]

        if t == 0:
            real_prev_pos_active = torch.zeros(32, dtype=torch.bool)
            real_prev_rot_active = torch.zeros(32, dtype=torch.bool)
        else:
            prev_p = poses_9d[t - 1]
            prev_dist_pos = torch.norm(curr_p[..., :3] - prev_p[..., :3], dim=-1)
            real_prev_pos_active = prev_dist_pos > 0.05

            prev_dist_rot = torch.norm(curr_p[..., 3:9] - prev_p[..., 3:9], dim=-1)
            real_prev_rot_active = prev_dist_rot > 0.01

        target_pos_delta = next_p[..., :3] - curr_p[..., :3]
        step_dist_pos = torch.norm(target_pos_delta, dim=-1)

        pos_thresh = torch.where(
            real_prev_pos_active,
            torch.tensor(0.03),  # Keep
            torch.tensor(0.1)    # Start
        )
        gt_mask_pos = (step_dist_pos > pos_thresh).float()

        target_rot_delta_raw = next_p[..., 3:9] - curr_p[..., 3:9]
        step_dist_rot = torch.norm(target_rot_delta_raw, dim=-1)

        rot_thresh = torch.where(
            real_prev_rot_active,
            torch.tensor(0.01),
            torch.tensor(0.02)
        )
        gt_mask_rot = (step_dist_rot > rot_thresh).float()

        return gt_mask_pos, gt_mask_rot

    def compute_transition_scores(self, poses_9d, teeth_mask):
        """
        计算每个候选 step 的 transition score。
        score(t) = 平移 gate 状态切换数 + 旋转 gate 状态切换数。
        对 t=0，使用当前 active 数量作为初始启动 transition。
        """
        T = poses_9d.shape[0]
        max_t = T - 1  # valid t: [0, T-2]
        scores = []

        prev_pos_mask = None
        prev_rot_mask = None
        valid_mask = teeth_mask.float()

        for step_t in range(max_t):
            gt_mask_pos, gt_mask_rot = self.compute_gate_masks_at_step(poses_9d, step_t)
            gt_mask_pos = gt_mask_pos * valid_mask
            gt_mask_rot = gt_mask_rot * valid_mask

            if step_t == 0:
                transition_pos = gt_mask_pos.sum()
                transition_rot = gt_mask_rot.sum()
            else:
                transition_pos = torch.abs(gt_mask_pos - prev_pos_mask).sum()
                transition_rot = torch.abs(gt_mask_rot - prev_rot_mask).sum()

            scores.append(transition_pos + transition_rot)
            prev_pos_mask = gt_mask_pos.clone()
            prev_rot_mask = gt_mask_rot.clone()

        if len(scores) == 0:
            return torch.zeros(0)

        return torch.stack(scores).float()

    def sample_timestep(self, poses_9d, teeth_mask):
        """
        Transition-aware mixed timestep sampling:
        - 90%: 原始 uniform timestep sampling。
        - 10%: 从 transition-aware hard steps 中均匀采样。

        Hard step 阈值来自训练集统计结果：transition_total >= 4 (P80)。
        这样只轻微提高启动/停止边界 step 的采样概率，同时尽量保持专家轨迹原始分布。
        """
        T = poses_9d.shape[0]
        max_t = T - 1  # valid t: [0, T-2], because next_p = poses_9d[t+1]

        if max_t <= 1:
            return 0

        # 90%: keep the original uniform sampling.
        if torch.rand(1).item() < 0.9:
            return torch.randint(0, max_t, (1,)).item()

        # 10%: sample from transition-aware hard steps.
        scores = self.compute_transition_scores(poses_9d, teeth_mask)
        if scores.numel() != max_t:
            return torch.randint(0, max_t, (1,)).item()

        hard_indices = torch.nonzero(scores >= 4.0, as_tuple=False).view(-1)

        # If a case has no hard transition step, fall back to uniform sampling.
        if hard_indices.numel() == 0:
            return torch.randint(0, max_t, (1,)).item()

        select_idx = torch.randint(0, hard_indices.numel(), (1,)).item()
        return int(hard_indices[select_idx].item())

    def __getitem__(self, idx):
        case_id = self.cases[idx]
        case_path = os.path.join(self.processed_root, case_id)
        
        try:
            poses_9d = torch.load(os.path.join(case_path, 'poses_9d.pt'), map_location='cpu', weights_only=True)
            shape_emb = torch.load(os.path.join(case_path, 'shape_feature.pt'), map_location='cpu', weights_only=True)
            meta = torch.load(os.path.join(case_path, 'meta.pt'), map_location='cpu', weights_only=True)
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.cases))

        teeth_mask = meta['mask']
        T = poses_9d.shape[0]
        
        # -----------------------------------------------------------
        # 🔥 时序切片采样 (Time Slicing)
        # -----------------------------------------------------------
        # 90% 保持原始 uniform 采样，10% 从 transition-aware hard step 中采样。
        # hard step 定义：专家平移/旋转 gate 状态切换总数 >= 4。
        t = self.sample_timestep(poses_9d, teeth_mask)

        # 🌟 核心升级：提取滑动窗口内的历史帧索引
        # 比如 t=2, window_size=5 -> [max(0, -2), max(0, -1), max(0, 0), max(0, 1), max(0, 2)] 
        # -> [0, 0, 0, 1, 2] (自动使用第0帧做 Padding)
        history_indices = [max(0, t - i) for i in range(self.window_size - 1, -1, -1)]
        history_poses = poses_9d[history_indices] # 形状: [W, 32, 9]

        # 保证以下物理逻辑依然只基于当前的 t 进行，保证兼容性与绝对独立
        curr_p = poses_9d[t]    # 当前状态
        next_p = poses_9d[t+1]  # 下一步状态 (Target)
        goal_p = poses_9d[-1]   # 最终目标状态
        
        # ===========================================================
        # 1. 物理层计算 (Physical Layer - 严格对齐当前 t)
        # ===========================================================
        # A. 基础残差 (Residual)
        diff_pos_vec = goal_p[..., :3] - curr_p[..., :3]
        res_pos = torch.norm(diff_pos_vec, dim=-1) # [32]
        
        diff_rot_vec = goal_p[..., 3:9] - curr_p[..., 3:9]
        res_rot = torch.norm(diff_rot_vec, dim=-1) # [32]
        
        # B. 真实惯性 (Real Inertia - 用于生成 Label)
        if t == 0:
            real_prev_pos_active = torch.zeros(32, dtype=torch.bool)
            real_prev_rot_active = torch.zeros(32, dtype=torch.bool)
        else:
            prev_p = poses_9d[t-1]
            prev_dist_pos = torch.norm(curr_p[..., :3] - prev_p[..., :3], dim=-1)
            real_prev_pos_active = prev_dist_pos > 0.05
            
            prev_dist_rot = torch.norm(curr_p[..., 3:9] - prev_p[..., 3:9], dim=-1)
            real_prev_rot_active = prev_dist_rot > 0.01

        # C. 生成标签 (GT Masks - Schmitt Trigger)
        target_pos_delta = next_p[..., :3] - curr_p[..., :3]
        step_dist_pos = torch.norm(target_pos_delta, dim=-1)
        
        pos_thresh = torch.where(real_prev_pos_active, 
                                 torch.tensor(0.03), # Keep
                                 torch.tensor(0.1))  # Start
        gt_mask_pos = (step_dist_pos > pos_thresh).float()
        
        target_rot_delta_raw = next_p[..., 3:9] - curr_p[..., 3:9]
        step_dist_rot = torch.norm(target_rot_delta_raw, dim=-1)
        
        rot_thresh = torch.where(real_prev_rot_active,
                                 torch.tensor(0.01), 
                                 torch.tensor(0.02))
        gt_mask_rot = (step_dist_rot > rot_thresh).float()

        # ===========================================================
        # 2. 战略特征工程 (V14 Strategic Logic)
        # ===========================================================
        
        # --- A. 身份 (Who am I?) ---
        type_ids_np = np.array([TOOTH_TYPE_MAP.get(fdi, 3) for fdi in FDI_LIST])
        type_ids = torch.tensor(type_ids_np, dtype=torch.long)
        
        feat_my_type = torch.zeros(32, 4)
        feat_my_type.scatter_(1, type_ids.unsqueeze(1), 1.0) # [32, 4] One-Hot
        
        # --- B. 环境信号 (Group Completion Rate) ---
        is_finished_pos = (res_pos < 0.2).float() 
        is_finished_rot = (res_rot < 0.05).float()
        
        group_rates_pos = []
        group_rates_rot = []
        
        for type_idx in range(4): # Inc, Can, Pre, Mol
            group_mask = (type_ids == type_idx).float()
            total_in_group = group_mask.sum() + 1e-6
            
            rate_pos = (is_finished_pos * group_mask).sum() / total_in_group
            rate_rot = (is_finished_rot * group_mask).sum() / total_in_group
            
            group_rates_pos.append(rate_pos)
            group_rates_rot.append(rate_rot)
            
        feat_group_pos = torch.stack(group_rates_pos).unsqueeze(0).expand(32, 4)
        feat_group_rot = torch.stack(group_rates_rot).unsqueeze(0).expand(32, 4)
        
        # --- C. 惯性特征 (Inertia with Dropout) ---
        feat_prev_active_pos = real_prev_pos_active.float()
        feat_prev_active_rot = real_prev_rot_active.float()
        
        if torch.rand(1).item() < 0.3: 
            drop_mask = (torch.rand(32) < 0.5) 
            feat_prev_active_pos = feat_prev_active_pos * (1 - drop_mask.float() * gt_mask_pos)
            
        if torch.rand(1).item() < 0.3:
            drop_mask = (torch.rand(32) < 0.5)
            feat_prev_active_rot = feat_prev_active_rot * (1 - drop_mask.float() * gt_mask_rot)

        # ===========================================================
        # 3. 组装返回数据 (Strict Decoupling)
        # ===========================================================
        
        # 🌟 核心升级：将过去的 W 帧全部转化为物理特征
        history_states = []
        for i in range(self.window_size):
            hist_state = self.make_combined_state(history_poses[i], goal_p, teeth_mask)
            history_states.append(hist_state)
        
        # 堆叠成时序序列张量 [W, 32, 18]
        states_seq = torch.stack(history_states) 
        
        # 组装战略向量 (依然只基于当前帧，完全独立于时序机制)
        strat_vec_pos = torch.cat([
            feat_my_type,                       # [32, 4]
            feat_group_pos,                     # [32, 4]
            res_pos.unsqueeze(-1),              # [32, 1]
            feat_prev_active_pos.unsqueeze(-1)  # [32, 1]
        ], dim=-1)

        strat_vec_rot = torch.cat([
            feat_my_type,                       # [32, 4]
            feat_group_rot,                     # [32, 4]
            res_rot.unsqueeze(-1),              # [32, 1]
            feat_prev_active_rot.unsqueeze(-1)  # [32, 1]
        ], dim=-1)

        return {
            'shape': shape_emb.view(32, 1024, 3),
            
            # 🌟 核心升级：输出展平的时序序列
            # 形状：[W, 32*18]
            # 当 W=1 时，形状 [1, 576] 与原版完全等价，完美兼容！
            'input_seq': states_seq.view(self.window_size, -1),
            
            # 🔥 战略通道特征保持原样 [1, 320] Flattened
            'strat_vec_pos': strat_vec_pos.view(1, -1), 
            'strat_vec_rot': strat_vec_rot.view(1, -1), 
            
            # --- 以下部分与原版完全一致，一字不差 ---
            'feat_prev_pos': feat_prev_active_pos.view(1, -1),
            'feat_prev_rot': feat_prev_active_rot.view(1, -1),
            'gt_pos_mu': target_pos_delta.view(1, -1),       
            'gt_rot_mu': (target_rot_delta_raw * 100.0).view(1, -1), 
            'gt_mask_pos': gt_mask_pos.view(1, -1),
            'gt_mask_rot': gt_mask_rot.view(1, -1),
            'timestep': torch.tensor([float(t)]),     
            'teeth_mask': teeth_mask.view(1, -1),
            'tooth_types': type_ids 
        }