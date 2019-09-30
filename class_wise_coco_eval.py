#coding=utf-8
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
import multiprocessing
import os.path as osp

def eval_single_core(tasklist, proc_id):
    final_ret = []
    for cnt,task in enumerate(tasklist):
        cocoGt, cat, resFile =task
        cocoDt = cocoGt.loadRes(resFile)
        cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
        cocoEval.params.catIds = [cat]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        ret = [cat, {}, resFile]
        fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
        val_dict = ret[1]
        for k in range(6):
            val_dict['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
        print("coreid:{} task finish:{}/{}".format(proc_id, cnt, len(tasklist)))
        final_ret.append(ret)
    return final_ret

def eval_multi_core(source_things_list, anngtFile):
    cocoGt = COCO(anngtFile)
    cat_ids = cocoGt.getCatIds()
    tasks = []
    for cat in cat_ids:
        for resFile in source_things_list:
            tasks.append((cocoGt, cat, resFile))

    core_num = multiprocessing.cpu_count()
    split_tasks = np.array_split(tasks, core_num)
    print("Number of cores: {}, tasks per core: {}".format(core_num, len(split_tasks[0])))
    workers = multiprocessing.Pool(processes=core_num)
    processes = []
    for proc_id, tasklist in enumerate(split_tasks):
        p = workers.apply_async(eval_single_core,
                                (tasklist, proc_id))
        processes.append(p)
    ensemble_results = []
    for p in processes:
        ensemble_results.extend(p.get())
    return ensemble_results



if __name__=='__main__':
    source_things_list = [
        # 'source_res/mj_results/epoch_8.pth.segm.json',
        'source_res/mj_results/epoch_8.pth_ms_val.segm.json',
        'source_res/wenyi_results/things.json']
    anngtFile = '../../pano/annotations/pano_instance_val2017.json'

    ensemble_results = eval_multi_core(source_things_list, anngtFile)

    result_collector = {}
    for cat, res, resFile in ensemble_results:
        res_id = source_things_list.index(resFile)
        if cat not in result_collector:
            result_collector[cat] = [0, 0]
        result_collector[cat][res_id] = res['mAP(segm)/'+'IoU=0.5:0.95']
    catids = sorted(result_collector.keys())
    for cat in catids:
        choose = osp.basename(source_things_list[0]) if result_collector[cat][0]>result_collector[cat][1] else osp.basename(source_things_list[1])
        print("cat:{}\t{}\t{} choose:{}".format(cat, result_collector[cat][0], result_collector[cat][1],
                                      choose))

