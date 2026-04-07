import os
import glob
import argparse
from pathlib import Path
import numpy as np
import time
import numpyAc
import tqdm
from hydra import initialize, compose
import torch
from torch.utils.data import DataLoader
from dataloaders.encode_dataset import EncodeDataset
from HEM.dataloaders.encode_dataset_hem import EncodeEHEMDataset
from models import *


def encodeNode(pro, octvalue):
    assert octvalue <= 254 and octvalue >= 0
    pre = np.argmax(pro)
    return -np.log2(pro[octvalue] + 1e-07), int(octvalue == pre)


def compress_ehem(batch, outputfile, model, args):
    model.eval()

    context_size = 32768

    ids, pos, extent, pos_mm, data, oct_seq, pt_num, pc, bin_num, z_offset = batch

    pt_num = int(pt_num)
    bin_num = int(bin_num)
    z_offset = int(z_offset)

    levels = oct_seq[0, :, -1, 1].int()
    level_k = oct_seq.shape[2]
    oct_seq = oct_seq[0, :, -1, 0].int()
    oct_len = len(oct_seq)

    elapsed = 0
    proBit = []
    with torch.no_grad():
        # set interval
        interval = context_size
        coding_order = []
        coded_cnt = 0

        for l, (level_data, level_extent, level_pos, level_ids) in enumerate(zip(data, extent, pos, ids)):
            level_data, level_extent, level_pos, level_ids = level_data.cuda(), level_extent.cuda(), level_pos.cuda(), level_ids.cuda()

            probabilities = torch.zeros((level_data.shape[1], model.cfg.model.token_num)).cuda()
            for i in tqdm.trange(0, level_data.shape[1], interval, desc=f"level {l} of {len(data)}"):
                ipt_data = level_data[:, i : i + context_size].long()
                ipt_extent = level_extent[:, i : i + context_size]
                ipt_pos = level_pos[:, :, i : i + context_size]
                node_id = level_ids[0, i : i + context_size]
                start_time = time.time()
                output1, output2 = model(ipt_data, ipt_extent, ipt_pos, enc=True)
                elapsed = elapsed + time.time() - start_time

                if len(probabilities) == 1:
                    # level 0
                    probabilities[0] = torch.softmax(output1[:, -1], 1)
                    coding_order.append(node_id[-1:].detach().cpu().numpy())
                    continue

                p1 = torch.softmax(output1, 2)
                p2 = torch.softmax(output2, 2)
                probabilities[node_id[::2], :] = p1[0]
                probabilities[node_id[1::2], :] = p2[0]
                coding_order.append(node_id[::2].detach().cpu().numpy() + coded_cnt)
                coding_order.append(node_id[1::2].detach().cpu().numpy() + coded_cnt)
            coded_cnt += int(level_data.shape[1])
            proBit.append(probabilities.detach().cpu().numpy())

    proBit = np.vstack(proBit)
    coding_order = np.concatenate(coding_order)

    # entropy coding
    codec = numpyAc.arithmeticCoding()
    if args.cylin:
        outputfile += '_cylin'
    outputfile += '_' + str(len(data)) + '_' + str(bin_num) + '_' + str(z_offset) + '.bin'
    if not os.path.exists(os.path.dirname(outputfile)):
        os.makedirs(os.path.dirname(outputfile))
    _, real_rate = codec.encode(
        proBit[coding_order], oct_seq.numpy().astype(np.int16)[coding_order], outputfile
    )
    torch.save(torch.Tensor(pos_mm), outputfile + '.dat')

    np.set_printoptions(formatter={"float": "{: 0.4f}".format})
    print("outputfile                  :", outputfile)
    print("time(s)                     :", elapsed)
    print("pt num                      :", pt_num)
    print("oct num                     :", oct_len)
    print("total binsize               :", real_rate)
    print("bit per oct                 :", real_rate / oct_len)
    print("bit per pixel               :", real_rate / pt_num)

    return real_rate / pt_num, elapsed, real_rate

def main(args):
    # load ckpt config
    root_path = args.ckpt_path.split("ckpt")[0]
    test_output_path = (
        root_path + "test_output" + args.ckpt_path.split("ckpt")[1][:-1] + "/"
    )
    cfg_path = Path(root_path, ".hydra")
    initialize(config_path=str(cfg_path))
    cfg = compose(config_name="config.yaml")

    model_name = cfg.model.class_name
    model_class = AgentSA
    model = model_class.load_from_checkpoint(checkpoint_path=args.ckpt_path,map_location=torch.device('cuda:0')).cuda()

    test_files = args.test_files

    combine_results = False

    if '*' in test_files[0]:
        test_files = glob.glob(test_files[0])
        test_files = sorted(test_files)
        # calculate averaged results if input is a directory
        combine_results = True

    testset = EncodeEHEMDataset(test_files=test_files, 
                            context_size=model.cfg.model.context_size, 
                            data_type= args.type, 
                            lidar_level=args.lidar_level, 
                            preproc_path=args.preproc_path,
                            )
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    bpps = []
    times = []
    gnp = []
    psnr = []
    chamfer = []
    real_rates = []
       
    
    print("Encoding with", model_name)
    for i, (cur_file, batch) in enumerate(zip(test_files, test_loader)):
        print("Encoding ", cur_file, i, '/', len(test_files))

        bpp, t, real_rate = compress_ehem(batch[:-3], test_output_path + Path(cur_file).stem, model, args)
        times.append(t)
        bpps.append(bpp)
        real_rates.append(real_rate)
        gnp.append(float(batch[-2]))
        psnr.append(float(batch[-1]))
        chamfer.append(float(batch[-3]))
        with open(f'demo/20.txt', 'a') as f:
                f.write(f'{float(batch[-2])} {float(batch[-1])} {float(batch[-3])} {real_rate}\n')
        print(sum(gnp) / (i + 1), sum(psnr) / (i + 1), sum(bpps) / (i + 1), sum(chamfer) / (i + 1), sum(times) / (i + 1))

    if combine_results:
        print('bpp:', float(np.array(bpps).mean()))
        if args.type == 'kitti' or args.type == 'ford' or args.type == 'nuscenes':
            print('sample number:', len(bpps))
            print('times:', float(np.array(times).mean()))
            print('chamfer_dist:', float(np.array(chamfer).mean()))
            print('PSNR:', sum(psnr) / len(psnr))
            out = f'same {args.lidar_level} {args.test_files} {args.ckpt_path}\n' + \
                f'sample number: {len(bpps)}\ntimes: {float(np.array(times).mean())}\n' + \
                f'bpp: {float(np.array(bpps).mean())}\nchamfer_dist: {float(np.array(chamfer).mean())}\n' + \
                f'PSNR: {sum(psnr) / len(psnr)}\n\n'
            with open(f'test_results_same_{args.type}_{args.lidar_level}.txt', 'a') as f:
                f.write(out)
    else:
        print('bpps:', bpps)
        if args.type == 'kitti' or args.type == 'ford' or args.type == 'nuscenes':
            print('sample number:', len(bpps))
            print('times:', float(np.array(times).mean()))
            print('chamfer_dist:', chamfer)
            print('PSNR:', psnr)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="example: outputs/obj/2023-04-28/10-43-45/ckpt/epoch=7-step=64088.ckpt",
    )
    parser.add_argument(
        "--test_files",
        nargs="*",
        default=["data/obj/mpeg/8iVLSF_910bit/boxer_viewdep_vox9.ply"],
    )
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--type", type=str, default='obj', choices=['obj', 'kitti', 'ford','nuscenes'])
    parser.add_argument("--lidar_level", type=int, default=12)
    parser.add_argument("--level_wise", action="store_true")
    parser.add_argument("--cylin", action="store_true")
    parser.add_argument("--spher", action="store_true")
    parser.add_argument("--spher_circle", action="store_true")
    parser.add_argument("--preproc_path", type=str, default="")
    parser.add_argument("--use_scaling", type=bool, default=True)
    parser.add_argument("--use_rope", type=bool, default=True)
    parser.add_argument("--use_hilnet", type=bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
