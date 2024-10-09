import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle, axis_angle_to_matrix, quaternion_to_matrix
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

def run(cfg,
        video,
        image_folder,
        output_pth,
        network,
        calib=None,
        run_global=True,
        save_pkl=False,
        visualize=False,
        return_y_up=True):
    
    # image_folder is not really working; somehow Python global interpreter lock is not working during slam process
    if image_folder is not None:
        image_paths = sorted(glob(osp.join(image_folder, '*.jpg')))
        length = len(image_paths)
        width, height = cv2.imread(image_paths[0]).shape[:2]
        fps = 30
    else:
        cap = cv2.VideoCapture(video)
        assert cap.isOpened(), f'Faild to load video file {video}'
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    
    # Preprocess
    with torch.no_grad():
        if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and 
                osp.exists(osp.join(output_pth, 'slam_results.pth'))):
            
            detector = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            
            if run_global: 
                if image_folder is not None:
                    slam = SLAMModel(image_folder, output_pth, width, height, calib)
                else:
                    slam = SLAMModel(video, output_pth, width, height, calib)
            else: slam = None
            
            bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            if image_folder is not None:
                for img_path in image_paths:
                    img = cv2.imread(img_path)
                    detector.track(img, fps, length)
                    if slam is not None: slam.track()
                    bar.next()
            else:
                while (cap.isOpened()):
                    flag, img = cap.read()
                    if not flag: break
                    
                    # 2D detection and tracking
                    detector.track(img, fps, length)
                    if slam is not None: slam.track()
                    bar.next()

            tracking_results = detector.process(fps)
            
            if slam is not None: 
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                # Is this bug?
                slam_results[:, 3] = 1.0    # Unit quaternion
            logger.info('2D detection and SLAM complete!')

            # Extract image features
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
            if image_folder is not None:
                tracking_results = extractor.run(image_paths, tracking_results)
            else:   
                tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')
            
            # Save the processed data
            joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
            joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Save processed data at {output_pth}')
        
        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
            slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
    
    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # run WHAM
    results = defaultdict(dict)
    
    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=return_y_up, **kwargs)

                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=return_y_up, **kwargs)

                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2
                
                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=return_y_up)

            else:
                # data
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                
                # inference
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=return_y_up, **kwargs)

            # Hongsuk
            cam_R_world = output['cam_R'].mT
            cam_origin_world = - (output['cam_R'].mT @ output['cam_T'].unsqueeze(-1)).squeeze(-1)
            
            # Move slam results to the world coordinate
            slam_cam_position = slam_results[:, :3] # xyz

            slam_cam_quat = slam_results[:, 3:] # xyzw
            slam_cam_quat = slam_cam_quat[:, [3, 0, 1, 2]]
            slam_cam_matrix = quaternion_to_matrix(torch.tensor(slam_cam_quat)).float().to(output['poses_root_world'].device)
            slam_cam_position = torch.tensor(slam_cam_position).float().to(output['poses_root_world'].device)   

            # Not sure... I didn't check the slam code
            # according this repo's code, it should be camera to world transformation
            # so no need to inverse it
            # slam_cam_matrix = slam_cam_matrix.unsqueeze(0).mT
            # slam_cam_position = - (slam_cam_matrix @ slam_cam_position.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
            
            if return_y_up:
                
                yup2ydown = axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float().to(output['poses_root_world'].device)
                cam_R_world = yup2ydown.mT @ cam_R_world
                cam_origin_world = (yup2ydown.mT @ cam_origin_world.unsqueeze(-1)).squeeze(-1)
                
                slam_cam_matrix = yup2ydown.mT @ slam_cam_matrix
                slam_cam_position = (yup2ydown.mT @ slam_cam_position.unsqueeze(-1)).squeeze(-1)
                
            output['cam_axes'] = cam_R_world
            output['cam_origin'] = cam_origin_world
            output['slam_cam_axes'] = slam_cam_matrix
            output['slam_cam_origin'] = slam_cam_position
                
        # if False:
        if args.run_smplify:
            smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            input_keypoints = dataset.tracking_results[_id]['keypoints']
            pred = smplify.fit(pred, input_keypoints, **kwargs)
            
            with torch.no_grad():
                network.pred_pose = pred['pose']
                network.pred_shape = pred['betas']
                network.pred_cam = pred['cam']
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=return_y_up)

        # ========= Store results ========= #
        pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()
        
        results[_id]['pose'] = pred_pose
        results[_id]['trans'] = pred_trans
        results[_id]['pose_world'] = pred_pose_world
        results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[_id]['frame_ids'] = frame_id
        
        # results[_id]['cam_R'] = pred['cam_R'].cpu().squeeze(0).numpy()
        # results[_id]['cam_T'] = pred['cam_T'].cpu().squeeze(0).numpy()
        results[_id]['cam_axes'] = pred['cam_axes'].cpu().squeeze(0).numpy()
        results[_id]['cam_origin'] = pred['cam_origin'].cpu().squeeze(0).numpy()
        results[_id]['slam_cam_axes'] = pred['slam_cam_axes'].cpu().squeeze(0).numpy()
        results[_id]['slam_cam_origin'] = pred['slam_cam_origin'].cpu().squeeze(0).numpy()
    
    if save_pkl:
        joblib.dump(results, osp.join(output_pth, "wham_output.pkl"))
     
    # Visualize
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo
        with torch.no_grad():
            # TEMP
            run_global = False
            run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, 
                        default='examples/demo_video.mp4', 
                        help='input video path or youtube link')

    parser.add_argument('--image_folder', type=str, 
                        default=None, #'examples/demo_images', 
                        help='input image folder path')

    parser.add_argument('--output_pth', type=str, default='output/demo', 
                        help='output folder to write results')
    
    parser.add_argument('--calib', type=str, default=None, 
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')
    
    parser.add_argument('--save_pkl', action='store_true',
                        help='Save output as pkl file')
    
    parser.add_argument('--run_smplify', action='store_true',
                        help='Run Temporal SMPLify for post processing')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Output folder
    sequence = '.'.join(args.video.split('/')[-1].split('.')[:-1])
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    run(cfg, 
        args.video, 
        args.image_folder, 
        output_pth, 
        network, 
        args.calib, 
        run_global=not args.estimate_local_only, 
        save_pkl=args.save_pkl,
        visualize=args.visualize)
        
    print()
    logger.info('Done !')