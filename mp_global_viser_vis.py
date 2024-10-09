import os, sys
import yaml
import torch
import tyro
import viser
import imageio
import numpy as onp
import joblib
import time
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from loguru import logger

from configs import constants as _C
from lib.models.smpl import SMPL

def build_body_model(device, batch_size=1, gender='neutral', **kwargs):
    sys.stdout = open(os.devnull, 'w')
    body_model = SMPL(
        model_path=_C.BMODEL.FLDR,
        gender=gender,
        batch_size=batch_size,
        create_transl=False).to(device)
    sys.stdout = sys.__stdout__
    return body_model


def get_color_for_sid(sid):
    # Simple hash function to generate a color
    hash_value = sid * 123456789 + 111111111
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    return (r, g, b )
    
def main(result_pkl: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = build_body_model(device)
    smpl_faces = smpl.faces

    wham_results = joblib.load(result_pkl)
    
    # global_verts_list = []
    # global_y_min_list = []
    data_frames = defaultdict(dict)

    for sid in wham_results.keys():
        # # extract vertices from wham results
        # global_output = smpl.get_output(
        #     body_pose=torch.tensor(wham_results[sid]['pose_world'][:, 3:]).to(device), 
        #     global_orient=torch.tensor(wham_results[sid]['pose_world'][:, :3]).to(device),
        #     betas=torch.tensor(wham_results[sid]['betas']).to(device),
        #     transl=torch.tensor(wham_results[sid]['trans_world']).to(device)
        # )

        # global_verts = global_output.vertices.cpu().numpy()
        # # set the ground to be the minimum y value
        # global_verts_list.append(global_verts)
        
        # ground_y = global_verts[..., 1].min()
        # global_y_min_list.append(ground_y)

        frame_ids = wham_results[sid]['frame_ids']

        for i, frame_id in enumerate(frame_ids):
            data_frames[frame_id][sid] = {
                # 'global_verts': global_verts[i],
                'cam_axes': wham_results[sid]['cam_axes'][i],
                'cam_origin': wham_results[sid]['cam_origin'][i],

                'pose_world': wham_results[sid]['pose_world'][i],
                'trans_world': wham_results[sid]['trans_world'][i],
                'betas': wham_results[sid]['betas'][i],
            }

    # Make sure the frame_id starts from 0 
    min_frame_id = min(data_frames.keys())
    new_data_frames = defaultdict(dict)
    for frame_id in data_frames.keys():
        new_data_frames[frame_id - min_frame_id] = data_frames[frame_id]
    data_frames = new_data_frames

    global_y_min_list = []


    prev_first_sid = min(list(data_frames[0].keys()))
    ref_sid_cam_origin = prev_first_sid_cam_origin = data_frames[0][prev_first_sid]['cam_origin']

    # integrate everyone to one global coordinate per frame
    for frame_id in sorted(data_frames.keys()):
        first_sid = min(list(data_frames[frame_id].keys()))
        first_sid_cam_axes = data_frames[frame_id][first_sid]['cam_axes']
        first_sid_cam_origin = data_frames[frame_id][first_sid]['cam_origin']


        # handle the abrupt change of camera origin
        if first_sid != prev_first_sid:
            ref_sid_cam_origin = prev_first_sid_cam_origin + 1 * cam_origin_vel # assuming frame_id change only for one time frame.
            prev_first_sid_cam_origin = first_sid_cam_origin
            prev_first_sid = first_sid
        else:
            cam_origin_vel = first_sid_cam_origin - prev_first_sid_cam_origin
            ref_sid_cam_origin = ref_sid_cam_origin + 1 * cam_origin_vel # assuming frame_id change only for one time frame.
            prev_first_sid_cam_origin = first_sid_cam_origin
            prev_first_sid = first_sid

        # TODO: handle the abrupt change of camera axes
        ref_sid_cam_axes = first_sid_cam_axes
        

        for sid in data_frames[frame_id].keys():
            sid_cam_axes = data_frames[frame_id][sid]['cam_axes']
            sid_cam_origin = data_frames[frame_id][sid]['cam_origin']

            sid_pose_world = data_frames[frame_id][sid]['pose_world']
            sid_trans_world = data_frames[frame_id][sid]['trans_world']
            sid_betas = data_frames[frame_id][sid]['betas']

            sid_global_orient = sid_pose_world[:3]
            sid_body_pose = sid_pose_world[3:]
            sid_global_orient = R.from_rotvec(sid_global_orient).as_matrix()

            # compute relative transformation from sid to first_sid
            # rel_cam_rot = first_sid_cam_axes @ sid_cam_axes.T
            # rel_cam_transl = first_sid_cam_origin - sid_cam_origin
            rel_cam_rot = ref_sid_cam_axes @ sid_cam_axes.T
            rel_cam_transl = ref_sid_cam_origin - sid_cam_origin

            sid_global_orient = rel_cam_rot @ sid_global_orient
            sid_trans_world = rel_cam_transl + sid_trans_world

            # compute global vertices
            sid_global_orient = R.from_matrix(sid_global_orient).as_rotvec()

            global_output = smpl.get_output(
                body_pose=torch.tensor(sid_body_pose).to(device).float().unsqueeze(0), 
                global_orient=torch.tensor(sid_global_orient).to(device).float().unsqueeze(0),
                betas=torch.tensor(sid_betas).to(device).float().unsqueeze(0),
                transl=torch.tensor(sid_trans_world).to(device).float().unsqueeze(0)
            )

            data_frames[frame_id][sid]['global_verts'] = global_output.vertices.cpu().numpy()[0]
            global_y_min = data_frames[frame_id][sid]['global_verts'][..., 1].min()
            global_y_min_list.append(global_y_min)
            
        data_frames[frame_id]['ref_cam_axes'] = ref_sid_cam_axes # first_sid_cam_axes
        data_frames[frame_id]['ref_cam_origin'] = ref_sid_cam_origin # first_sid_cam_origin

    print("Integrated all frames into one global coordinate!")



    # pick groun_y from the first frame
    ground_y = global_y_min_list[0]

    timesteps = max(data_frames.keys()) - min(data_frames.keys()) + 1

    # setup viser server
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.add_grid("ground", width=35, height=35, cell_size=1, plane="xz")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        print("new client!")

        client.camera.position = onp.array([1.14120013, 0.60690449, 5.17581808]) # onp.array([-1, 4, 13])
        client.camera.wxyz = onp.array([-1.75483266e-01,  9.83732196e-01 , 4.88596244e-04, 3.84233121e-02])
            
        # # This will run whenever we get a new camera!
        # @client.camera.on_update
        # def _(_: viser.CameraHandle) -> None:
        #     print(f"New camera on client {client.client_id}!")
        #     print(f"Camera pose for client {id}")
        #     print(f"\tfov: {client.camera.fov}")
        #     print(f"\taspect: {client.camera.aspect}")
        #     print(f"\tlast update: {client.camera.update_timestamp}")
        #     print(f"\twxyz: {client.camera.wxyz}")
        #     print(f"\tposition: {client.camera.position}")
        #     print(f"\tlookat: {client.camera.look_at}")
            
        # Show the client ID in the GUI.
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True
    

    frame_nodes: list[viser.FrameHandle] = []
    for t in range(timesteps):
        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/t{t}", show_axes=False))
        if t not in data_frames.keys():
            continue

            
        cam_axes_matrix = data_frames[t]['ref_cam_axes']
        cam_axes_quat = R.from_matrix(cam_axes_matrix).as_quat(scalar_first=True)
        cam_origin = data_frames[t]['ref_cam_origin'] - ground_y
        server.scene.add_frame(
            f"/t{t}/cam",
            wxyz=cam_axes_quat,
            position=cam_origin,
            show_axes=True,
            axes_length=0.5,
            axes_radius=0.04,
        )

        for sid in data_frames[t].keys():
            if sid == 'ref_cam_axes' or sid == 'ref_cam_origin':
                continue

            global_verts = data_frames[t][sid]['global_verts'] - ground_y
            server.scene.add_mesh_simple(
                f"/t{t}/mesh{sid}",
                vertices=onp.array(global_verts),
                faces=onp.array(smpl_faces),
                flat_shading=False,
                wireframe=False,
                color=get_color_for_sid(sid),
            )

            # cam_axes_matrix = data_frames[t][sid]['cam_axes']
            # cam_axes_quat = R.from_matrix(cam_axes_matrix).as_quat(scalar_first=True)
            
            # server.scene.add_frame(
            #     f"/t{t}/cam{sid}",
            #     wxyz=cam_axes_quat,
            #     position=data_frames[t][sid]['cam_origin'],
            #     show_axes=True,
            #     axes_length=0.5,
            #     axes_radius=0.04,
            # )
            


    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=timesteps - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=15
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % timesteps

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % timesteps

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    render_button = server.gui.add_button("Render motion", disabled=False)
    recording = False
    @render_button.on_click
    def _(event: viser.GuiEvent) -> None:
        nonlocal recording
     
        client = event.client
        if not recording:
            recording = True
            gui_playing.value = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value
            gui_framerate.disabled = False
            
            # images = []
            writer = imageio.get_writer(
                'output.mp4', 
                fps=gui_framerate.value, mode='I', format='FFMPEG', macro_block_size=1
            )
            while True:
                if recording:
                    gui_timestep.value = (gui_timestep.value + 1) % timesteps
                    # images.append(client.camera.get_render(height=720, width=1280))
                    img = client.camera.get_render(height=480, width=720)
                    writer.append_data(img)
                    print('recording...')
                else:
                    print("Recording stopped")
                    gui_framerate.disabled = True
                    writer.close()
                    break
        else:
            recording = False

        

        
    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % timesteps

        time.sleep(1.0 / gui_framerate.value)

if __name__ == "__main__":
    tyro.cli(main)