import os, sys
import yaml
import torch
import tyro
import viser
import imageio
import numpy as onp
import joblib
import time
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


    
def main(sid: int = 0, result_pkl: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = build_body_model(device)
    smpl_faces = smpl.faces

    # load slam and wham results from ./output/demo/jump/
    # slam_results = joblib.load('./output/demo/jump/slam_results.pth') # trans: xyz, qaut: xyzw

    # wham_results = joblib.load('./output/demo/jump/wham_output.pkl')
    # wham_results = joblib.load('./output/demo/IMG_9732/wham_output.pkl')
    # wham_results = joblib.load('./output/demo/moving_cam/wham_output.pkl')
    # wham_results = joblib.load('./output/demo/walk-2/wham_output.pkl')
    wham_results = joblib.load(result_pkl)
    
    # extract vertices from wham results
    global_output = smpl.get_output(
        body_pose=torch.tensor(wham_results[sid]['pose_world'][:, 3:]).to(device), 
        global_orient=torch.tensor(wham_results[sid]['pose_world'][:, :3]).to(device),
        betas=torch.tensor(wham_results[sid]['betas']).to(device),
        transl=torch.tensor(wham_results[sid]['trans_world']).to(device)
    )

    global_verts = global_output.vertices.cpu().numpy()
    # set the ground to be the minimum y value
    ground_y = global_verts[..., 1].min()
    global_verts[..., 1] = global_verts[..., 1] - ground_y

    timesteps = len(global_verts)

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
    

    # get cameras
    cam_axes_in_world = wham_results[sid]['cam_axes']
    cam_origin_world = wham_results[sid]['cam_origin'] 
    slam_cam_axes_in_world = wham_results[sid]['slam_cam_axes']
    slam_cam_origin_world = wham_results[sid]['slam_cam_origin']
    
    # match the y level
    cam_origin_world[..., 1] = cam_origin_world[..., 1] - ground_y
    # slam_cam_origin_world[..., 1] = slam_cam_origin_world[..., 1] - ground_y

    # scale of DVPO is unknown
    # from slam_cam_origin_world, find an index where the 3d coordinate changes
    # then, use the scale of the first frame to scale the rest of the frames
    # Doesn't work for multiple people
    diff_idx_list = [0]
    for i in range(1, len(cam_axes_in_world)):
        if onp.linalg.norm(slam_cam_origin_world[i] - slam_cam_origin_world[i-1]) > 1e-3:
            diff_idx_list.append(i)

    # average the scale o
    scale_list = []        
    for i in range(1, len(diff_idx_list)):
        scale = onp.linalg.norm(cam_origin_world[diff_idx_list[i-1]] - cam_origin_world[diff_idx_list[i]]) / onp.linalg.norm(slam_cam_origin_world[diff_idx_list[i-1]] - slam_cam_origin_world[diff_idx_list[i]])
        scale_list.append(scale)

    scale = onp.mean(scale_list, axis=0, keepdims=True)

    slam_cam_origin_world = slam_cam_origin_world * scale
    slam_cam_origin_world = slam_cam_origin_world - slam_cam_origin_world[0:1] + cam_origin_world[0:1]

    # trick scaling
    # cam_origin_world[..., 2] = cam_origin_world[..., 2] * 0.3
    # slam_cam_origin_world[..., 2] = slam_cam_origin_world[..., 2] * 0.3

   


    frame_nodes: list[viser.FrameHandle] = []
    for t in range(timesteps):
        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/t{t}", show_axes=False))

        server.scene.add_mesh_simple(
            f"/t{t}/mesh",
            vertices=onp.array(global_verts[t]),
            faces=onp.array(smpl_faces),
            flat_shading=False,
            wireframe=False,
        )
        
        cam_axes_matrix = cam_axes_in_world[t]
        cam_axes_quat =R.from_matrix(cam_axes_matrix).as_quat(scalar_first=True)
        
        server.scene.add_frame(
            f"/t{t}/cam",
            wxyz=cam_axes_quat,
            position=cam_origin_world[t],
            show_axes=True,
            axes_length=0.5,
            axes_radius=0.04,
        )
        
        slam_cam_axes_matrix = slam_cam_axes_in_world[t]
        slam_cam_axes_quat =R.from_matrix(slam_cam_axes_matrix).as_quat(scalar_first=True)
        
        # server.scene.add_frame(
        #     f"/t{t}/slam_cam",
        #     wxyz=slam_cam_axes_quat,
        #     position=slam_cam_origin_world[t],
        #     show_axes=True,
        #     axes_length=0.8,
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