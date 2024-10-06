import os, sys
import yaml
import torch
import tyro
import viser
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

@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    print("new client!")

    # This will run whenever we get a new camera!
    @client.camera.on_update
    def _(_: viser.CameraHandle) -> None:
        print(f"New camera on client {client.client_id}!")

    # Show the client ID in the GUI.
    gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
    gui_info.disabled = True
    
def main(sid: int = 0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = build_body_model(device)
    smpl_faces = smpl.faces

    # load slam and wham results from ./output/demo/jump/
    # slam_results = joblib.load('./output/demo/jump/slam_results.pth') # trans: xyz, qaut: xyzw
    wham_results = joblib.load('./output/demo/jump/wham_output.pkl')

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
    # slam_results[..., 1] = slam_results[..., 1] - ground_y


    timesteps = len(global_verts)

    # setup viser server
    server = viser.ViserServer()
    import pdb; pdb.set_trace()
    server.scene.world_axes.visible = True
    # server.scene.set_up_direction("+y")
    server.scene.add_grid("ground", width=10, height=10, cell_size=0.5, plane="xz")

    clients = server.get_clients()
    print("Connected client IDs", clients.keys())
    
    # rotate the mesh and camera frame around the y-axis to match the conventional camera system
    # rotate_y_matrix = R.from_euler('y', 180, degrees=True).as_matrix()
    # global_verts = global_verts @ rotate_y_matrix.T
    cam_axes_in_world = wham_results[sid]['cam_axes']
    cam_origin_world = wham_results[sid]['cam_origin'] - ground_y
    
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
        
        """
        # get camera poses in world
        cam_R = wham_results[sid]['cam_R'][t]
        # rotate around y-axis to match the conventional camera system
        # cam_R = rotate_y_matrix @ cam_R
        
        cam_T = wham_results[sid]['cam_T'][t]
        cam_axes_in_world = cam_R.T
        cam_T_in_world = - cam_R.T @ cam_T
        cam_T_in_world[..., 1] = cam_T_in_world[..., 1] - ground_y
        # convert cam_axes_in_world to wxyz using scipy
        cam_axes_in_world = R.from_matrix(cam_axes_in_world).as_quat(scalar_first=True) # wxyz
        """
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