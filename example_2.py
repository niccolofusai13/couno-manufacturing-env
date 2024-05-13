import mujoco
import numpy as np
import imageio

def main():
    xml_path = 'gym_xarm/tasks/assets/factory.xml'

    # Load the model from the XML file
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Initialize the renderer
    renderer = mujoco.Renderer(model)


    # Set duration and framerate for the simulation
    duration = 3.8  # in seconds
    framerate = 60  # in frames per second
    num_frames = int(duration * framerate)

    # Prepare for video recording
    frames = []

    # Compute initial conditions
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)

    # Simulate and record frames
    for i in range(num_frames):
        mujoco.mj_step(model, data)
        if i % (framerate // 30) == 0:  # Reduce the number of frames to save
            renderer.update_scene(data)
            image = renderer.render()
            frames.append(image)

    # Save frames to a video file using imageio
    with imageio.get_writer('factory_simulation.mp4', fps=30) as writer:
        for frame in frames:
            writer.append_data((frame * 255).astype(np.uint8))

    print("Video saved as factory_simulation.mp4")


if __name__ == "__main__":
    main()
