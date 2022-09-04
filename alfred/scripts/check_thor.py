from ai2thor.controller import Controller
from ai2thor_colab import show_video, plot_frames, side_by_side
from ai2thor.video_controller import VideoController
with VideoController() as vc:
    vc.play(vc.MoveAhead())
    vc.wait(5)
    vc.play(vc.MoveAhead())
    vc.export_video('thor.mp4')

controller = Controller(fullscreen=False)

for _ in range(20):
    action = input('next action:')
    controller.step(action=action)

# controller.start()
# controller.step("MoveLeft")
# controller.step("MoveLeft")
frames = [controller.step(action="RotateLeft", degrees=5).frame for _ in range(360//10)]
# show_video(frames, fps=10)

side_by_side(
    frame1=controller.last_event.frame,
    frame2=controller.step("RotateRight", degrees=30).frame,
    title="RotateRight Result"
)

# while True:
#     controller.step("RotateLeft", degrees=5)
#     controller.step("RotateLeft", degrees=5)
# controller.step("MoveAhead")
# controller.step("MoveAhead")
# controller.step("MoveAhead")
# controller.step("MoveAhead")
# controller.step("MoveLeft")
# controller.step("MoveRight")
