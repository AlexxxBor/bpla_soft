from pioneer_sdk import Pioneer, Camera
import cv2
import numpy as np
import time
import os

save_dir = "images"
os.makedirs(save_dir, exist_ok=True)


def load_mission(filename):
    mission = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            command = parts[0].upper()
            args = list(map(float, parts[1:]))

            mission.append((command, args))
    return mission


def save_frame(camera, index):
    frame = camera.get_frame()
    if frame is None:
        print("No frame from camera")
        return

    img = cv2.imdecode(
        np.frombuffer(frame, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    filename = f"{save_dir}/frame_{index:03d}.jpg"
    cv2.imwrite(filename, img)
    print(f"Saved {filename}")


def execute_mission(pioneer, camera, mission):
    print("=== START MISSION ===")
    frame_counter = 0

    for command, args in mission:
        print(f"> {command} {args}")

        if command == "ARM":
            pioneer.arm()
            time.sleep(2)

        elif command == "TAKEOFF":
            height = args[0] if args else 1.5
            pioneer.takeoff()
            time.sleep(3)

        elif command == "MOVE":
            x, y, z = args
            pioneer.go_to_local_point_body_fixed(
                x=x, y=y, z=z, yaw=0
            )
            time.sleep(3)

        elif command == "WAIT":
            time.sleep(args[0])

        elif command == "SNAP":
            frame_counter += 1
            save_frame(camera, frame_counter)

        elif command == "SNAP_N":
            n = int(args[0])
            for _ in range(n):
                frame_counter += 1
                save_frame(camera, frame_counter)
                time.sleep(1)

        elif command == "LAND":
            pioneer.land()
            time.sleep(3)

        else:
            print(f"Unknown command: {command}")

    print("=== MISSION COMPLETE ===")


if __name__ == "__main__":
    pioneer = Pioneer()
    camera = Camera()

    try:
        mission = load_mission("mission.txt")
        print("Mission loaded")
        print("s - start mission | q - exit")

        while True:
            frame = camera.get_frame()
            if frame is not None:
                img = cv2.imdecode(
                    np.frombuffer(frame, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                cv2.imshow("pioneer_camera", img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                execute_mission(pioneer, camera, mission)

            elif key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        pioneer.land()
        pioneer.disarm()
        pioneer.close_connection()