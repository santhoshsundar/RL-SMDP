from gym.envs.registration import register

register(
    id="Rooms-v0", entry_point="rooms.env:FourRoomsEnv",
)
